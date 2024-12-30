import os
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from tkinter import messagebox
import undouble

class Gui:
    def __init__(self, root, image_groups, targetdir="undouble", figsize=(400, 300), logger=None):
        # Function to filter out non-existent files
        self.root = root
        self.image_groups = [[path for path in group if os.path.isfile(path)] for group in image_groups]
        self.targetdir = targetdir
        self.current_group_idx = 0
        self.selected_images = []  # Store selected images
        self.image_buttons = []  # Buttons to show images
        self.figsize = figsize

        self.root.title("Image Selector to Undouble")
        self.create_widgets()
        self.display_group()
        if len(self.image_groups)==0:
            logger.info('Nothing to show.')
            return

    def create_widgets(self):
        # Frame for navigation buttons
        self.nav_frame = tk.Frame(self.root)
        self.nav_frame.pack(side="bottom", fill="x")

        # Back Button
        self.back_button = tk.Button(
            self.nav_frame, text="Back", command=self.previous_group
        )
        self.back_button.pack(side="left", padx=10, pady=10)

        # Next Button
        self.next_button = tk.Button(
            self.nav_frame, text="Next", command=self.next_group
        )
        self.next_button.pack(side="left", padx=10, pady=10)

        # Keep Selected Image Button
        self.mark_unselected_button = tk.Button(
            self.nav_frame,
            text="Keep selected image(s) and continue to next image group.",
            command=lambda: [self.mark_unselected_files(), self.root.destroy()],
        )
        self.mark_unselected_button.pack(side="right", padx=10, pady=10)

        # Quit Button
        # self.quit_button = tk.Button(
        #     self.nav_frame, text="Quit", command=self.root.destroy
        # )
        # self.quit_button.pack(side="right", padx=10, pady=10)

        # Frame for image display
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(expand=True, fill="both")

    def display_group(self):
        """Display images for the current group."""
        # Clear the previous images
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        # Update the list so that the files can be shown and are not yet moved
        self.image_groups = [[path for path in group if os.path.isfile(path)] for group in self.image_groups]

        # Get the current group
        if self.current_group_idx < len(self.image_groups):
            current_group = self.image_groups[self.current_group_idx]
            self.image_buttons = []

            for idx, img_path in enumerate(current_group):
                # Get the filename and filesize
                filename = os.path.basename(img_path)
                filesize = os.path.getsize(img_path) # Get file size in bytes
                filesize_mb = round(filesize / (1024 * 1024), 2)
                if filesize_mb < 1:
                    filesize_kb = round(filesize / 1024, 2)
                    file_info = f"{os.path.split(img_path)[0]}\n{filename}\n{filesize_kb} KB"
                else:
                    file_info = f"{os.path.split(img_path)[0]}\n{filename}\n{filesize_mb} MB"

                # Split the file_info text into lines
                # file_info = file_info.split("\n")

                # Load and resize the image
                pil_image = Image.open(img_path)
                pil_image = pil_image.resize(self.figsize)  # Resize for display
                tk_image = ImageTk.PhotoImage(pil_image)

                # Create a grayed-out version of the image
                gray_pil_image = pil_image.convert("L")  # Convert to grayscale
                tk_gray_image = ImageTk.PhotoImage(gray_pil_image)

                # Add text (filename and filesize) to the center of the image
                draw = ImageDraw.Draw(pil_image)
                try:
                    # font = ImageFont.load_default()  # Use default font; you can change this if needed
                    font = ImageFont.truetype("arial.ttf", size=int(pil_image.height // 20))  # Scaled based on image size
                except IOError:
                    font = ImageFont.load_default()

                # Calculate text size using textbbox (returns the bounding box)
                bbox = draw.textbbox((0, 0), file_info, font=font)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

                text_position = ((pil_image.width - text_width) // 2, (pil_image.height - text_height) // 2)
                draw.text(text_position, file_info, fill="white", font=font)

                # Update the Tk image with the new image (with text drawn)
                tk_image = ImageTk.PhotoImage(pil_image)  # Create a new PhotoImage object after drawing text

                # Create a button for each image
                img_button = tk.Button(
                    self.image_frame,
                    image=tk_image,
                    command=lambda idx=idx: self.toggle_selection(idx),
                    bd=1,
                    relief="solid",
                )
                img_button.image = tk_image  # Keep reference to the normal image
                img_button.gray_image = tk_gray_image  # Keep reference to the grayed-out image
                img_button.grid(row=(idx // 3) + 1, column=idx % 3, padx=10, pady=10)

                # Check if the image is already selected
                if img_path not in self.selected_images:
                    img_button.config(image=img_button.gray_image)  # Default to grayed-out

                self.image_buttons.append(img_button)


    def toggle_selection(self, idx):
        """Toggle the selection of an image."""
        current_group = self.image_groups[self.current_group_idx]
        selected_image = current_group[idx]

        if selected_image in self.selected_images:
            # Unselect the image
            self.selected_images.remove(selected_image)
            self.image_buttons[idx].config(
                relief="solid",
                bd=1,
                highlightbackground="#FFFFFF",
                highlightcolor="#FFFFFF",
                image=self.image_buttons[idx].gray_image,  # Switch to grayscale image
            )
        else:
            # Select the image
            self.selected_images.append(selected_image)
            self.image_buttons[idx].config(
                relief="solid",
                bd=5,
                highlightbackground="#880808",
                highlightcolor="#880808",
                image=self.image_buttons[idx].image,  # Switch back to normal image
            )

    def next_group(self):
        """Navigate to the next group."""
        if self.current_group_idx < len(self.image_groups) - 1:
            self.current_group_idx += 1
            self.display_group()

    def previous_group(self):
        """Navigate to the previous group."""
        if self.current_group_idx > 0:
            self.current_group_idx -= 1
            self.display_group()

    def mark_unselected_files(self):
        """Rename unselected files by appending .del to their filenames."""
        unselected_files = []

        for group in self.image_groups:
            for img_path in group:
                if img_path not in self.selected_images:
                    unselected_files.append(img_path)

        # Move to dir
        # print(unselected_files)
        # print('--------------------------------')
        undouble.move_to_target_dir(unselected_files, targetdir=None)

        # Show message box after pressing hte button
        # messagebox.showinfo("Undouble Operation Completed", f"{len(unselected_files)} files are moved to directory: [{self.targetdir}]")

# To run the application
if __name__ == "__main__":
    # Sample image groups for testing (Replace with actual paths)
    image_groups = [
        [
            "path_to_image_1.jpg",  # Replace with actual image paths
            "path_to_image_2.jpg",
        ],
        [
            "path_to_image_3.jpg",
            "path_to_image_4.jpg",
        ],
    ]

    root = tk.Tk()
    app = Gui(root, image_groups)
    root.mainloop()
