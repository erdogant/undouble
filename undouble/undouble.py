"""Python package undouble is to detect (near-)identical images."""

import os
import logging
import numpy as np
from tqdm import tqdm
from clustimage import Clustimage
import datazets as dz
import clustimage.clustimage as cl
import shutil
import cv2
import copy
from ismember import ismember
import matplotlib.pyplot as plt

logger = logging.getLogger('')
[logger.removeHandler(handler) for handler in logger.handlers[:]]
console = logging.StreamHandler()
formatter = logging.Formatter('[undouble] >%(levelname)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger()


class Undouble():
    """Detect duplicate images.

    Python package undouble is to detect (near-)identical images based on image hashes.

    The following steps are taken:
        1. Read recursively all images from directory with the specified extensions.
        2. Compute image hash.
        3. Group similar images.
        4. Move if desired.

    Parameters
    ----------
    method : str, (default: 'phash')
        Image hash method.
        * 'ahash': Average hash
        * 'phash': Perceptual hash
        * 'dhash': Difference hash
        * 'whash-haar': Haar wavelet hash
        * 'crop-resistant-hash': Crop resistant hash
    savedir : str, (default: None)
        Directory to read the images.
    hash_size : integer (default: 8)
        The hash_size will be used to scale down the image and create a hash-image of length: hash_size*hash_size.
    ext : list, (default: ['png','tiff','tif', 'jpg', 'jpeg', 'heic'])
        Images with the file extentions are used.
    grayscale : Bool, (default: True)
        Colorscaling the image to gray.
    dim : tuple, (default: (128,128))
        Rescale images. This is required because the feature-space need to be the same across samples.
    verbose : int, (default: 20)
        Print progress to screen. The default is 20.
        10:Debug, 20:Info, 30:Warn 40:Error, 60:None

    Returns
    -------
    Object.
    dict containing keys:
        pathnames : list of str.
            Full path to images that are used in the model.
        filenames : list of str.
            Filename of the input images.

    Example
    -------
    >>> # Import library
    >>> from undouble import Undouble
    >>>
    >>> # Init with default settings
    >>> model = Undouble(method='phash', hash_size=8)
    >>>
    >>> # Import example data
    >>> savedir = model.import_example(data='flowers')
    >>>
    >>> # Importing the files files from disk, cleaning and pre-processing
    >>> model.import_data(savedir)
    >>>
    >>> # Compute image-hash
    >>> model.compute_hash()
    >>>
    >>> # Find images with image-hash <= threshold
    >>> model.group(threshold=0)
    >>>
    >>> # Plot the images
    >>> model.plot()
    >>>
    >>> # Move the images
    >>> model.move_to_dir(filters='location')

    References
    ----------
    * Blog: https://towardsdatascience.com/detection-of-duplicate-images-using-image-hash-functions-4d9c53f04a75
    * Github: https://github.com/erdogant/undouble
    * Documentation: https://erdogant.github.io/undouble/
    * https://content-blockchain.org/research/testing-different-image-hash-functions/

    """

    def __init__(self, method='phash', targetdir='', grayscale=False, dim=(128, 128), hash_size=8, ext=['png','tiff','tif', 'jpg', 'jpeg', 'heic'], verbose=20):
        """Initialize undouble with user-defined parameters."""
        if isinstance(ext, str): ext = [ext]
        # Clean readily fitted models to ensure correct results
        self.clean_init()
        if verbose<=0: verbose=60
        # Store user setting in params
        self.params = {'method': method, 'grayscale': grayscale, 'dim': dim, 'ext': ext, 'hash_size': hash_size, 'verbose': verbose}
        # Initialize the clustimage library
        self.clustimage = Clustimage(method=self.params['method'], grayscale=self.params['grayscale'], ext=self.params['ext'], dim=self.params['dim'], params_hash={'hash_size': hash_size}, verbose=self.params['verbose'])
        # Set the logger
        set_logger(verbose=verbose)

    def import_data(self, targetdir, black_list=['undouble'], return_results=False):
        """Preprocessing.

        Parameters
        ----------
        black_list : list, (default: ['undouble'])
            Exclude directory with all subdirectories from processing.

        Example
        -------
        >>> # Import library
        >>> from undouble import Undouble
        >>> #
        >>> # Init with default settings
        >>> model = Undouble()
        >>> #
        >>> #
        >>> # Import example flower data set
        >>> list_of_filepaths = model.import_example(data='flowers')
        >>> #
        >>> # Read from file names
        >>> model.import_data(list_of_filepaths)
        >>> #
        >>> #
        >>> # Read from directory
        >>> input_directory, _ = os.path.split(input_list_of_files[0])
        >>> model.import_data(input_directory)
        >>> #
        >>> #
        >>> # Import from numpy array
        >>> IMG = model.import_example(data='mnist')
        >>> # Compute hash
        >>> model.compute_hash()
        >>> #
        >>> #
        >>> # Find images with image-hash <= threshold
        >>> model.group(threshold=0)
        >>> #
        >>> # Plot the images
        >>> model.plot()

        Returns
        -------
        model.results

        """
        if isinstance(black_list, str): black_list = [black_list]
        # Set targetdir
        self.params['targetdir'] = targetdir
        # logger.info("Retrieving files from: [%s]" %(self.params['targetdir']))
        # Preprocessing the images the get them in the right scale and size etc
        self.results = self.clustimage.import_data(self.params['targetdir'], black_list=black_list)
        # Remove keys that are not used.
        if 'labels' in self.results: self.results.pop('labels')
        if 'xycoord' in self.results: self.results.pop('xycoord')
        # Return
        if return_results:
            return self.results

    def compute_hash(self, method=None, hash_size=8, return_dict=False):
        """Compute the hash for each image.

        Parameters
        ----------
        method : str, (default: 'phash')
            Image hash method.
            * 'ahash': Average hash
            * 'phash': Perceptual hash
            * 'dhash': Difference hash
            * 'whash-haar': Haar wavelet hash
            * 'crop-resistant-hash' : Crop resistance hash
        hash_size : integer (default: 8)
            The hash_size will be used to scale down the image and create a hash-image of length: hash_size*hash_size.

        Returns
        -------
        None.

        """
        if len(self.results['img'])==0:
            logger.warning('No images')
            return

        # Set parameters
        if method is not None:
            self.params['method'] = method
            self.clustimage.params['method'] = method
        if method=='whash-haar':
            if (np.ceil(np.log2(hash_size)) != np.floor(np.log2(hash_size))):
                logger.error('hash_size should be power of 2 (8, 16, 32, 64, ...)')
                return None
        if method=='crop-resistant-hash':
            logger.info('Hash size is set to 8 for crop-resistant and can not be changed.')
            hash_size=8
        if hash_size is not None:
            self.params['hash_size'] = hash_size
            self.clustimage.params_hash['hash_size'] = hash_size
        # Set hash parameters
        self.clustimage.params_hash = cl.get_params_hash(self.params['method'], self.clustimage.params_hash)

        # Compute image-hash features
        self.results['img_hash_bin'] = np.array(list(map(self.clustimage.compute_hash, tqdm(self.results['img'], disable=disable_tqdm()))))
        self.results['img_hash_hex'] = self.bin2hex()

        # Build adjacency matrix for the image-hash based on nr. of differences
        logger.info('Compute adjacency matrix [%gx%g] with absolute differences based on the image-hash of [%s].' %(self.results['img_hash_bin'].shape[0], self.results['img_hash_bin'].shape[0], self.params['method']))
        self.results['adjmat'] = (self.results['img_hash_bin'][:, None, :] != self.results['img_hash_bin']).sum(2)
        # An alternative solution
        # self.result = np.not_equal(self.results['img_hash_bin'][:, None, :], self.results['img_hash_bin']).sum(2)


        # Remove keys that are not used.
        if 'labels' in self.results: self.results.pop('labels')
        if 'xycoord' in self.results: self.results.pop('xycoord')
        if 'feat' in self.results: self.results.pop('feat')
        if return_dict:
            return self.results

    def group(self, threshold=0, return_dict=False):
        """Find similar images using the hash signatures.

        Parameters
        ----------
        threshold : float, (default: 0)
            Threshold on the hash value to determine similarity.

        Returns
        -------
        None.

        """
        if len(self.results['img'])==0:
            logger.warning('No images')
            return None
        if self.results['img_hash_bin'] is None:
            logger.warning('Can not group similar images because no features are present. Tip: Use the compute_hash() function first.')
            return None
        if self.results['pathnames'] is None:
            logger.warning("Can not group images because results['pathnames']=None. This happen due to the use of 'clean_files' functionality.")
            return None

        # Make sets of images that are similar based on the minimum defined threshold.
        pathnames, indexes, thresholds = [], [], []

        # Only the files that exists (or are not moved in an previous action)
        _, idx = get_existing_pathnames(self.results['pathnames'])
        if np.sum(idx==False)>0:
            logger.warning('Files that are moved in a previous action are kept untouched.')
        paths = self.results['pathnames'][idx]
        feat = copy.deepcopy(self.results['adjmat'])
        feat = feat[idx, :]
        feat = feat[:, idx]

        # Extract similar images with minimum threshold
        for i in tqdm(np.arange(0, feat.shape[0]), disable=disable_tqdm()):
            idx = np.where(feat[i, :]<=threshold)[0]
            if len(idx)>1:
                # if ~np.any(list(map(lambda x: np.all(np.isin(x, idx)), indexes))):
                if ~np.any(list(map(lambda x: np.any(np.isin(x, idx)), indexes))):
                    indexes.append(idx)
                    pathnames.append(paths[idx])
                    thresholds.append(feat[i, idx])

        # Sort on threshold
        for i, _ in enumerate(thresholds):
            isort = np.argsort(thresholds[i])
            indexes[i] = indexes[i][isort]
            pathnames[i] = pathnames[i][isort]
            thresholds[i] = thresholds[i][isort]

        # Sort on directory
        idx = np.argsort(list(map(lambda x: os.path.split(x[0])[0], pathnames)))
        self.results['select_pathnames'] = np.array(pathnames, dtype=object)[idx].tolist()
        self.results['select_scores'] = np.array(thresholds, dtype=object)[idx].tolist()
        self.results['select_idx'] = indexes
        logger.info('[%d] groups with similar image-hash.' %(len(self.results['select_pathnames'])))

        totfiles = np.sum(list(map(len, self.results['select_pathnames'])))
        totgroup = len(self.results['select_pathnames'])
        logger.info('[%d] groups are detected for [%d] images.' %(totgroup, totfiles))
        self.results['stats'] = {'groups': totgroup, 'files': totfiles}

        if return_dict:
            return self.results

    def move_to_dir(self, savedir=None, action='move', overwrite=False, gui=True, filter_on=None):
        """Move images.

        Files are moved that are listed by the group() functionality.

        Parameters
        ----------
        savedir : str, optional
            The base directory where the images will be moved. If None, the images will be moved
            * 'c:/temp/'
            * None: to the parent directory of their current location.
        action : str, 'copy' default
            * 'copy': copy files
            * 'move': move files
        overwrite : Bool, False default
            * True: Overwrite files
            * False: Do not overwrite files
        gui : Bool, True default
            * True: Show GUI for user decision
            * False: No GUI
        filter_on : list, (Default: ['location'])
            'location' : Only move images that are seen in the same directory.

        Returns
        -------
        None.

        """
        # Do some checks and set defaults
        self._check_status()
        if savedir is not None:
            if not os.path.isdir(savedir): raise Exception(logger.error(''))
        # logger.info('Detected images: [%d] across of [%d] groups.' %(totfiles, totgroup))

        if gui:
            # Libraries
            try:
                from tkinter import Tk
                from undouble.gui import Gui
                # Create the root Tkinter window
                # root = Tk()
                # # Initialize the ImageMoverApp
                # app = Gui(root, self.results['select_pathnames'])
                # # Run the Tkinter mainloop
                # root.mainloop()

                for pathnames in self.results['select_pathnames']:
                    # print(pathnames)
                    # Check whether move is allowed
                    # filterOK = filter_checks(pathnames, filter_on)
                    root = Tk()
                    # Initialize the ImageMoverApp
                    app = Gui(root, [pathnames], savedir=savedir, action=action, overwrite=overwrite, logger=logger)
                    if len(app.image_groups[0]) > 1:
                        # Run the Tkinter mainloop
                        root.mainloop()
                    else:
                        root.destroy()
                # Return
                return
            except ImportError:
                raise ImportError("Tkinter is not available in this environment.")
        else:
            totfiles = np.sum(list(map(len, self.results['select_pathnames'])))
            totgroup = len(self.results['select_pathnames'])
            tdir = 'undouble' if savedir is None else savedir
            # answer = input('\n-------------------------------------------------\n>You are at the point of physically moving files.\n-------------------------------------------------\n>[%d] similar images are detected over [%d] groups.\n>[%d] images will be moved to the [%s] subdirectory.\n>[%d] images will be copied to the [%s] subdirectory.\n\n>[C]ontinue moving all files.\n>[W]ait in each directory.\n>[Q]uit\n>Answer: ' %(totfiles, totgroup, totfiles - totgroup, tdir, totgroup, tdir))
            answer = input(
                '\n--------------------------------------------------------------------\n'
                '>This functions allows you to physically move the duplicate files!\n'
                '--------------------------------------------------------------------\n'
                '>[%d] similar images are detected over [%d] groups.\n'
                '>[%d] images will be moved to the [%s] subdirectory.\n'
                '>[%d] images will be copied to the [%s] subdirectory.\n'
                '--------------------------------------------------------------------\n'
                '>[A]utomatically move all file without warnings.\n'
                '>[M]ove files per directory.\n'
                '>[Q]uit\n'
                '--------------------------------------------------------------------\n'
                '>Answer: ' % (totfiles, totgroup, totfiles - totgroup, tdir, totgroup, tdir)
            )
            answer = str.lower(answer)
            if answer == 'q':
                return

            # For each group, check the resolution and location.
            pathmem = ''
            for pathnames in self.results['select_pathnames']:
                curdir = os.path.split(pathnames[0])[0]
                if pathmem != curdir:
                    pathmem = curdir
                    logger.info('Working in dir: [%s]' %(curdir))
                    if answer != 'a':
                        answer = input('--------------------------------------------------------------------\n'
                                       '>Press <enter> to proceed to the next directory.\n'
                                       '>[A]utomatically move all file without warnings.\n'
                                       '>[Q]uit\nAnswer: ')
                        answer = str.lower(answer)
                        if answer == 'q':
                            return

                # Check file exists
                pathnames = np.array(pathnames)
                pathnames = pathnames[list(map(os.path.isfile, pathnames))]

                # Check whether move is allowed
                filterOK = filter_checks(pathnames, filter_on)

                # Move to savedir
                if filterOK:
                    # Sort images on resolution and least amount of blur (best first)
                    pathnames = sort_images(pathnames)['pathnames']
                    # Move to dir
                    filepaths_status = cl.move_files(pathnames, savedir, action=action, overwrite=overwrite)


    def clean_init(self, params=True, results=True):
        """Clean or removing previous results and models to ensure correct working."""
        if hasattr(self, 'results'):
            logger.info('Cleaning previous fitted model results')
            if results and hasattr(self, 'results'): del self.results
            if params and hasattr(self, 'params'): del self.params
            if params and hasattr(self, 'clustimage'): del self.clustimage
        # Store results
        # self.results = {'img':None, 'feat':None, 'xycoord':None, 'pathnames':None, 'labels': None}

    def clean_files(self, clean_tempdir=False):
        """Remove the entire temp directory with all its contents."""
        # Cleaning temp directory

        if os.path.isdir(self.clustimage.params['tempdir']):
            files_in_tempdir = os.listdir(self.clustimage.params['tempdir'])
            _, idx = ismember(files_in_tempdir, self.results['filenames'])
            logger.info('Removing images in temp directory %s', self.clustimage.params['tempdir'])
            for i in idx:
                logger.debug('remove: %s', self.results['pathnames'][i])
                os.remove(self.results['pathnames'][i])
                self.results['filenames'][i]=None
                self.results['pathnames'][i]=None
            if clean_tempdir:
                logger.info('Removing the entire temp directory %s', self.clustimage.params['tempdir'])
                shutil.rmtree(self.clustimage.params['tempdir'])
                self.results['filenames'] = None
                self.results['pathnames'] = None

    def import_example(self, data='flowers', url=None, sep=','):
        """Import example dataset from github source.

        Import one of the datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            Images:
                * 'faces'
                * 'mnist'
            Files with images:
                * 'southern_nebula'
                * 'flowers'
                * 'scenes'
                * 'cat_and_dog'

        url : str
            url link to to dataset.

        Returns
        -------
        list: images

        References
        ----------
            * https://github.com/erdogant/datazets

        """
        df = dz.get(data=data, url=url, sep=sep)

        if data=='mnist' or data=='faces':
            X=df.iloc[:, 1:].values
            y=df['target'].values
            return X, y
        else:
            return df

    def plot_hash(self, idx=None, filenames=None):
        """Plot the image-hash.

        Parameters
        ----------
        idx : list of int, optional
            The index of the images to plot.
        filenames : list of str, optional
            The (list of) filenames to plot.

        Returns
        -------
        fig : Figure
        ax : Axis

        Examples
        --------
        >>> # Import library
        >>> from undouble import Undouble
        >>>
        >>> # Init with default settings
        >>> model = Undouble()
        >>>
        >>> # Import example data
        >>> savedir = model.import_example(data='flowers')
        >>>
        >>> # Importing the files files from disk, cleaning and pre-processing
        >>> model.import_data(r'./undouble/data/flower_images/')
        >>>
        >>> # Compute image-hash
        >>> model.compute_hash(method='phash', hash_size=6)
        >>>
        >>> # Hashes are stored in the result dict.
        >>> model.results['img_hash_bin']
        >>>
        >>> Plot the image-hash for a set of indexes
        >>> model.plot_hash(idx=[0, 1])
        >>>
        >>> Plot the image-hash for a set of filenames
        >>> filenames = model.results['filenames'][0:2]
        >>> filenames = ['0001.png', '0002.png']
        >>> model.plot_hash(filenames=filenames)
        >>>

        """
        if idx is None and filenames is None:
            logger.error('You must either specify [idx] or [filenames] as input.\nExample: model.plot_hash(filenames=["01.png", "02.png"])')
            return None, None
        if idx is not None:
            logger.info('Gathering filenames based on input index.')
            if isinstance(idx, int) or isinstance(idx, float): idx = [idx]
            filenames = self.results['filenames'][idx]
        if (filenames is not None) and isinstance(filenames, str):
            filenames = [filenames]

        fig, axs = plt.subplots(len(filenames), 2)
        if len(filenames)==1: axs = [axs]
        for f, ax in zip(filenames, axs):
            logger.info('Creating hash plot for [%s]' %(f))
            idx = np.where(self.results['filenames']==f)[0][0]
            # Make the BGR image and RGB image
            ax[0].imshow(self.results['img'][idx][..., ::-1])
            ax[1].imshow(self.results['img_hash_bin'][idx].reshape(self.params['hash_size'], self.params['hash_size']), cmap='gray')
            title = '[%s]' %(self.results['filenames'][idx])
            ax[0].set_title(title)
            # ax[0].axis('off')
            ax[0].get_xaxis().set_ticks([])
            ax[0].get_yaxis().set_ticks([])
            ax[1].set_title('Image hash')
            # ax[1].axis('off')
            ax[1].get_xaxis().set_ticks([])
            ax[1].get_yaxis().set_ticks([])
        return fig, ax

    def plot(self, cmap=None, figsize=(15, 10), invert_colors=False):
        """Plot the results.

        Parameters
        ----------
        labels : list, (default: None)
            Cluster label to plot. In case of None, all cluster labels are plotted.
        ncols : int, (default: None)
            Number of columns to use in the subplot. The number of rows are estimated based on the columns.
        Colorscheme for the images.
            'gray', 'binary',  None (uses rgb colorscheme)
        show_hog : bool, (default: False)
            Plot the hog features next to the input image.
        min_clust : int, (default: 1)
            Plots are created for clusters with > min_clust samples
        figsize : tuple, (default: (15, 10).
            Size of the figure (height,width).
        invert_colors: Invert colors for the plot.
            True: RGB-> BGR
            False: Keep as is

        Returns
        -------
        None.

        """
        # Do some checks and set defaults
        self._check_status()
        ncols=None
        cmap = cl._set_cmap(cmap, self.params['grayscale'])
        colorscale = 0 if self.params['grayscale'] else 2

        # Plot the clustered images
        if (self.results.get('select_pathnames', None) is not None):
            # Set logger to error only
            # verbose = logger.getEffectiveLevel()
            # set_logger(verbose=50)

            # Run over all labels.
            for i, pathnames in tqdm(enumerate(self.results['select_pathnames']), disable=disable_tqdm()):
                # pathnames = self.results['pathnames'][idx]
                # Check whether file exists.
                pathnames = np.array(pathnames)[list(map(os.path.isfile, pathnames))]
                # Only groups with > 1 images needs to be moved.
                if len(pathnames)>1:
                    # Sort images
                    imgscores = sort_images(pathnames, hash_scores=self.results['select_scores'][i])
                    # Get the images that cluster together
                    imgs = list(map(lambda x: self.clustimage.imread(x, colorscale=colorscale, dim=self.params['dim'], flatten=False), imgscores['pathnames']))
                    # Setup rows and columns
                    _, ncol = self.clustimage._get_rows_cols(len(imgs), ncols=ncols)
                    labels = list(map(lambda x, y, z: 'score: ' + str(int(x)) + ' and blur: ' + str(int(y)) + '\nresolution: ' + str(int(z)), imgscores['hash_scores'], imgscores['blur'], imgscores['resolution']))
                    # Make subplots
                    self.clustimage._make_subplots(imgs, ncol, cmap, figsize, title=("[%s] groups with similar image-hash" %(len(imgscores['pathnames']))), labels=labels, invert_colors=invert_colors)

                # Restore verbose status
                # set_logger(verbose=verbose)
        else:
            logger.warning('Selection on threshold does not exits yet. You firt need to group images with image-hash <= threshold using the function: model.group(threshold=0)')

    def _check_status(self):
        if not hasattr(self, 'results'):
            raise Exception(logger.error('Results missing! Hint: try to first use the model.group() functionality'))

    def compute_imghash(self, img, hash_size=None, to_array=False):
        """Compute image hash per image.

        Parameters
        ----------
        img : Object or RGB-image.
            Image.
        hash_size : integer (default: None)
            The hash_size will be used to scale down the image and create a hash-image of length: hash_size*hash_size.
        to_array : Bool (default: False)
            True: Return the hash-array in the same size as the scaled image.
            False: Return the hash-image vector.

        Examples
        --------
        >>> from undouble import Undouble
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Initialize with method
        >>> model = Undouble(method='ahash')
        >>>
        >>> # Import flowers example
        >>> X = model.import_example(data='flowers')
        >>> imgs = model.import_data(X, return_results=True)
        >>>
        >>> # Compute hash for a single image
        >>> hashs = model.compute_imghash(imgs['img'][0], to_array=False, hash_size=8)
        >>>
        >>> # The hash is a binairy array or vector.
        >>> print(hashs)
        >>>
        >>> # Plot the image using the undouble plot_hash functionality
        >>> model.results['img_hash_bin']
        >>> model.plot_hash(idx=0)
        >>>
        >>> # Plot the image
        >>> fig, ax = plt.subplots(1, 2, figsize=(8,8))
        >>> ax[0].imshow(imgs['img'][0])
        >>> ax[1].imshow(hashs[0])
        >>>
        >>> # Compute hash for multiple images
        >>> hashs = model.compute_imghash(imgs['img'][0:10], to_array=False, hash_size=8)

        Returns
        -------
        imghash : numpy-array
            Hash.

        """
        if hash_size is None: hash_size=self.params['hash_size']
        if hasattr(img, 'results'):
            img = self.results['img']
        elif len(img.shape)<=3:
            img = [img]

        # Compute hash
        hashes = list(map(lambda x: self.clustimage.compute_hash(x, hash_size=hash_size), img))

        # Convert to image-hash
        if to_array:
            hashes = np.array(list(map(lambda x: x.ravel().astype(int), hashes)))
            hashes = np.c_[hashes]
        else:
            hashes = list(map(lambda x: x.reshape(hash_size, hash_size), hashes))

        # Store the hash
        self.results['img_hash_bin'] = hashes
        return hashes

    def bin2hex(self):
        """Binary to hex.

        Returns
        -------
        str
            Hex of image hash.

        """
        if hasattr(self, 'results'):
            return np.array(list(map(lambda x: hex(int(''.join(x.astype(int).astype(str)), 2)), self.results['img_hash_bin'])))
        else:
            logger.warning('Results missing! Hint: try to first use compute_hash()')


# %%
# def move_to_dir(pathnames, savedir, make_moved_filename_consistent=False, action='move'):
#     """Move to target directory.

#     All files that are marked as being "double" are moved. The first image in the array is untouched.

#     Parameters
#     ----------
#     pathnames : list of str
#     savedir : target directory to copy and move the files

#     action : str, 'copy' default
#         * 'copy': copy files
#         * 'move': move files

#     """
#     # Store function
#     shutil_action = shutil.move if action.lower() == 'move' else shutil.copy

#     # Create savedir
#     movedir, dirname, filename, ext = create_savedir(pathnames[0], savedir)

#     # Move all others
#     for i, file in enumerate(pathnames[1:]):
#         if os.path.isfile(file):
#             logger.info(f'Move> {file} -> {movedir}')
#             if make_moved_filename_consistent:
#                 ext = os.path.split(file)[1][-4:].lower()
#                 shutil_action(file, os.path.join(movedir, filename + str(i) + '.' + ext))
#             else:
#                 # Original filename
#                 _, filename1, ext1 = seperate_path(os.path.split(file)[1])
#                 shutil_action(file, os.path.join(movedir, filename1 + ext1))
#         else:
#             logger.info(f'File not found> {file}')



# %%
# def move_to_target_dir(pathnames, savedir, action='move'):
#     """Move to target directory.

#     Move all pathnames to the target directory

#     Parameters
#     ----------
#     pathnames : list of str
#     savedir : target directory to copy and move the files

#     action : str, 'copy' default
#         * 'copy': copy files
#         * 'move': move files

#     """
#     # Store function
#     shutil_action = shutil.move if action.lower() == 'move' else shutil.copy

#     # Move all pathnames to the target directory
#     for filepath in pathnames:
#         # logger.info(f'Moving> {filepath} -> ')
#         if os.path.isfile(filepath):
#             # Create savedir
#             movedir, _, filename, ext = create_savedir(filepath, savedir)
#             # Original filename
#             try:
#                 shutil_action(filepath, os.path.join(movedir, filename + ext))
#             except:
#                 logger.error(f'Error moving file: {filepath}')
#             logger.info(f'{action} {filepath} -> {os.path.join(movedir, filename + ext)}')
#         else:
#             logger.info(f'File not found> {filepath}')


# def create_savedir(pathname, savedir=None):
#     """Create directory.

#     Parameters
#     ----------
#     pathname : str
#         Absolute path location of the image of interest.
#     savedir : str
#         Target directory.

#     Returns
#     -------
#     movedir : str
#         Absolute path to directory.
#     dirname : str
#         Absolute path to directory.
#     filename : str
#         Name of the file.
#     ext : str
#         Extension.

#     """
#     dirname, filename, ext = seperate_path(pathname)
#     # Set the savedir
#     if savedir is None:
#         movedir = os.path.join(dirname, 'undouble')
#     else:
#         movedir = savedir

#     if not os.path.isdir(movedir):
#         logger.debug('Create dir: <%s>' %(movedir))
#         os.makedirs(movedir, exist_ok=True)
#     # Return
#     return movedir, dirname, filename, ext


# %%
def _compute_rank(scores, higher_is_better=True):
    rankscore = np.argsort(scores)[::-1]
    uires = np.unique(scores)
    uirankscore = np.argsort(uires)
    if higher_is_better: uirankscore = uirankscore[::-1]
    rank_res = uirankscore[ismember(scores, uires)[1]]
    return rank_res, rankscore


# %% Sort images.
def sort_images(pathnames, hash_scores=None, sort_first_img=False):
    """Sort images.

    Sort images on the following conditions:
        1. Resolution
        2. Amount of blur

    Parameters
    ----------
    pathnames : list of str.
        Absolute locations to image path.

    Returns
    -------
    list of str
        images sorted on conditions.

    """
    if not sort_first_img:
        # Remove the first image from the list.
        pathname1 = pathnames[0]
        pathnames = pathnames[1:]

    # Compute resolution (higher is better)
    scor_res = np.array(list(map(lambda x: np.prod(cl._imread(x).shape[0:2]), pathnames)))
    ranks_sim, rank_exact = _compute_rank(scor_res, higher_is_better=True)

    # Compute amount of blurr (higher is better)
    scor_blr = np.ceil(np.array(list(map(compute_blur, pathnames))))

    # Within the ordering of the resolution, prefer the least blurry images.
    for r in ranks_sim:
        idx = np.where(r==ranks_sim)[0]
        if len(idx)>1:
            rank_exact[idx] = idx[np.argsort(scor_blr[idx])[::-1]]

    for i in rank_exact:
        logger.debug('%g - %g' %(scor_res[i], scor_blr[i]))

    results = {'pathnames': pathnames[rank_exact], 'resolution': scor_res[rank_exact], 'blur': scor_blr[rank_exact], 'idx': rank_exact}
    # Stack together
    if not sort_first_img:
        scor_res1 = np.prod(cl._imread(pathname1).shape[0:2])
        scor_blr1 = np.ceil(compute_blur(pathname1))
        results['pathnames'] = np.array([pathname1] + list(results['pathnames']))
        results['resolution'] = np.array([scor_res1] + list(results['resolution']))
        results['blur'] = np.array([scor_blr1] + list(results['blur']))
        results['idx'] = [0] + list(results['idx'] + 1)

    hash_scores = np.array(hash_scores)[results['idx']] if hash_scores is not None else np.array([0] *len(results['idx']))
    results['hash_scores'] = hash_scores
    # Return
    return results


# %%
def filter_checks(pathnames, filters):
    """Filter checks.

    Parameters
    ----------
    pathnames : list of str
        pathnames to the images.
    filters : list, (Default: ['location'])
        'location' : Only move images that are seen in the same directory.

    Returns
    -------
    bool
        When all filters are true.

    """
    resOK, locOK = True, True
    # Check nr. of files
    fileOK = True if len(pathnames)>1 else False
    # Get resolution
    if (filters is not None) and np.isin('resolution', filters):
        res = np.array(list(map(lambda x: cl._imread(x).shape[0:2], pathnames)))
        resOK = np.all(np.isin(res[0], res))
    # Get location
    if (filters is not None) and np.isin('location', filters):
        loc = list(map(lambda x: os.path.split(x)[0], pathnames))
        locOK = np.all(np.isin(loc[0], loc))

    return np.all([fileOK, resOK, locOK])


# %%
def compute_blur(pathname):
    """Compute amount of blur in image.

    load the image, convert it to grayscale, and compute the focus measure of
    the image using the Variance of Laplacian method. The returned scores <100
    are generally more blurry.

    Parameters
    ----------
    pathname : str
        Absolute path location to image.

    Returns
    -------
    fm_score : float
        Score the depicts the amount of blur. Scores <100 are generally more blurry.

    """
    # method
    img = cv2.imread(pathname)
    if img is None:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # compute the Laplacian of the image and then return the focus measure, which is simply the variance of the Laplacian
    fm_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # if the focus measure is less than the supplied threshold, then the image should be considered "blurry"
    if fm_score < 100:
        logger.debug('Blurry image> %s' %(pathname))
    # show the image
    # cv2.putText(img, "{}: {:.2f}".format('amount of blur:', fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    # cv2.imshow("Image", img)
    # key = cv2.waitKey(0)
    return fm_score


# %%
def get_existing_pathnames(pathnames):
    """Get existing pathnames.

    Parameters
    ----------
    pathnames : list of str
        pathnames to the images.

    """
    Iloc = np.array(list(map(os.path.isfile, pathnames)))
    return pathnames[Iloc], np.array(Iloc)


# %%
def set_logger(verbose=20):
    """Set the logger for verbosity messages."""
    logger.setLevel(verbose)


# %%
def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)


# %%
def seperate_path(pathname):
    """Seperate path.

    Parameters
    ----------
    pathnames : list of str
        pathnames to the images.

    Returns
    -------
    dirname : str
        directory path.
    filename : str
        filename.
    ext
        Extension.

    """
    dirname, filename = os.path.split(pathname)
    filename, ext = os.path.splitext(filename)
    return dirname, filename, ext.lower()

# %%

# %% Main
# if __name__ == "__main__":
#     import undouble as undouble
#     df = undouble.import_example()
#     out = undouble.compute_hash(df)
#     fig,ax = undouble.plot(out)
