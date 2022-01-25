# %%
# from undouble import Undouble
# import undouble
# print(dir(undouble))
# print(undouble.__version__)

targetdir = 'D://magweg/101_ObjectCategories'

# %%
# Import library
from undouble import Undouble

# Init with default settings
model = Undouble()

# Import example data
targetdir = model.import_example(data='flowers')

# Importing the files files from disk, cleaning and pre-processing
model.import_data(targetdir)

# Compute image-hash
model.fit_transform(method='phash', hash_size=16)

# Find images with image-hash <= threshold
model.group(threshold=5)

# Plot the images
model.plot()

# Move the images
# model.move()


# %%
from clustimage import Clustimage

# Cluster on image-hash
cl = Clustimage(method='phash', params_hash={'threshold':0, 'hash_size':32})

# Example data
X = cl.import_example(data='mnist')

# Preprocessing, feature extraction and cluster evaluation
results = cl.fit_transform(X, min_clust=4, max_clust=15, metric='euclidean', linkage='ward')

# Scatter
cl.scatter(zoom=3, img_mean=False, text=False)
cl.scatter(zoom=None, img_mean=False, dotsize=20, text=False)
cl.scatter(zoom=3, img_mean=False, text=True, plt_all=True, figsize=(35, 25))

# cl.clusteval.plot()
# cl.plot_unique(img_mean=False)
# cl.plot(min_clust=5)


# %%
import pandas as pd
df = pd.DataFrame(index=[model.results['filenames']], data=model.results['img_hash_hex'], columns=['image_hash_hex'])


# %% Make plots in medium blog
import cv2
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from imagesc import imagesc
from undouble import Undouble

methods = ['ahash', 'phash', 'dhash', 'whash-haar']

for method in methods:
    # Average Hash
    model = Undouble(method=method, hash_size=8)
    # Import example data
    targetdir = model.import_example(data='cat_and_dog')
    # Grayscaling and scaling
    model.import_data(targetdir)
    # Compute image for only the first image.
    hashs = model.compute_hash(model.results['img'][0], to_array=True)
    # Compute the image-hash
    print(method + ' Hash:')
    image_hash = ''.join(hashs[0].astype(int).astype(str).ravel())
    print(image_hash)

    # Import image for plotting purposes
    img_g = cv2.imread(model.results['pathnames'][0], cv2.IMREAD_GRAYSCALE)
    img_r = cv2.resize(img_g, (8, 8), interpolation=cv2.INTER_AREA)

    # Make the figure
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax[0][0].imshow(model.results['img'][0][..., ::-1])
    ax[0][0].axis('off')
    ax[0][0].set_title('Source image')
    ax[0][1].imshow(img_g, cmap='gray')
    ax[0][1].axis('off')
    ax[0][1].set_title('grayscale image')
    ax[1][0].imshow(img_r, cmap='gray')
    ax[1][0].axis('off')
    ax[1][0].set_title('grayscale image, size %.0dx%.0d' %(8, 8))
    ax[1][1].imshow(hashs[0].reshape(8, 8), cmap='gray')
    ax[1][1].axis('off')
    ax[1][1].set_title(method + ' function')

    # Compute hash for the 10 images.
    hashs = model.compute_hash(model, to_array=False)


# %% Make plots in medium
from undouble import Undouble
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
methods = ['ahash', 'phash', 'dhash', 'whash-haar']
colors = {'ahash': 'k', 'phash': 'r', 'dhash': 'g', 'whash-haar': 'b'}

# Initialize model
model = Undouble()
# Import example data
targetdir = 'D://magweg/101_ObjectCategories'
# targetdir = model.import_example(data='flowers')
# Importing the files files from disk, cleaning and pre-processing
model.import_data(targetdir)

for method in methods:
    # Compute image-hash
    model.fit_transform(method=method)

    groups = []
    files = []
    thresholds = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    for threshold in thresholds:
        model.group(threshold=threshold)
        groups.append(model.results['stats']['groups'])
        files.append(model.results['stats']['files'])

    plt.plot(thresholds, groups, label=method+ ' groups', c=colors[method], linestyle='dashed', linewidth=2)
    plt.plot(thresholds, files, label=method+ ' images', c=colors[method], linewidth=2)
    plt.grid(True)
    plt.xlabel('image-hash thresholds')
    plt.ylabel('Number of detected groups/images')
    plt.xticks(thresholds)
    plt.legend()

# %%
