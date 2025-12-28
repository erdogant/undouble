# from undouble import Undouble
# import undouble
# print(dir(undouble))
# print(undouble.__version__)

# Import library
from clustimage import Clustimage
import numpy as np
from matplotlib.colors import ListedColormap
from scatterd import scatterd
cmap = ListedColormap(["blue", "red"])

# init
method = 'hog'
cl = Clustimage(method=method, params_hash={'threshold': 0, 'hash_size': 16},)
# ahash
# phash
# dhash
# whash-haar
# whash-db4
# colorhash

# Note that you manually need to download the data from the caltech website and simply provide the directory where the images are stored.
# Download dataset

# Cluster  images in path location.
targetdir = r'C://101_ObjectCategories//'
results = cl.fit_transform(targetdir, min_clust=1, max_clust=20)

# If you want to experiment with a different clustering and/or evaluation approach, use the cluster functionality.
# This will avoid pre-processing, and performing the feature extraction of all images again.
# You can also cluster on the 2-D embedded space by setting the cluster_space parameter 'low'
#
# cluster(cluster='agglomerative', evaluate='dbindex', metric='euclidean', linkage='ward', min_clust=15, max_clust=200, cluster_space='high')

# Cluster evaluation plots such as the Silhouette plot
cl.clusteval.plot(title=f'Method: {method}')
cluster_labels = cl.results['labels']
y=np.array(list(map(lambda x: r'\fake' in x, cl.results['pathnames'])))
cl.results['labels'] = y.astype(int)
# cl.clusteval.scatter(cl.results['xycoord'], s=100)

scatterd(x=cl.results['xycoord'][:,0], y=cl.results['xycoord'][:,1], labels=y, cmap=cmap, title=f'Method: {method}')

# %%


# PCA explained variance plot

cl.pca.plot()

# Dendrogram
cl.dendrogram()

# Plot unique image per cluster
cl.plot_unique(img_mean=False)

# Scatterplot
cl.scatter(dotsize=50, zoom=0.2, img_mean=False)

# Plot images per cluster or all clusters
cl.plot(labels=8)

# %%

# Import library
from undouble import Undouble

# Initialize model
model = Undouble(method='phash', hash_size=8)

# Download dataset here: http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
# targetdir = 'C://101_ObjectCategories'
targetdir = r'C:/101_ObjectCategories//'
# from pathlib import Path
# targetdir.exists()

# Importing the files files from disk, cleaning and pre-processing
model.import_data(targetdir, use_thumbnail_cache=False)

# Compute image-hash
model.compute_hash()

# [undouble] >INFO> Extracting images from: [C://101_ObjectCategories]
# [undouble] >INFO> [9144] files are collected recursively from path: [D://101_ObjectCategories]
# [undouble] >INFO> [9144] images are extracted.
# [undouble] >INFO> Reading and checking images.
# [undouble] >INFO> Reading and checking images.
# 100%|██████████| 9144/9144 [02:14<00:00, 67.95it/s] 
# 100%|██████████| 9144/9144 [00:02<00:00, 3514.21it/s]
# [undouble] >INFO> Compute adjacency matrix [9144x9144] with absolute differences based on the image-hash of [phash].

# Group images that are identical in image-hash, i.e. those with a hash difference of 0.
model.group(threshold=0)

# [undouble] >INFO> [38] groups with similar image-hash.
# [undouble] >INFO> [38] groups are detected for [79] images.

# print([*model.results.keys()])
#   * img: Preprocessed images
#   * pathnames: Absolute path location to image file
#   * filenames: Filename
#   * img_hash_bin: binary image hash
#   * img_hash_hex: hex image hash
#   * adjmat: adjacency matrix containing absolute hash differences between the images
#   * select_pathnames: Selected path locations that have image-hash score <= threshold
#   * select_scores: Image-hash scores of the selected images
#   * stats: Some simple statistics about the number of groups and imges detected.

# Plot the marked images
model.plot()

# Move the files
model.move_to_dir()

# -------------------------------------------------
# >You are at the point of physically moving files.
# -------------------------------------------------
# >[79] similar images are detected over [38] groups.
# >[41] images will be moved to the [undouble] directory.
# >[38] images will be copied to the [undouble] directory.

# >[C]ontinue moving all files.
# >[W]ait in each directory.
# >[Q]uit
# >Answer: W

# [undouble] >INFO> Working in dir: [D://101_ObjectCategories/BACKGROUND_Google]
# ><enter> to proceed to the next directory.
# >[C]ontinue to move all files.
# >[Q]uit.
# >Answer:
    


# %%

import cv2
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from undouble import Undouble

methods = ['ahash', 'dhash', 'whash-haar']

for method in methods:
    # Average Hash
    model = Undouble(method=method, hash_size=8)
    # Import example data
    targetdir = model.import_example(data='cat_and_dog')
    # Grayscaling and scaling
    model.import_data(targetdir)
    # Compute image for only the first image.
    hashs = model.compute_imghash(model.results['img'][0], to_array=False)
    # Compute the image-hash
    image_hash = ''.join(hashs[0].astype(int).astype(str).ravel())
    print(f'{method } Hash:')
    print(f"Binary image hash: {image_hash}")
    print(f"Hex image hash: {(hex(int(image_hash, 2)))}")

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
    ax[1][1].imshow(hashs[0], cmap='gray')
    ax[1][1].axis('off')
    ax[1][1].set_title(method + ' function')
    
    
# %% issue
from undouble import Undouble

# Init with default settings
model = Undouble()

# Importing the files files from disk, cleaning and pre-processing
url_to_images = ['https://erdogant.github.io/datasets/images/flower_images/flower_orange.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_white_1.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_white_2.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_yellow_1.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_yellow_2.png']

# Import into model
model.import_data(url_to_images)

# Compute image-hash
model.compute_hash(method='phash', hash_size=16)

# Find images with image-hash <= threshold
model.group(threshold=0)

# Plot the images
model.plot()


# %%
from undouble import Undouble

# Init with default settings
model = Undouble(grayscale=False,
                 method='phash',
                 hash_size=8,
                 verbose=10,
                 )

# Import example data
# input_list_of_files = model.import_example(data='flowers')
# model.import_data(r'C:\Users\beeld\.conda\envs\env_undouble\Lib\site-packages\datazets\data')
model.import_data(r'D://temp//photos//')
# model.import_data(r'\\NAS_SYNOLOGY\Photo\2023\vluchten')


# Import flowers example
# X = model.import_example(data='scenes')
# Importing the files files from disk, cleaning and pre-processing
# model.import_data(X)

# Compute image-hash
model.compute_hash()

# Find images with image-hash <= threshold
model.group(threshold=10)

# Plot the images
model.plot()

# Move the images
model.move_to_dir(gui=True, action='move', overwrite=False)
# model.move_to_dir(savedir=None, action='move', overwrite=False, gui=True, filter_on=None)


# %% issue #9 homogenious part
from undouble import Undouble

model = Undouble(method='phash')
# Import flowers example
X = model.import_example(data='flowers')
X = X[182:184]
# Import data
# model.import_data(X, return_results=False)
# Compute Hash
# model.compute_hash()
# assert set(model.results.keys())==set(['img', 'url', 'pathnames', 'filenames', 'img_hash_bin', 'img_hash_hex', 'adjmat'])

combination = ('crop-resistant-hash', False, 16, (256, 256))
model = Undouble(method=combination[0], grayscale=combination[1], hash_size=combination[2], dim=combination[3], verbose=40)
# Import data
model.import_data(X, return_results=False)
# Compute Hash
model.compute_hash(return_dict=True)



# %%
# Import library
from undouble import Undouble

# Init with default settings
model = Undouble()

# Import data; Pathnames to the images.
input_list_of_files, y = model.import_example(data='faces')
# input_list_of_files, y = model.import_example(data='mnist')
# input_list_of_files = model.import_example(data='flowers')
# input_list_of_files = model.import_example(data='cat_and_dog')

# Import data from files.
model.import_data(input_list_of_files)

# Compute hash
model.compute_hash()

# Find images with image-hash <= threshold
model.group()

# [undouble] >INFO> [3] groups with similar image-hash.
# [undouble] >INFO> [3] groups are detected for [7] images.

# Plot the images
model.plot()

# Extract the pathnames for each group
for idx_group in model.results['select_idx']:
    print(idx_group)
    print(model.results['pathnames'][idx_group])


# %% Issue: group does not work, Check crop-resistant-hash

# Import library
from undouble import Undouble

# Init with default settings
model = Undouble()

# Import example data
targetdir = model.import_example(data='flowers')

# Importing the files files from disk, cleaning and pre-processing
model.import_data(targetdir)

# Compute image-hash
model.compute_hash(method='whash-haar', hash_size=32)

results = model.group(threshold=5)

# Plot the images
model.plot()

# Plot the hash
model.plot_hash(filenames=model.results['filenames'][model.results['select_idx'][0]])

import pandas as pd
df = pd.DataFrame(index=[model.results['filenames']], data=model.results['img_hash_hex'], columns=['image_hash_hex'])


# %% Issue #7
from undouble import Undouble
import matplotlib.pyplot as plt

# Initialize with method
model = Undouble(method='dhash')

# Import flowers example
# X = model.import_example(data='flowers')
X = model.import_example(data='cat_and_dog')
imgs = model.import_data(X, return_results=True)

# Compute hash for a single image
hashs = model.compute_imghash(imgs['img'][0], to_array=False, hash_size=8)

# The hash is a binairy array or vector.
print(hashs)

# Plot the image using the undouble plot_hash functionality
model.results['img_hash_bin']
model.plot_hash(idx=0)

# Plot the image
fig, ax = plt.subplots(1, 2, figsize=(8,8))
ax[0].imshow(imgs['img'][0])
ax[1].imshow(hashs[0])


# %% Check crop-resistant-hash

# Import library
from undouble import Undouble

# Init with default settings
model = Undouble()

# Import example data
targetdir = model.import_example(data='flowers')

# Importing the files files from disk, cleaning and pre-processing
model.import_data(targetdir[0:100])

# Compute image-hash
model.compute_hash(method='crop-resistant-hash')

# Find images with image-hash <= threshold
results = model.group(threshold=5)
print(model.results.keys())
model.results['select_pathnames'][0]
model.results['select_idx'][0]

# Plot the images
model.plot()

# Plot the hash
model.plot_hash(filenames=model.results['filenames'][model.results['select_idx'][0]])

import pandas as pd
df = pd.DataFrame(index=[model.results['filenames']], data=model.results['img_hash_hex'], columns=['image_hash_hex'])


# %% Import list of images from url adresses

# Import library
from undouble import Undouble

# Init with default settings
model = Undouble()

# Importing the files files from disk, cleaning and pre-processing
url_to_images = ['https://erdogant.github.io/datasets/images/flower_images/flower_orange.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_white_1.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_white_2.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_yellow_1.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_yellow_2.png',
                 'https://www.gardendesign.com/pictures/images/675x529Max/site_3/helianthus-yellow-flower-pixabay_11863.jpg',
                 'https://www.gardendesign.com/pictures/images/675x529Max/site_3/helianthus-yellow-flower-pixabay_11863.jpg']

# Import into model
model.import_data(url_to_images)

# Compute image-hash
model.compute_hash(method='phash', hash_size=16)

# Find images with image-hash <= threshold
model.group(threshold=0)

# Plot the images
model.plot()

# Plot hash
# model.plot_hash([4])

model.clean_files(clean_tempdir=True)

# %%
# Import library
from undouble import Undouble

# Init with default settings
model = Undouble()

# Import example data
targetdir = model.import_example(data='flowers')

# Importing the files files from disk, cleaning and pre-processing
model.import_data(r'D:\REPOS\undouble\undouble\data\flower_images/')
model.import_data(r'D:\undouble_examples/')


# Compute image-hash
model.compute_hash(method='phash', hash_size=16)

# Find images with image-hash <= threshold
model.group(threshold=5)

# Plot the images
model.plot()

# Move the images
# model.move_to_dir()

# %%
model.plot_hash()
model.plot_hash(filenames=['165252721-d734174b-b5f6-4768-aa82-0ffd13d05f70.png', '165252723-407ba7ca-4df1-43d2-8fec-4bb82dfb6a35.png'])

model.plot_hash(filenames=['deson.png', 'deson-copy.png'])
model.plot_hash(filenames=['NOVA-NA-Dry-Iron-Grey-SDL881739255-1-3a886.jpeg', 'nova-plus-1100-w-amaze-ni-10-original-imaf3qxpabhhdwss.jpeg'])

import pandas as pd
df = pd.DataFrame(index=[model.results['filenames']], data=model.results['img_hash_hex'], columns=['image_hash_hex'])

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
    hashs = model.compute_imghash(model.results['img'][0], to_array=True)
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
    hashs = model.compute_imghash(model, to_array=False)


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
    model.compute_hash(method=method)

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
