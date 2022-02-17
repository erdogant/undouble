# undouble

[![Python](https://img.shields.io/pypi/pyversions/undouble)](https://img.shields.io/pypi/pyversions/undouble)
[![PyPI Version](https://img.shields.io/pypi/v/undouble)](https://pypi.org/project/undouble/)
[![License](https://img.shields.io/badge/license-BSD3-green.svg)](https://github.com/erdogant/undouble/blob/master/LICENSE)
[![Github Forks](https://img.shields.io/github/forks/erdogant/undouble.svg)](https://github.com/erdogant/undouble/network)
[![GitHub Open Issues](https://img.shields.io/github/issues/erdogant/undouble.svg)](https://github.com/erdogant/undouble/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-blue)](https://erdogant.github.io/undouble/)
[![Downloads](https://pepy.tech/badge/undouble/month)](https://pepy.tech/project/undouble)
[![Downloads](https://pepy.tech/badge/undouble)](https://pepy.tech/project/undouble)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-blue)](https://erdogant.github.io/undouble/)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

Python package ``undouble`` is to detect (near-)identical images.

# 
**Star this repo if you like it! ⭐️**
#

## Blog/Documentation

* <a href="https://erdogant.github.io/XXX/"> <img src="https://img.shields.io/badge/Sphinx-Docs-Green" alt="Open documentation pages"/> </a> XXX documentation pages 
* <a href="https://colab.research.google.com/github/erdogant/XXX/blob/master/notebooks/XXX.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open example In Colab"/> </a> Notebook example 
* <a href="https://towardsdatascience.com/a-step-by-step-guide-for-clustering-images-4b45f9906128"> <img src="https://img.shields.io/badge/Medium-Blog-blue" alt="Open Blog"/> </a> Blog: A step-by-step guide for clustering images 

[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)
[![Medium](https://img.shields.io/badge/Medium-Blog-blue)](https://towardsdatascience.com/detection-of-duplicate-images-using-image-hash-functions-4d9c53f04a75)
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

* [**Read the blog: Detection of Duplicate Images Using Image Hash Functions**](https://towardsdatascience.com/detection-of-duplicate-images-using-image-hash-functions-4d9c53f04a75)

Python package undouble is to detect (near-)identical images.

The aim of ``undouble`` is to detect (near-)identical images. It works using a multi-step process of pre-processing the
images (grayscaling, normalizing, and scaling), computing the image hash, and the grouping of images.
A threshold of 0 will group images with an identical image hash. The results can easily be explored by the plotting
functionality and images can be moved with the move functionality. When moving images, the
image in the group with the largest resolution will be copied, and all other images are moved to the "undouble"
subdirectory. In case you want to cluster your images, I would recommend reading the [blog](https://towardsdatascience.com/a-step-by-step-guide-for-clustering-images-4b45f9906128) and use the
[clustimage library](https://github.com/erdogant/clustimage).

The following steps are taken in the ``undouble`` library:
 * 1. Read recursively all images from directory with the specified extensions.
 * 2. Compute image hash.
 * 3. Group similar images.
 * 4. Move if desired.


### Installation
* Install undouble from PyPI (recommended). undouble is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* A new environment can be created as following:

```bash
conda create -n env_undouble python=3.8
conda activate env_undouble
```

```bash
pip install undouble            # new install
pip install -U undouble         # update to latest version
```

* Alternatively, you can install from the GitHub source:
```bash
# Directly install from github source
pip install -e git://github.com/erdogant/undouble.git@0.1.0#egg=master
pip install git+https://github.com/erdogant/undouble#egg=master
pip install git+https://github.com/erdogant/undouble

# By cloning
git clone https://github.com/erdogant/undouble.git
cd undouble
pip install -U .
```  

#### Import undouble package
```python
from undouble import Undouble
```

#### Example:
```python

# Import library
from undouble import Undouble

# Init with default settings
model = Undouble(method='phash', hash_size=8)

# Import example data
targetdir = model.import_example(data='flowers')

# Importing the files files from disk, cleaning and pre-processing
model.import_data(targetdir)

# Compute image-hash
model.compute_hash()

# [undouble] >INFO> Store examples at [./undouble/data]..
# [undouble] >INFO> Downloading [flowers] dataset from github source..
# [undouble] >INFO> Extracting files..
# [undouble] >INFO> [214] files are collected recursively from path: [./undouble/data/flower_images]
# [undouble] >INFO> Reading and checking images.
# [undouble] >INFO> Reading and checking images.
# 100%|██████████| 214/214 [00:02<00:00, 96.56it/s]
# [undouble] >INFO> Extracting features using method: [phash]
# 100%|██████████| 214/214 [00:00<00:00, 3579.14it/s]
# [undouble] >INFO> Build adjacency matrix with phash differences.
# [undouble] >INFO> Extracted features using [phash]: (214, 214)
# 100%|██████████| 214/214 [00:00<00:00, 129241.33it/s]


# Group images with image-hash <= threshold
model.group(threshold=0)

# [undouble] >INFO> Number of groups with similar images detected: 3
# [undouble] >INFO> [3] groups are detected for [7] images.

# Plot the images
model.plot()

# Move the images
model.move()

# -------------------------------------------------
# >You are at the point of physically moving files.
# -------------------------------------------------
# >[7] similar images are detected over [3] groups.
# >[4] images will be moved to the [undouble] subdirectory.
# >[3] images will be copied to the [undouble] subdirectory.

# >[C]ontinue moving all files.
# >[W]ait in each directory.
# >[Q]uit
# >Answer: w

```
<p align="center">
  <img src="https://github.com/erdogant/undouble/blob/main/docs/figs/flowers1.png" width="400" />
  <img src="https://github.com/erdogant/undouble/blob/main/docs/figs/flowers2.png" width="400" />
  <img src="https://github.com/erdogant/undouble/blob/main/docs/figs/flowers3.png" width="400" />
</p>


#### plot image hash

```python

# Import library
import matplotlib.pyplot as plt
from undouble import Undouble

# Init with default settings
hash_size=8
model = Undouble(method='phash', hash_size=hash_size)

# Import example data
targetdir = model.import_example(data='flowers')

# Importing the files files from disk, cleaning and pre-processing
model.import_data(targetdir)

# Compute image-hash
results = model.compute_hash(return_dict=True)


fig, ax=plt.subplots(1,2)
ax[0].imshow(results['img'][0])
ax[1].imshow(results['img_hash_bin'][0].reshape(hash_size, hash_size), cmap='gray')
plt.title('image hash hex: %s' %(results['img_hash_hex'][0]))

```

<p align="center">
  <img src="https://github.com/erdogant/undouble/blob/main/docs/figs/imghash_example.png" width="400" />
</p>


#### Example of three different image imports


```python

# Import library
import os
from undouble import Undouble

# Init with default settings
model = Undouble(method='phash', hash_size=16)

# Import data; Pathnames to the images.
input_list_of_files = model.import_example(data='flowers')

# Import data; Directory to read.
input_directory, _ = os.path.split(input_list_of_files[0])
print(input_directory)
# '.\\undouble\\undouble\\data\\flower_images'

# Import data; numpy array containing images.
input_img_array = model.import_example(data='mnist')

# Importing the files files from disk, cleaning and pre-processing
model.import_data(input_list_of_files)
model.import_data(input_directory)
model.import_data(input_img_array)

# Compute image-hash
model.compute_hash()

# Group images with image-hash <= threshold
model.group(threshold=0)

# Plot the images
model.plot()

# Move the images
# model.move()

```

#### Finding identical mnist digits.

```python

# Import library
from undouble import Undouble

# Init with default settings
model = Undouble()

# Import example data
targetdir = model.import_example(data='mnist')

# Importing the files files from disk, cleaning and pre-processing
model.import_data(targetdir)

# Compute image-hash
model.compute_hash(method='phash', hash_size=16)

# Group images with image-hash <= threshold
model.group(threshold=0)

# Plot the images
model.plot()

```

#### References
* https://github.com/erdogant/undouble

#### Citation
Please cite in your publications if this is useful for your research (see citation).
   
### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

### Contribute
* All kinds of contributions are welcome!
* If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated :)

### Licence
See [LICENSE](LICENSE) for details.

### Other interesting stuf
* https://github.com/JohannesBuchner/imagehash
* https://towardsdatascience.com/a-step-by-step-guide-for-clustering-images-4b45f9906128
