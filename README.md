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
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

The aim of ``undouble`` is to detect (near-)identical images. It works using a multi-step process of pre-processing the images (grayscaling, normalizing, and scaling), computing the image hash, and the grouping of images. A threshold of 0 will group images with an identical image hash. The results can easily be explored by the plotting
functionality and images can be moved with the move functionality. When moving images, the image in the group with the largest resolution will be copied, and all other images are moved to the **undouble** subdirectory. In case you want to cluster your images, I would recommend reading the [blog](https://towardsdatascience.com/a-step-by-step-guide-for-clustering-images-4b45f9906128) and use the [clustimage library](https://erdogant.github.io/clustimage).

The following steps are taken in the ``undouble`` library:
 * Read recursively all images from directory with the specified extensions.
 * Compute image hash.
 * Group similar images.
 * Move if desired.


# 
**⭐️ Star this repo if you like it ⭐️**
#

### Blogs

* Read the blog to get a structured overview how to [detect duplicate images using image hash functions.](https://erdogant.medium.com/detection-of-duplicate-images-using-image-hash-functions-4d9c53f04a75")

# 

### [Documentation pages](https://erdogant.github.io/undouble/)

On the [documentation pages](https://erdogant.github.io/undouble/) you can find detailed information about the working of the ``undouble`` with many examples. 

# 


### Installation

##### It is advisable to create a new environment (e.g. with Conda). 
```bash
conda create -n env_undouble python=3.8
conda activate env_undouble
```

##### Install bnlearn from PyPI
```bash
pip install undouble            # new install
pip install -U undouble         # update to latest version
```

#### Directly install from github source
```bash
pip install git+https://github.com/erdogant/undouble
```  

#### Import Undouble package

```python
from undouble import Undouble
```

<hr>

### Examples:

##### [Example: Grouping similar images of the flower dataset](https://erdogant.github.io/undouble/pages/html/Examples.html#)

<p align="left">
  <a href="https://erdogant.github.io/undouble/pages/html/Examples.html#">
  <img src="https://github.com/erdogant/undouble/blob/main/docs/figs/flowers1.png" width="400" />
  </a>
</p>

<p align="left">
  <a href="https://erdogant.github.io/undouble/pages/html/Examples.html#">
  <img src="https://github.com/erdogant/undouble/blob/main/docs/figs/flowers2.png" width="400" />
  </a>
</p>

<p align="left">
  <a href="https://erdogant.github.io/undouble/pages/html/Examples.html#">
  <img src="https://github.com/erdogant/undouble/blob/main/docs/figs/flowers3.png" width="400" />
  </a>
</p>


# 

##### [Example: List all file names that are identifical](https://erdogant.github.io/undouble/pages/html/Examples.html#get-identical-images)

# 


##### [Example: Moving similar images in the flower dataset](https://erdogant.github.io/undouble/pages/html/Examples.html#move-files)

```python
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

# 

##### [Example: Plot the image hashes](https://erdogant.github.io/undouble/pages/html/Examples.html#plot-image-hash)


<p align="left">
  <a href="https://erdogant.github.io/undouble/pages/html/Examples.html#plot-image-hash">
  <img src="https://github.com/erdogant/undouble/blob/main/docs/figs/imghash_example.png" width="400" />
  </a>
</p>

# 

##### [Example: Three different imports](https://erdogant.github.io/undouble/pages/html/core_functions.html#input-data)

The input can be the following three types:

    * Path to directory
    * List of file locations
    * Numpy array containing images

#


##### [Example: Finding identical mnist digits](https://erdogant.github.io/undouble/pages/html/Examples.html#mnist-dataset)


<p align="left">
  <a href="https://erdogant.github.io/undouble/pages/html/Examples.html#mnist-dataset">
  <img src="https://github.com/erdogant/undouble/blob/main/docs/figs/mnist_1.png" width="400" />
  </a>
</p>



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
