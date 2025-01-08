from datazets import get as import_example
from undouble.gui import Gui

from undouble.undouble import Undouble

from undouble.undouble import (
    compute_blur,
    seperate_path,
    sort_images,
)


__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.4.1'

# module level doc-string
__doc__ = """
undouble
=====================================================================

Python package undouble is to detect (near-)identical images.

The aim of ``undouble`` is to detect (near-)identical images. It works using a multi-step proces of pre-processing the
images (grayscaling, normalizing, and scaling), computing the image-hash, and grouping of images based on threshold value.
A threshold of 0 will group images with an identical image-hash.
The grouped can be visualized with the plot() functionality and easily moved with the move() functionality. When
moving images, the image in the group with the largest resolution will be copied, and all other images are moved to
the "undouble" subdirectory.

The following steps are taken:
    1. Read recursively all images from directory with the specified extensions.
    2. Compute image hash.
    3. Group similar images.
    4. Move if desired.

Example
-------
>>> # Import library
>>> from undouble import Undouble
>>>
>>> # Init with default settings
>>> model = Undouble(method='phash', hash_size=8)
>>>
>>> # Import example data
>>> targetdir = model.import_example(data='flowers')
>>>
>>> # Importing the files files from disk, cleaning and pre-processing
>>> model.import_data(targetdir)
>>>
>>> # Compute image-hash
>>> model.compute_hash()
>>>
>>> # Group images with image-hash <= threshold
>>> model.group(threshold=0)
>>>
>>> # Plot the images
>>> model.plot()
>>>
>>> # Move the images
>>> model.move_to_dir(gui=True)

References
----------
* Blog: https://towardsdatascience.com/detection-of-duplicate-images-using-image-hash-functions-4d9c53f04a75
* Github: https://github.com/erdogant/undouble
* Documentation: https://erdogant.github.io/undouble/

"""
