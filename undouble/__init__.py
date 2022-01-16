from undouble.undouble import Undouble

from undouble.undouble import (
    import_example,
    load_example,
    compute_blur,
)


__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.0'

# module level doc-string
__doc__ = """
undouble
=====================================================================

Description
-----------
Python package undouble is to (near-)identical images.

The aim of ``undouble`` is to detect (near-)identical images. It works using a multi-step proces of pre-processing the
images (grayscaling, normalizing, and scaling), computing the image-hash, and finding images that have image-hash
with a maximum difference <= threshold. A threshold of 0 will group images with an identical image-hash.
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
>>> model.preprocessing(targetdir)
>>>
>>> # Compute image-hash
>>> model.fit_transform()
>>>
>>> # Find images with image-hash <= threshold
>>> model.find(threshold=0)
>>>
>>> # Plot the images
>>> model.plot()
>>>
>>> # Move the images
>>> model.move()


References
----------
https://github.com/erdogant/undouble

"""
