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
The aim of this library is to detect (near-)duplicate images and move the images.
The following steps are taken:
    1. Read recursively all images from directory with the specified extensions.
    2. Compute image hash per photo.
    3. Mark similar images.

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
