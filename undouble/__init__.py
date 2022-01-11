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
undouble is for the detection of duplicate photos and mark or move the photos to undouble the collection.
The following steps are taken:
    1. Read recursively all images from directory with the specified extensions.
    2. Compute image hash per photo.
    3. Mark similar images.

Example
-------
>>> from undouble import Undouble
>>>
>>> # Init with default settings
>>> model = Undouble()
>>>
>>> # load example with faces
>>> X = cl.import_example(data='mnist')
>>>
>>> # Cluster digits
>>> results = cl.fit_transform(X)
>>>
>>> # Find images
>>> results_find = cl.find(X[0,:], k=None, alpha=0.05)
>>> cl.plot_find()
>>> cl.scatter()
>>>

References
----------
https://github.com/erdogant/undouble

"""
