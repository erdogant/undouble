undouble's documentation!
========================

The aim of the ```undouble``` library is to detect (near-)identical images across an entire system or directory.
It works using a multi-step process of pre-processing the images (grayscaling, normalizing, and scaling), computing the image hash, and the grouping of images based on threshold value.

    * 1. Detects images with a identical image-hash in a specified folder or your entire system.
    * 2. The threshold can be used to detect near-identical images, such as photo-bursts.
    * 3. Plots to examine the groupings.
    * 4. Functionality to systematically undouble.


.. _schematic_overview:

.. figure:: ../figs/schematic_overview.png
    

Content
=======

.. toctree::
   :maxdepth: 1
   :caption: Background
   
   Abstract


.. toctree::
   :maxdepth: 1
   :caption: Installation
   
   Installation


.. toctree::
  :maxdepth: 1
  :caption: Core functionalities

  core_functions


.. toctree::
  :maxdepth: 1
  :caption: hash_functions

  hash_functions


.. toctree::
  :maxdepth: 1
  :caption: Examples

  Examples


.. toctree::
  :maxdepth: 1
  :caption: Code Documentation
  
  Documentation
  Coding quality
  undouble.undouble



Quick install
-------------

.. code-block:: console

   pip install undouble




Source code and issue tracker
------------------------------

Available on Github, `erdogant/undouble <https://github.com/erdogant/undouble/>`_.
Please report bugs, issues and feature extensions there.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
