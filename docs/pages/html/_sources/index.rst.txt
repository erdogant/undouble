undouble's documentation!
============================

The aim of the ```undouble``` library is to detect (near-)identical images across an entire system or directory.
It works using a multi-step process of pre-processing the images (grayscaling, normalizing, and scaling), computing the image hash, and the grouping of images based on threshold value.

    * 1. Detects images with a identical image-hash in a specified folder or your entire system.
    * 2. The threshold can be used to detect near-identical images, such as photo-bursts.
    * 3. Plots to examine the groupings.
    * 4. Functionality to systematically undouble.


.. _schematic_overview:

.. figure:: ../figs/schematic_overview.png
    

..	tip::
	`Read more details and the usage in the Medium Blog: Detection of Duplicate Images Using Image Hash Functions <https://towardsdatascience.com/detection-of-duplicate-images-using-image-hash-functions-4d9c53f04a75>`_



You contribution is important
==============================
If you ❤️ this project, **star** this repo at the `github page <https://github.com/erdogant/undouble/>`_ and have a look at the `sponser page <https://erdogant.github.io/undouble/pages/html/Documentation.html>`_!


Github
======
Please report bugs, issues and feature extensions at `github <https://github.com/erdogant/undouble/>`_.


Quick install
=============

.. code-block:: console

   pip install undouble



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
  :caption: Documentation

  Documentation
  Coding quality
  undouble.undouble



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>

