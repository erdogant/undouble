undouble's documentation!
=========================

|python| |pypi| |docs| |stars| |LOC| |downloads_month| |downloads_total| |license| |forks| |open issues| |project status| |medium| |colab| |repo-size| |donate|

.. include:: add_top.add


.. _schematic_overview:

.. figure:: ../figs/schematic_overview.png

    
-----------------------------------


The aim of the ```undouble``` library is to detect (near-)identical images across an entire system or directory.
It works using a multi-step process of pre-processing the images (grayscaling, normalizing, and scaling), computing the image hash, and the grouping of images based on threshold value.

    * 1. Detects images with a identical image-hash in a specified folder or your entire system.
    * 2. The threshold can be used to detect near-identical images, such as photo-bursts.
    * 3. Plots to examine the groupings.
    * 4. Functionality to systematically undouble.

..	tip::
	`Read more details and the usage in the Medium Blog: Detection of Duplicate Images Using Image Hash Functions <https://towardsdatascience.com/detection-of-duplicate-images-using-image-hash-functions-4d9c53f04a75>`_



-----------------------------------

.. note::
	**Your ❤️ is important to keep maintaining this package.** You can `support <https://erdogant.github.io/undouble/pages/html/Documentation.html>`_ in various ways, have a look at the `sponser page <https://erdogant.github.io/undouble/pages/html/Documentation.html>`_.
	Report bugs, issues and feature extensions at `github <https://github.com/erdogant/undouble/>`_ page.

	.. code-block:: console

	   pip install undouble

-----------------------------------



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


.. |python| image:: https://img.shields.io/pypi/pyversions/undouble.svg
    :alt: |Python
    :target: https://erdogant.github.io/undouble/

.. |pypi| image:: https://img.shields.io/pypi/v/undouble.svg
    :alt: |Python Version
    :target: https://pypi.org/project/undouble/

.. |docs| image:: https://img.shields.io/badge/Sphinx-Docs-blue.svg
    :alt: Sphinx documentation
    :target: https://erdogant.github.io/undouble/

.. |stars| image:: https://img.shields.io/github/stars/erdogant/undouble
    :alt: Stars
    :target: https://img.shields.io/github/stars/erdogant/undouble

.. |LOC| image:: https://sloc.xyz/github/erdogant/undouble/?category=code
    :alt: lines of code
    :target: https://github.com/erdogant/undouble

.. |downloads_month| image:: https://static.pepy.tech/personalized-badge/undouble?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month
    :alt: Downloads per month
    :target: https://pepy.tech/project/undouble

.. |downloads_total| image:: https://static.pepy.tech/personalized-badge/undouble?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
    :alt: Downloads in total
    :target: https://pepy.tech/project/undouble

.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :alt: License
    :target: https://github.com/erdogant/undouble/blob/master/LICENSE

.. |forks| image:: https://img.shields.io/github/forks/erdogant/undouble.svg
    :alt: Github Forks
    :target: https://github.com/erdogant/undouble/network

.. |open issues| image:: https://img.shields.io/github/issues/erdogant/undouble.svg
    :alt: Open Issues
    :target: https://github.com/erdogant/undouble/issues

.. |project status| image:: http://www.repostatus.org/badges/latest/active.svg
    :alt: Project Status
    :target: http://www.repostatus.org/#active

.. |medium| image:: https://img.shields.io/badge/Medium-Blog-green.svg
    :alt: Medium Blog
    :target: https://erdogant.github.io/undouble/pages/html/Documentation.html#medium-blog

.. |donate| image:: https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors
    :alt: donate
    :target: https://erdogant.github.io/undouble/pages/html/Documentation.html#

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Colab example
    :target: https://erdogant.github.io/undouble/pages/html/Documentation.html#colab-notebook

.. |repo-size| image:: https://img.shields.io/github/repo-size/erdogant/undouble
    :alt: repo-size
    :target: https://img.shields.io/github/repo-size/erdogant/undouble

.. include:: add_bottom.add
