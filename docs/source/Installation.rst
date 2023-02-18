.. include:: add_top.add

.. _code_directive:

-------------------------------------

Installation
''''''''''''

Create environment
------------------


If desired, install ``undouble`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_undouble python=3.8
    conda activate env_undouble


Install via ``pip``:

.. code-block:: console

    # The installation from pypi is disabled:
    pip install undouble

    # Install directly from github
    pip install git+https://github.com/erdogant/undouble


Uninstalling
''''''''''''

If you want to remove your ``undouble`` installation with your environment, it can be as following:

.. code-block:: console

   # List all the active environments. undouble should be listed.
   conda env list

   # Remove the undouble environment
   conda env remove --name undouble

   # List all the active environments. undouble should be absent.
   conda env list


Quickstart
''''''''''

A quick example how to learn a model on a given dataset.


.. code:: python

    # Import library
    from undouble import Undouble

    # Init with default settings
    model = Undouble()

    # Import example data
    targetdir = model.import_example(data='flowers')

    # Importing the files files from disk, cleaning and pre-processing
    model.import_data(targetdir)

    # Compute image-hash
    model.compute_hash()

    # Group images with image-hash <= threshold
    model.group(threshold=0)

    # Plot the images
    model.plot()

    # Move the images
    model.move()




.. include:: add_bottom.add