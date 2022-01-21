.. _code_directive:

-------------------------------------

Quickstart
''''''''''

A quick example how to learn a model on a given dataset.


.. code:: python

    # Import library
    import undouble

    # Retrieve URLs of malicous and normal urls:
    X, y = undouble.load_example()

    # Learn model on the data
    model = undouble.fit_transform(X, y, pos_label='bad')

    # Plot the model performance
    results = undouble.plot(model)


Installation
''''''''''''

Create environment
------------------


If desired, install ``undouble`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_undouble python=3.6
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
