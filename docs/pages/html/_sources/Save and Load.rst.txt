.. _code_directive:

-------------------------------------

Save and Load
''''''''''''''

Saving and loading models is desired as the learning proces of a model for ``undouble`` can take up to hours.
In order to accomplish this, we created two functions: function :func:`undouble.save` and function :func:`undouble.load`
Below we illustrate how to save and load models.


Saving
----------------

Saving a learned model can be done using the function :func:`undouble.save`:

.. code:: python

    import undouble

    # Load example data
    X,y_true = undouble.load_example()

    # Learn model
    model = undouble.compute_hash(X, y_true, pos_label='bad')

    Save model
    status = undouble.save(model, 'learned_model_v1')



Loading
----------------------

Loading a learned model can be done using the function :func:`undouble.load`:

.. code:: python

    import undouble

    # Load model
    model = undouble.load(model, 'learned_model_v1')
