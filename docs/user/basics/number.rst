.. _number:

******
Number
******

A :meth:`gamspy.Number` object is needed for assigning records to a symbol conditionally. For example:

.. code-block:: python

    import gamspy as gp
    
    m = gp.Container()
    k = gp.Set(m, "k", records=["1964-i","1964-ii","1964-iii","1964-iv"])
    ki = gp.Set(m, domain=k, description="initial period")
    ki[k] = gp.Number(1).where[gp.Ord(k) == 1]

The code snippet above would assign only `1964-i` to `ki` since only the order of `1964-i` is equal to 1.
