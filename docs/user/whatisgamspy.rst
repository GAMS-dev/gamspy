.. _whatisgamspy:

===============
What is GAMSPy?
===============

GAMSPy combines the high-performance GAMS execution system with the flexible Python language, creating 
a powerful mathematical optimization package. It acts as a bridge between the expressive Python language 
and the robust GAMS system, allowing you to create complex mathematical models effortlessly. This alows 
creating full pipelines in Python by using your favorite packages for preprocessing (e.g. data cleaning) and 
postprocessing (e.g. visualization) along with GAMSPy.

.. image:: ../_static/whatis.png
  :alt: What is GAMSPy?


Why is GAMSPy fast?
===================

GAMSPy delegates the expensive assignment and solve statements to GAMS execution system. Set-based operations 
are at the core of the GAMSPy and GAMS execution system. For example, in many optimization libraries what you 
would write your equations as given in the ``Other Libraries`` block below: 

.. tab-set-code::

    .. code-block:: Other-Libraries

        import other_library as ol
        
        I = ['i1', 'i2']
        J = ['j1', 'j2']
        a = ol.Parameter()
        x = ol.Variable()
        b = ol.Parameter()

        objective = sum(
          a[i,j] * x[i, j] 
          for i in I 
          for j in J
        ) >= b[i,j]

    .. code-block:: GAMSPy

        import gamspy as gp
        
        m = gp.Container()
        i = gp.Set(m, records=['i1', 'i2'])
        j = gp.Set(m, records=['j1', 'j2'])
        a = gp.Parameter(m)
        x = gp.Variable(m)
        b = gp.Parameter(m)

        objective = gp.Sum((i,j), a[i,j] * x[i,j]) >= b[i,j]

With the approach of other libraries, you iterate over all items of ``I`` and ``J``. This approach certain disadvantages:

- It can get pretty verbose for long statements with many loops which decreases the readability.
- The performance might suffer severely (depending on the implementation) if there are lots of items to iterate through since Python loops are known to be very slow.

Meanwhile GAMSPy implementation employs set-based operations. This results in:

- Concise and easier to read definitions.
- GAMSPy definitions closely resembles mathematical notation that you in papers (making it easier, typically, to correctly code mathematical constructs).
- Great performance since the actual operation is performed by GAMS using highly optimized low level code that has been improved in the last 40 years. 

Which backends GAMSPy support?
==============================

GAMSPy currently supports three backends: your local gams installation, `GAMS Engine <https://www.gams.com/sales/engine_facts/>`_ and `NEOS Server <https://neos-server.org/neos/>`_. 
The default backend is the local gams installation that comes with GAMSPy. With GAMS Engine backend, you can run hundreds of jobs in cloud on parallel to speed up your pipeline. 
One can also send their jobs to NEOS Server which is a free service for solving numerical optimization problems.  

GAMSPy is also integrated with `GAMS MIRO <https://www.gams.com/sales/miro_facts/>`_ which allows you to create fully interactive applications. Extensive visualization options of 
GAMS MIRO support you to make decisions based on optimization. 

.. image:: ../_static/miro.png
  :alt: GAMS MIRO

Interoperability of GAMSPy
==========================

A mathematical optimization model written with GAMSPy can be converted to GAMS or Latex format. GAMSPy to GAMS conversion might be useful to do fast prototyping in GAMSPy and 
converting the model to GAMS to use certain features of GAMS (e.g. EMP models) that are missing in GAMSPy. GAMSPy to Latex conversion can be convenient to put your equations 
into an academic paper or to verify the equations you have written matches with what you think mathematically. 
