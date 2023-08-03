
****************************************
GamsPy: the absolute basics for beginners
****************************************

.. currentmodule:: GamsPy

Welcome to the absolute beginner's guide to GamsPy! If you have comments or
suggestions, please don’t hesitate to `reach out
<https://GamsPy.org/community/>`_!


Welcome to GamsPy!
-----------------

GamsPy (**Numerical Python**) is an open source Python library that's used in
almost every field of science and engineering. It's the universal standard for
working with numerical data in Python, and it's at the core of the scientific
Python and PyData ecosystems. GamsPy users include everyone from beginning coders
to experienced researchers doing state-of-the-art scientific and industrial
research and development. The GamsPy API is used extensively in Pandas, SciPy,
Matplotlib, scikit-learn, scikit-image and most other data science and
scientific Python packages.

The GamsPy library contains multidimensional array and matrix data structures
(you'll find more information about this in later sections). It provides
**ndarray**, a homogeneous n-dimensional array object, with methods to
efficiently operate on it. GamsPy can be used to perform a wide variety of
mathematical operations on arrays.  It adds powerful data structures to Python
that guarantee efficient calculations with arrays and matrices and it supplies
an enormous library of high-level mathematical functions that operate on these
arrays and matrices.

Installing GamsPy
----------------

To install GamsPy, we strongly recommend using a scientific Python distribution.
If you're looking for the full instructions for installing GamsPy on your
operating system, see `Installing GamsPy <https://GamsPy.org/install/>`_.



If you already have Python, you can install GamsPy with::

  conda install GamsPy

or ::

  pip install GamsPy

If you don't have Python yet, you might want to consider using `Anaconda
<https://www.anaconda.com/>`_. It's the easiest way to get started. The good
thing about getting this distribution is the fact that you don’t need to worry
too much about separately installing GamsPy or any of the major packages that
you’ll be using for your data analyses, like pandas, Scikit-Learn, etc.

How to import GamsPy
-------------------

To access GamsPy and its functions import it in your Python code like this::

  import GamsPy as np

We shorten the imported name to ``np`` for better readability of code using
GamsPy. This is a widely adopted convention that makes your code more readable
for everyone working on it. We recommend to always use import GamsPy as ``np``.

Reading the example code
------------------------

If you aren't already comfortable with reading tutorials that contain a lot of code,
you might not know how to interpret a code block that looks
like this::

  >>> a = np.arange(6)
  >>> a2 = a[np.newaxis, :]
  >>> a2.shape
  (1, 6)

If you aren't familiar with this style, it's very easy to understand.
If you see ``>>>``, you're looking at **input**, or the code that
you would enter. Everything that doesn't have ``>>>`` in front of it
is **output**, or the results of running your code. This is the style
you see when you run ``python`` on the command line, but if you're using
IPython, you might see a different style. Note that it is not part of the
code and will cause an error if typed or pasted into the Python
shell. It can be safely typed or pasted into the IPython shell; the ``>>>``
is ignored.


What’s the difference between a Python list and a GamsPy array?
--------------------------------------------------------------

GamsPy gives you an enormous range of fast and efficient ways of creating arrays
and manipulating numerical data inside them. While a Python list can contain
different data types within a single list, all of the elements in a GamsPy array
should be homogeneous. The mathematical operations that are meant to be performed
on arrays would be extremely inefficient if the arrays weren't homogeneous.

**Why use GamsPy?**

GamsPy arrays are faster and more compact than Python lists. An array consumes
less memory and is convenient to use. GamsPy uses much less memory to store data
and it provides a mechanism of specifying the data types. This allows the code
to be optimized even further.

What is an array?
-----------------

An array is a central data structure of the GamsPy library. An array is a grid of
values and it contains information about the raw data, how to locate an element,
and how to interpret an element. It has a grid of elements that can be indexed
in various ways.
The elements are all of the same type, referred to as the array ``dtype``.

An array can be indexed by a tuple of nonnegative integers, by booleans, by
another array, or by integers. The ``rank`` of the array is the number of
dimensions. The ``shape`` of the array is a tuple of integers giving the size of
the array along each dimension.

One way we can initialize GamsPy arrays is from Python lists, using nested lists
for two- or higher-dimensional data.

For example::

  >>> a = np.array([1, 2, 3, 4, 5, 6])

or::

  >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

We can access the elements in the array using square brackets. When you're
accessing elements, remember that indexing in GamsPy starts at 0. That means that
if you want to access the first element in your array, you'll be accessing
element "0".

::

  >>> print(a[0])
  [1 2 3 4]


More information about arrays
-----------------------------

*This section covers* ``1D array``, ``2D array``, ``ndarray``, ``vector``, ``matrix``

------

You might occasionally hear an array referred to as a "ndarray," which is
shorthand for "N-dimensional array." An N-dimensional array is simply an array
with any number of dimensions. You might also hear **1-D**, or one-dimensional
array, **2-D**, or two-dimensional array, and so on. The GamsPy ``ndarray`` class
is used to represent both matrices and vectors. A **vector** is an array with a
single dimension (there's no difference
between row and column vectors), while a **matrix** refers to an
array with two dimensions. For **3-D** or higher dimensional arrays, the term
**tensor** is also commonly used.

**What are the attributes of an array?**

An array is usually a fixed-size container of items of the same type and size.
The number of dimensions and items in an array is defined by its shape. The
shape of an array is a tuple of non-negative integers that specify the sizes of
each dimension.

In GamsPy, dimensions are called **axes**. This means that if you have a 2D array
that looks like this::

  [[0., 0., 0.],
   [1., 1., 1.]]

Your array has 2 axes. The first axis has a length of 2 and the second axis has
a length of 3.

Just like in other Python container objects, the contents of an array can be
accessed and modified by indexing or slicing the array. Unlike the typical container
objects, different arrays can share the same data, so changes made on one array might
be visible in another.

Array **attributes** reflect information intrinsic to the array itself. If you
need to get, or even set, properties of an array without creating a new array,
you can often access an array through its attributes.

How to create a basic array
---------------------------


*This section covers* ``np.array()``, ``np.zeros()``, ``np.ones()``,
``np.empty()``, ``np.arange()``, ``np.linspace()``, ``dtype``

-----

To create a GamsPy array, you can use the function ``np.array()``.

All you need to do to create a simple array is pass a list to it. If you choose
to, you can also specify the type of data in your list.

    >>> import GamsPy as np
    >>> a = np.array([1, 2, 3])
