.. _whatisgamspy:

***************
What is GAMSPy?
***************

GAMSPy is a mathematical optimization package that combines the power of our high performance 
GAMS execution system and the flexible Python language. It is a Python library that provides 
GAMS-like syntax to write mathematical models and thus, GAMSPy combines the best of two world.

In this section we will provide an overview of the features and benefits that are unique
to GAMSPy and which are supposed to help you to find the perfect modeling language and
environment for your modeling experience. 


Model Instances vs. Mathematical Models
---------------------------------------

One key element in writing good quality, good readable, good maintainable, and thus, transparent
models that can be used for long time in production comes down to the algebraic formulation.
The algebraic formulation is something that is around and accepted for years and allows 
to understand models. Consequently, being able to write mathematical models in a language
that can be processed by computers **and** are close to the origitnal algebraic notation is 
what will be key to achive maintable models.

With this goal in mind we developed GAMSPy to be able to generate mathematical models instead
of model instances. While a mathematical model is soley composed of mathematical symbols, a
model instance is the translation of a mathematical model filled with the according instance data.
Thus, sum expressions are resolved into its individual components and equation domains are resoveld to
individual scalar equations. 

Mathematical Model

.. math::

    \sum_{i \in \mathcal{I}} p_{i,j} \cdot x_{i,j} \le d_j \forall j \in \mathcal{J}

Model Instance

.. math::

    5 \cdot x_{i1,j1} + 3 \cdot x_{i2,j1} + 2 \cdot x_{i3,j1} \le 7
    2 \cdot x_{i1,j2} + 6 \cdot x_{i2,j2} + 4 \cdot x_{i3,j2} \le 10

Especially for large models with many variables and equations, a model instance becomes large
and heavy to handle, maintain, and read. 



Sparsity
---------

One key aspect of any modeling language is how it handles sparse multidimensional data structures.
Many optimization problems are subject to a particular structure in which the data cube 
has a lot of zeros and only a few non-zeros, a characteristic referred to as sparse. In 
optimization problems, it is often necessary to account for complex mappings of indices 
to subsets.

.. attention::
    Do we want the sparsity example from the blog post here?

While you might be used to taking on the full responsibility to make sure only the relevant combinations
of indices go into your variable definition in the Python modeling world, we especially focused on 
transferring the convenience and mindset of GAMS into Python by designing GAMSPy. Thus, GAMSPy 
automatically takes care of generating variables only for the relevant combinations of indices based 
on the algebraic formulation. This feature is particularly useful when working with a large multidimensional 
index space, where generating all possible combinations of indices would be computationally expensive and unnecessary. 
GAMSPy quietly handles this task in the background, allowing us to focus on the formulation of the model::

    Variable(container, "x", [i, j], type="Positive")

The same argument holds for equation domain and sum definitions::

    e = Equation(container, ...)
    e[...] = ... 


Performance
-----------

GAMSPy generates GAMS code and executes it by using the GAMS 
backend to resolve assignment operations, generate and solve models. Since GAMS 
have been optimized for decades (since 1970s), and supports many solvers
that have been developed by optimization experts, it provides good performance
for model generation and solving models. This is the main source of the speed of
GAMSPy.

.. seealso::
    `Performance in Optimization Models: A Comparative Analysis of GAMS, Pyomo, GurobiPy, and JuMP <https://www.gams.com/blog/2023/07/performance-in-optimization-models-a-comparative-analysis-of-gams-pyomo-gurobipy-and-jump/>`_


.. attention::
    Do we want a plot showing the differences of GAMS vs GAMSPy performance based on the performance blog post?


Optimization Pipeline Management
---------------------------------

GAMSPy opens up entirely new opportunities to streamline optimization and data pipelines:

**Seamless Pipeline Management**
- No switching of environments: Manage data preprocessing and optimization tasks within a single, intuitive environment.
- Leverage your favorite Python libraries (e.g. Numpy, Pandas, Networkx) to comfortably manipulate and visualize data.
- Import and export data and optimization results to many data formats.

**Different Runner Options***
You can run your model on:
- Your local machine
- GAMS Engine One (you host your own server hardware)
- GAMS Engine SASS (you don't even need to run a server. We make sure you have access to the right resources, any time.)


Summarizing the Benefits
------------------------

- Generation of mathematical models
- Data independent modeling
- Convenient handling of sparse data structures
- Streamlined optimization pipeline management