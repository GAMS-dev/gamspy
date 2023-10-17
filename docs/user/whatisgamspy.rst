.. _whatisgamspy:

***************
What is GAMSPy?
***************

GAMSPy blends the power of our high performance GAMS execution system with the flexible Python 
language, creating a powerful mathematical optimization package. It's like a bridge between 
the expressive Python language and the robust GAMS system, allowing you to create complex 
mathematical models effortlessly.

In this section we provide an overview of the features and benefits that are unique
to GAMSPy and which are supposed to help you to find the perfect modeling language and
environment for your modeling experience. 


Model Instances vs. Mathematical Models
---------------------------------------

Creating robust, readable, and maintainable models is an art rooted in algebraic formulation. 
The ability to express mathematical models in a language that retains the essence of algebraic 
notation and is machine-processable is paramount. 

With this goal in mind we developed GAMSPy to be able to generate mathematical models instead
of model instances. Think of a mathematical model as a pure representation of mathematical symbols, 
devoid of specific data. In contrast, a model instance is the same model filled with actual data, 
In a model instance sum expressions are resolved into its individual components and equation 
domains are resolved to individual scalar equations.

Mathematical Model

.. math::

    \sum_{i \in \mathcal{I}} p_{i,j} \cdot x_{i,j} \le d_j \forall j \in \mathcal{J}

Model Instance

.. math::

    5 \cdot x_{i1,j1} + 3 \cdot x_{i2,j1} + 2 \cdot x_{i3,j1} \le 7 
    
    2 \cdot x_{i1,j2} + 6 \cdot x_{i2,j2} + 4 \cdot x_{i3,j2} \le 10

Especially for large models with many variables and equations, a model instance becomes large, machine
making it complex and harder to manage. Therefore, GAMSPy leverages the idea of a standalone,
data independent, and indexed representation of a mathematical model which is very close 
to the original algebraic formulation.


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

Working on an optimization problem does not solely include the mathematical model but also includes tasks regarding
data pre- and postprocessing as well as visualization. At GAMS, we understand the importance of making these tasks as 
comfortable and efficient as possible. With GAMSPy we are now able to streamline the complete optimization pipeline
starting with data input and preprocessing followed by the implementation of the mathematical model and data postprocessing
and visualization, in a single, intuitive Python environment. GAMSPy allows you to leverage your favorite Python libraries 
(e.g. Numpy, Pandas, Networkx) to comfortably manipulate and visualize data. And it allows to import and export data and 
optimization results to many data formats. 

On top, GAMSPy seamlessly works with GAMS MIRO and GAMS Engine which allows you to run your GAMSPy optimization either on
your local machine or on your own server hardware (GAMS Engine One) as well as on GAMS Engine SaaS, where you don't even 
need to run a server. We make sure you have access to the right resources, any time.


How is GAMSPy different from GAMS?
--------------------------------

.. attention::
    Discuss and maybe move to FAQs?

Summarizing the Benefits
------------------------

- Generation of mathematical models
- Data independent modeling
- Convenient handling of sparse data structures
- Streamlined optimization pipeline management
