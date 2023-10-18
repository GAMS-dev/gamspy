.. _gamspy_docs_mainpage:

.. toctree::
   :maxdepth: 1
   :hidden:

   User Guide <user/index>
   API Reference <reference/index>
   Release Notes <release/index>

###############
What is GAMSPy?
###############

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
devoid of specific data. In contrast, a model instance is a model instance is the unrolled and 
constant folded representation of a model with its actual data.
In a model instance sum expressions are resolved into its individual components and equation 
domains are resolved to individual scalar equations.

Mathematical Model

.. math::

    \sum_{i \in \mathcal{I}} \frac{p_{i,j} - q_i}{a_j} \cdot x_{i,j} \le \sum_{k \in \mathcal{K}} d_{k,j} \hspace{1cm} \forall \: j \in \mathcal{J}

Model Instance

.. math::

    5 \cdot x_{i1,j1} + 3 \cdot x_{i2,j1} + 2 \cdot x_{i3,j1} \le 7 
    
    2 \cdot x_{i1,j2} + 6 \cdot x_{i2,j2} + 4 \cdot x_{i3,j2} \le 10

Especially for large models with many variables and equations, a model instance becomes large, machine
making it complex and harder to manage. Therefore, GAMSPy leverages the idea of a standalone,
data independent, and indexed representation of a mathematical model which is very close 
to the original mathematical algebraic formulation.


Sparsity
---------

One key aspect of any modeling language is how it handles sparse multidimensional data structures.
Many optimization problems are subject to a particular structure in which the data cube 
has a lot of zeros and only a few non-zeros, a characteristic referred to as sparse. In 
optimization problems, it is often necessary to account for complex mappings of indices 
to subsets.

While you might be used to taking on the full responsibility to make sure only the relevant combinations
of indices go into your variable definition in the Python modeling world, we especially focused on 
transferring the convenience and mindset of GAMS into Python by designing GAMSPy. Thus, GAMSPy 
automatically takes care of generating variables only for the relevant combinations of indices based 
on the algebraic formulation. This feature is particularly useful when working with a large multidimensional 
index space, where generating all possible combinations of indices would be computationally expensive and unnecessary. 
GAMSPy quietly handles this task in the background, allowing us to focus on the formulation of the model.


Performance
-----------

GAMSPy utilizes the GAMS backend to resolve assignment operations, generate and solve models. Since GAMS 
have been optimized for decades, and supports many solvers that have been developed by optimization experts, 
it provides outstanding performance for model generation and solving models. This is the main source of the 
speed of GAMSPy.

.. seealso::
    `Performance in Optimization Models: A Comparative Analysis of GAMS, Pyomo, GurobiPy, and JuMP <https://www.gams.com/blog/2023/07/performance-in-optimization-models-a-comparative-analysis-of-gams-pyomo-gurobipy-and-jump/>`_


Optimization Pipeline Management
---------------------------------

Working on an optimization problem does not solely include the mathematical model but also includes tasks regarding
data pre- and postprocessing as well as visualization. At GAMS, we understand the importance of making these tasks as 
comfortable and efficient as possible. With GAMSPy we introduce a new way to streamline the complete optimization pipeline
starting with data input and preprocessing followed by the implementation of the mathematical model and data postprocessing
and visualization, in a single, intuitive Python environment. GAMSPy allows you to leverage your favorite Python libraries 
(e.g. Numpy, Pandas, Networkx) to comfortably manipulate and visualize data. And it allows to import and export data and 
optimization results to many data formats. 

On top, GAMSPy seamlessly works with GAMS MIRO and GAMS Engine which allows you to run your GAMSPy optimization either on
your local machine or on your own server hardware (GAMS Engine One) as well as on GAMS Engine SaaS, where you don't even 
need to run a server. We make sure you have access to the right resources, any time.


How is GAMSPy different from GAMS?
----------------------------------

GAMS is a domain-specific declarative language that incorporates procedural elements from a general-purpose programming language, such as loops and conditional statements. In contrast, Python is a general-purpose programming language where these elements are already inherent. With the integration of the GAMSPy library, domain-specific features like parallel assignment statements of a declarative language are now made available in Python. This facilitates a seamless connection between the specialized modeling capabilities of GAMS and the flexibility and versatility of Python.

Summarizing the Benefits
------------------------

- Generation of mathematical models in Python
- Abstract algebraic data independent modeling in Python
- Convenient handling of sparse data structures in Python
- Streamlined optimization pipeline management in Python
