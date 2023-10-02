==========
Quickstart
==========

Prerequisites
=============

You'll need to know a bit of Python. For a refresher, see the `Python
tutorial <https://docs.python.org/tutorial/>`__.

**Learner profile**

This is a quick overview of how you use GAMSPy constructs to generate mathematical models. 
It demonstrates how you can create symbols, manipulate their records, and use them in models. 
In particular, if you don't know how to define models by using **Set**\s, **Parameter**\s, **Variable**\s, **Equation**\s  
this article might be of help. 

**Learning Objectives**

After reading, you should be able to:

- Create **Set**\s, **Parameter**\s, **Variable**\s, **Equation**\s, **Model**\s and **Container**\s.
- Manipulate the records of symbols.
- Combine the symbols to create **Equation**\s.
- Solve models and retrieve their results.

.. _quickstart.the-basics:

The Basics
==========

GAMSPy's main components are:

- Container
- Set
- Alias
- Parameter
- Variable
- Equation
- Model

A symbol is either a Set, Alias, Parameter, Variable, or Equation.

**Container**
    Container is the component which contains symbols. Each symbol
    has to have a container associated with it.

**Set**
    Sets are fundamental building blocks in any GAMS model. They allow the model to be succinctly defined and easily read.

**Alias**
    It is sometimes necessary to have more than one name for the same set. 
    For example, in input-output models, each commodity may be used in the production of all other commodities 
    and it is necessary to have two names for the set of commodities to specify the problem without ambiguity

**Parameter**
    A parameter is a data holder used to enter list oriented data which can be indexed over one or several sets.

**Variable**
    A variable is the GAMS name for what are called endogenous variables by economists, columns or 
    activities by linear programming experts, and decision variables by industrial Operations Research practitioners. 
    They are the entities whose values are generally unknown until after a model has been solved.

**Equation**
    An equation name is associated with the symbolic algebraic relationships that will be used to generate the constraints in a model. 
    The algebraic relationships are defined by using constants, mathematical operators, functions, sets, parameters and variables. 

**Model**
    The model statement is used to collect equations into groups and to label them so that they can be solved.

An example::

    m = Container()
    
    t = Set(m, name="t", records=[f"q-{i}" for i in range(1, 5)])
    
    price = Parameter(
        m, name="price", domain=[t], records=np.array([10, 12, 8, 9])
    )
    istock = Parameter(
        m, name="istock", domain=[t], records=np.array([50, 0, 0, 0])
    )  # OR records=pd.DataFrame([["q-1", 50]])
    storecost = Parameter(m, name="storecost", records=1)
    storecap = Parameter(m, name="storecap", records=100)

    stock = Variable(m, name="stock", domain=[t], type="Positive")
    sell = Variable(m, name="sell", domain=[t], type="Positive")
    buy = Variable(m, name="buy", domain=[t], type="Positive")
    cost = Variable(m, name="cost")

    sb = Equation(m, name="sb", domain=[t])
    at = Equation(m, name="at")

    sb[t] = stock[t] == stock[t.lag(1, "linear")] + buy[t] - sell[t] + istock[t]
    
    at.definition = cost == Sum( 
        t, price[t] * (buy[t] - sell[t]) + storecost * stock[t]
    )

    stock.up[t] = storecap

    swp = Model(
        m,
        name="swp",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=cost,
    )
    swp.solve()

    print("Objective function value: ", swp.objective_value)
    

Symbol Creation
---------------

There are two ways to create a symbol.

Manipulating and Printing Records
---------------------------------

Further reading
===============

-  The `GAMS tutorial <https://www.gams.com/latest/docs/UG_MAIN.html#UG_Language_Environment>`__