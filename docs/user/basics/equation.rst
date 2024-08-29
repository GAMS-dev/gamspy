.. _equation:

.. meta::
   :description: Documentation of GAMSPy Equation (gamspy.Equation)
   :keywords: Equation, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance

********
Equation
********

.. _equation_introduction:

Introduction
------------

Equations in GAMSPy are associated with the symbolic algebraic relationships
that will be used to generate the constraints in a model. The algebraic
relationships are defined by using constants, mathematical operators,
functions, sets, parameters and variables. As with variables,
an equation may be defined over a group of sets and in turn map into several
individual constraints associated with the elements of those sets.
Equations are specified in two steps. First they have to be :ref:`declared <equation_declaration>`, afterwards
they get a :ref:`definition <equation_definition>`. Finally, in order
to be considered, they have to be added to an instance of :ref:`gamspy.Model <model>` through
the ``equations`` argument of its constructor. A handy shortcut to retrieve all equations
contained in a :ref:`gamspy.Container <container>` is the
:meth:`getEquations() <gamspy.Container.getEquations>` method.

.. _equation_declaration:

Equation Declaration
--------------------

A GAMSPy equation must be declared before it may
be referenced by creating an instance of :meth:`Equation <gamspy.Equation>`.
The declaration of an equation is similar to a set, parameter,
or variable, in that it requires a container, an optional name, a domain (if applicable),
and an optional description. For a complete list of available arguments, see
the :meth:`Equation reference<gamspy.Equation>`.

Below is an example of equation declarations adapted from `trnsport.py <https://github.com/GAMS-dev/gamspy-examples/blob/master/models/trnsport/trnsport.py>`_. 
for illustration:::

    from gamspy import Container, Set, Equation
    m = Container()

    i = Set(m, name="i", records=["seattle", "san-diego"])
    j = Set(m, name="j", records=["new-york", "chicago", "topeka"])

    supply = Equation(
        m,
        name="supply",
        domain=[i],
        description="observe supply limit at plant i",
    )

    demand = Equation(
        m,
        name="demand",
        domain=[j],
        description="satisfy demand at market j",
    )

First a ``gamspy.Container`` is created and two sets ``i`` and ``j`` are added.
Those will be used as domain for the equations about to be declared.
Two instances of ``gamspy.Equation`` are created and assigned to Python
variables for later use. In this case both equations are one dimensional,
and use the sets ``i`` (``supply``) and ``j`` (``demand``) as domain.
In typical circumstances an indexed equation declaration implies that a block
of constraints will be generated. For example, equation ``supply`` implies that
two constraints will be generated, one for each element of the set ``i``.

.. _equation_definition:

Equation Definition
-------------------

After declaring equations they have to be defined. The definition of an
equation specifies its algebraic structure by using sets, parameters,
logical operators and functions. A definition is made
by assigning an expression of the form ``expression [==|>=|<=] expression``
to the Python variable that references the :meth:`Equation <gamspy.Equation>` instance. For
indexed equations, the index operator is used to specify the domain::
    
    equation[index_list] = expression [==|>=|<=] expression

The ``index_list`` consists of one or multiple sets which correspond to the
sets that were used when the equation was declared using the ``domain`` argument
of the ``Equation`` constructor. One or more logical conditions are optional.
After the assignment operator ``=``, the left hand side of the equation follows.
It is an arbitrary algebraic expression which may include variables, parameters,
functions, and constants among other items. The left hand side is followed by one
of the supported relational operators which define the relation between the left hand side
and the right hand side of the equation:

- ``==``: Equal
- ``>=``: Greater than or equal
- ``<=``: Less than or equal

Note that other operators like ``<``, ``>`` or ``!=`` are not supported. Furthermore
the operator is only significant in case of equations of ``type="regular"`` which is
the default. See the :ref:`equation_types` section for more details about the available
equation types.

A zero dimensional or scalar equation which is not declared over one or multiple sets
has to use the ellipsis literal ``[...]`` instead of the indexing operator like 
follows::

    equation[...] = expression [==|>=|<=] expression

.. note::
    Note that each equation has to be declared before it can be defined.

Scalar Equations
^^^^^^^^^^^^^^^^^

A scalar equation will produce one equation in the associated optimization problem.
The following is an example of a scalar equation definition from the `ramsey.py <https://github.com/GAMS-dev/gamspy-examples/blob/master/models/ramsey/ramsey.py>`_.
model::

    utility[...] = W == Sum(t, beta[t] * gams_math.log(C[t]))

The equation ``utility`` defined above is an example of a scalar equation that uses the scalar
variable ``W``. In addition, scalar equations may contain indexed variables like ``C``.
However, they must occur with an indexed operator such as :meth:`Sum<gamspy.Sum>` or :meth:`Product<gamspy.Product>`, unless the indexed
variables refer to a singleton set (a set with only one element).

Indexed Equations
^^^^^^^^^^^^^^^^^

All the set references in scalar equations are within the scope of indexed operators or
they refer to singleton sets, thus many variable, set and parameter references can be
included in one equation. In addition, GAMSPy also allows for equations to be defined
over a domain, thereby developing a compact representation for constraints. The
index sets to the left of the Python assignment operator ``=`` are called the domain
of definition of the equation.

.. note::
    - Domain checking ensures that the domain over which an equation is defined
      is the set (or the sets) or a subset of the set (or the sets) over which
      the equation was declared.
    - As a corollary, domain checking also catches the error of the indices being
      listed in an inconsistent order. For example, declaring an equation with ``domain=[s,t]``
      and then naming it in the definition as ``myequation[t,s]`` causes an error
      (unless ``s`` and ``t`` are aliases of the same set). For more information, see section
      `Domain Checking <https://www.gams.com/latest/docs/UG_SetDefinition.html#UG_SetDefinition_DomainChecking>`_ 
      in the GAMS documentation.

The following is an example of indexed equation definitions, again taken from the
`trnsport.py <https://github.com/GAMS-dev/gamspy-examples/blob/master/models/trnsport/trnsport.py>`_ model. Besides the already introduced sets ``i``
and ``j``, parameters ``a`` and ``b`` are used as well as the :meth:`Sum<gamspy.Sum>` operator::

    from gamspy Parameter, Sum

    capacities = [["seattle", 350], ["san-diego", 600]]
    demands = [["new-york", 325], ["chicago", 300], ["topeka", 275]]

    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)

    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= b[j]

Given the set ``i`` containing the elements ``"seattle"`` and ``"san-diego"``, the
following two individual equations are generated for ``supply``::

    supply["seattle"] = Sum(j, x["seattle", j]) <= a["seattle"]
    supply["san-diego"] = Sum(j, x["san-diego", j]) <= a["san-diego"]

For the equation ``demand``, the number of generated constraints in three::

    demand["new-york"] = Sum(i, x[i, "new-york"]) >= b["new-york"]
    demand["chicago"] = Sum(i, x[i, "chicago"]) >= b["chicago"]
    demand["topeka"] = Sum(i, x[i, "topeka"]) >= b["topeka"]

Combining Equation Declaration and Definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it can be handy to combine an equation declaration and definition.
This is possible by using the optional ``definition`` argument of
the ``Equation`` constructor. A combined declaration and definition of the
preceding example would look like follows::

    from gamspy import Container, Equation, Sum

    supply = Equation(
        m,
        name="supply",
        domain=[i],
        description="observe supply limit at plant i",
        definition=Sum(j, x[i, j]) <= a[i],
    )

    demand = Equation(
        m,
        name="demand",
        domain=[j],
        description="satisfy demand at market j",
        definition=Sum(i, x[i, j]) >= b[j],
    )

.. note::
    The arrangement of the terms in the equation is a matter of choice, but
    often a particular one is chosen because it makes the model easier to understand.

Using Labels Explicitly in Equations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it can be necessary to refer to specific set elements in equations.
This can be done as with parameters - by using quotes or double quotes around
the label. Consider the following example from the model `cta.py <https://github.com/GAMS-dev/gamspy-examples/blob/master/models/cta/cta.py>`_ where
the label ``"total"`` is used on the second index position of the variable ``t``
explicitly::

    addrow[i, k] = Sum(v[i, j, k], t[v]) == 2 * t[i, "total", k]

.. _logic_equations:

Logic Equations
^^^^^^^^^^^^^^^

Logic equations defined by using ``type="boolean"`` in the :meth:`Equation <gamspy.Equation>` constructor
use boolean algebra and have to evaluate to ``True`` (or ``1``) to be feasible. Most
boolean functions can be used with the a Python operator as well as an equivalent method
from :meth:`gamspy.math<gamspy.math>`, but some do exist in the latter only. The following
table gives an overview of the available boolean functions in GAMSPy:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Function
     - Operator
     - Evaluation
   * - Negation
     - ``~x`` or ``gamspy.math.bool_not(x)``
     - ``1`` if ``x==0``, else ``0``
   * - Logical conjunction
     - ``x and y`` or ``gamspy.math.bool_and(x,y)``
     - ``1`` if ``x!=0 and y!=0``, else ``0``
   * - Logical disjunction
     - ``x or y`` or ``gamspy.math.bool_or(x,y)``
     - ``0`` if ``x==y==0``, else ``1``
   * - Exclusive disjunction
     - ``x ^ y`` or ``gamspy.math.bool_xor(x,y)``
     - ``1`` if exactly one argument is ``!=0``, else ``0``
   * - Material implication
     - ``gamspy.math.bool_imp(x,y)``
     - ``0`` if ``x!=0 and y==0``, else ``1``
   * - Material equivalence
     - ``gamspy.math.bool_eqv(x,y)``
     - ``0`` if exactly one argument is ``0``, else ``1``

.. _equation_types:

Equation Types
--------------

Equations can have different types. Most of the time, the default ``type="regular"``
is sufficient, but there are other types for specific needs
and modelling practices. The following table gives an overview of the available
equation types in GAMSPy:
    
.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Type
     - Description
   * - ``"regular"``
     - This is the default equation type which is suitible for ordinary equations using the ``==``, ``>=``, and ``<=`` operators in the equation definition.
   * - ``nonbinding``
     - No relationship implied between left-hand side and right-hand side. This equation type is ideally suited for use in MCP models and in variational inequalities.
   * - ``external``
     - Equation is defined by external programs. See the section `External Equations <https://gamspy.readthedocs.io/en/latest/user/advanced/external_equations.html>`_ in the GAMS documentation.
   * - ``boolean``
     - Boolean equations. See the section :ref:`logic_equations`.

.. _equation_expressions:

Expressions in Equation Definitions
-----------------------------------

The arithmetic operators and some of the functions provided by GAMSPy
may be used in equation definitions. But also certain native Python
operators can be used. Consider the following example adapted from the model
`ramsey.py <https://github.com/GAMS-dev/gamspy-examples/blob/master/models/ramsey/ramsey.py>`_ demonstrating the use of parentheses and exponentiation::

    production[t] = Y[t] == a * (K[t] ** b) * (L[t] ** (1 - b))

Functions in Equation Definitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The functions provided by GAMSPy can be found in :meth:`gamspy.math<gamspy.math>`.
Note that some functions like :meth:`uniform<gamspy.math.uniform>` and
:meth:`normal<gamspy.math.normal>` are not allowed in equation definitions.
The use of the other functions is determined by the type of arguments in the model.
There are two types of arguments:

- **Exogenous arguments:** The arguments are known. :ref:`Parameters <parameter>` and
  :ref:`variable attributes<variable-attributes>` (for example, ``.l`` and ``.m`` attributes) are used
  as arguments. The expression is evaluated once when the model is being
  set up and most mathematical functions are allowed.

- **Endogenous arguments:** The arguments are variables and therefore unknown
  at the time of model setup. The function will be evaluated many times at
  intermediate points while the model is being solved. Note that the
  occurrence of any function with endogenous arguments implies that the
  model is not linear.

There are two types of functions allowing endogenous arguments: smooth functions
and discontinuous functions. Smooth functions are continuous functions with
continuous derivatives (like :meth:`sin<gamspy.math.sin>`,
:meth:`exp<gamspy.math.exp>`, :meth:`log<gamspy.math.log>`). Discontinuous functions
include continuous functions with discontinuous derivatives
(like :meth:`Max<gamspy.math.Max>`, :meth:`Min<gamspy.math.Min>`, :meth:`abs<gamspy.math.abs>`)
and discontinuous functions (like :meth:`ceil<gamspy.math.ceil>`, :meth:`sign<gamspy.math.sign>`).
Smooth functions may be used routinely in nonlinear models. However, discontinuous
functions may cause numerical problems and should be used only if unavoidable,
and only in a special model type called ``DNLP``. For more details on model types see
:ref:`Model documentation<Model>`.

.. note::
    The best way to model discontinuous functions is with binary variables.
    The result is a model of the type ``MINLP``. The GAMS model
    `absmip <https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_absmip.html>`_
    demonstrates this formulation technique for the functions ``abs``, ``min``,
    ``max`` and ``sign``. See also section `Reformulating DNLP Models <https://www.gams.com/latest/docs/UG_NLP_GoodFormulations.html#UG_NLP_GoodFormulations_ReformulatingDNLPModels>`_ in the GAMS documentation.
    We strongly discourage the use of the ``DNLP`` model type.


Preventing Undefined Operations in Equations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some operations are not defined at particular values of the arguments. Two examples
are division by ``0`` and the ``log`` of ``0``. While this can easily be identified
at model setup for exogenous functions and expressions, it is a lot more difficult
when the terms involve variables. The expression may be evaluated many times when
the problem is being solved and the undefined result may arise only under certain
cases. One way to avoid an expression becoming undefined is adding bounds to the
respective variables. Consider the following example from the `ramsey.py <https://github.com/GAMS-dev/gamspy-examples/blob/master/models/ramsey/ramsey.py>`_.
model::

    C.lo[t] = 0.001
    utility[...] = W == Sum(t, beta[t] * gams_math.log(C[t]))

Specifying a lower bound for ``C[t]`` that is slightly larger than ``0``
prevents the ``log`` function from becoming undefined.

.. _equation-attributes:

Equation Attributes
-------------------

Similar to variables, equations have five attributes. Five values are
associated with each unique label combination of every equation. They
are denoted by the properties ``.l``, ``.m``, ``.lo``, ``.up`` and
``.scale``. A list of the attributes and their description is given in
the following table:

.. list-table::
   :widths: 25 10 65
   :header-rows: 1

   * - Equation Attribute
     - Property
     - Description
   * - Lower bound
     - ``.lo``
     - Negative infinity for ``<=`` equations. Right hand side value for
       ``>=``, ``==``, and ``type="boolean"`` equations. 
   * - Upper bound 
     - ``.up``
     - Right hand
       side value for ``<=``, ``==``, and ``type="boolean"`` equations.
   * - Equation level 
     - ``.l``
     - Level of the equation in the current solution, equal to the level of all
       terms involving variables.
   * - Marginal
     - ``.m``
     - Marginal value for equation. This attribute is reset to a new value when
       a model containing the equation is solved. The marginal value for an
       equation is also known as the shadow price for the equation and in
       general not defined before solution but if present it can help to
       provide a basis for the model 
   * - Scale factor
     - ``.scale``
     - Numerical scaling factor that scales all coefficients in the equation.
       This is only used when the model attribute ``scaleopt`` is set to ``1``.
   * - Stage
     - ``.stage``
     - This attribute allows to assign equations to stages in a stochastic
       program or other block structured model. Its current use is limited to
       2-stage stochastic programs solved with ``DECIS``.

Note that all properties except for ``.scale`` and ``.stage`` contain the
attribute values of equations after a solution of the model has been obtained.
For some solvers it can be useful to specify marginal values ``.m`` and level
values ``.l`` on input to provide starting information. Also note that the
marginal value is also known as the dual or shadow price. Roughly speaking, the
marginal value ``.m`` of an equation is the amount by which the value of the
objective variable would change if the equation level were moved one unit.

Equation attributes may be referenced in expressions and can be used to specify
starting values. In addition, they serve for scaling purposes and for reporting
after a model was solved. Here the attributes are not accessed via the Python
properties, but are contained in the data of the equation itself which can be
retrieved via the ``records`` property as the following example shows::

    transport = Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )
    transport.solve()
    print(supply.records)

::

               i  level  marginal  lower  upper  scale
    0    seattle  350.0      -0.0   -inf  350.0    1.0
    1  san-diego  550.0       0.0   -inf  600.0    1.0

The level values of the equation ``supply`` are displayed. As expected, there
are two level values, one for each member of the set ``i`` over which the
equation ``supply`` was defined.

In addition to the equation attributes introduced above, there are a number of
equation attributes that cannot be assigned but may be used in computations.
They are given in the following table:

.. list-table::
   :widths: 25 10 65
   :header-rows: 1

   * - Equation Attribute
     - Property
     - Description
   * - Range
     - ``.range``
     - The difference between the lower and upper bounds of an equation.
   * - Slack lower bound
     - ``.slacklo``
     - Slack from equation lower bound. This is defined as the greater of two
       values: zero or the difference between the level value and the lower
       bound of an equation.
   * - Slack upper bound
     - ``.slackup``
     - Slack from equation upper bound. This is defined as the greater of two
       values: zero or the difference between the upper bound and the level
       value of an equation.
   * - Slack
     - ``.slack``
     - Minimum slack from equation bound. This is defined as the minimum of two
       values: the slack from equation lower bound and the slack from equation
       upper bound.
   * - Infeasibility
     - ``.infeas``
     - Amount by which an equation is infeasible falling below its lower bound
       or above its upper bound. This is defined as max(0, lower bound - level, level - upper bound). 

`Equation` attributes can be assigned just like `Variable` attributes. For example to assign an initial value
to a scalar equation: ::

  import gamspy as gp
  m = gp.Container()
  e = gp.Equation(m, "e")
  e.l = 5

or to assign an initial value to an equation with non-scalar domain: ::

  import gamspy as gp
  m = gp.Container()
  i = gp.Set(m, "i", records=['i1', 'i2'])
  e = gp.Equation(m, "e", domain=[i])
  e.l[i] = 5


.. _inspecting_generated_equations:

Inspecting Generated Equations
------------------------------

The generated equations can be inspected by using :meth:`getEquationListing() <gamspy.Equation.getEquationListing>`
function after solving the model. Note that by studying the equation listing the user may determine whether the 
model generated by GAMS is the the model that the user has intended - an extremely important question. The equation
listing can be filtered with ``filters`` argument, the number of equations returned can be limited with ``n`` argument, and Infeasibilities
above a certain threshold can be filtered with ``infeasibility_threshold`` argument.

For example, in `Mexico Steel sector model <https://github.com/GAMS-dev/gamspy/blob/develop/tests/integration/models/mexss.py>`_ 
market requirements equation ``mr`` is defined over markets ``j`` which contain 3 elements and commodities ``cf`` which contain 
one element. If one prints the equation listing directly, ``getEquationListing`` would return all three generated equations. ::

  import gamspy as gp
  m = gp.Container()
  ...
  ...
  model_definition_goes_here
  ...
  ...
  model.solve(options=Options(equation_listing_limit=100))
  print(mr.getEquationListing())

Generated equations: ::

  mr(steel,mexico-df)..  x(steel,ahmsa,mexico-df) + x(steel,fundidora,mexico-df) + x(steel,sicartsa,mexico-df) + x(steel,hylsa,mexico-df) + x(steel,hylsap,mexico-df) + v(steel,mexico-df) =G= 4.01093 ; (LHS = 0, INFES = 4.01093 ****)
  mr(steel,monterrey)..  x(steel,ahmsa,monterrey) + x(steel,fundidora,monterrey) + x(steel,sicartsa,monterrey) + x(steel,hylsa,monterrey) + x(steel,hylsap,monterrey) + v(steel,monterrey) =G= 2.18778 ; (LHS = 0, INFES = 2.18778 ****)
  mr(steel,guadalaja)..  x(steel,ahmsa,guadalaja) + x(steel,fundidora,guadalaja) + x(steel,sicartsa,guadalaja) + x(steel,hylsa,guadalaja) + x(steel,hylsap,guadalaja) + v(steel,guadalaja) =G= 1.09389 ; (LHS = 0, INFES = 1.09389 ****)

One can alternatively filter certain equations by using the ``filters`` argument. For example, if one only wants to see 
the equations for monterrey market, they can provide the elements as follows: ::

  import gamspy as gp
  m = gp.Container()
  ...
  ...
  model_definition_goes_here
  ...
  ...
  model.solve(options=Options(equation_listing_limit=100))
  print(mr.getEquationListing(filters=[[], ['monterrey']]))

``filters`` argument is a list of lists where each list specifies the elements to be gathered. 
If an empty list is given as in the example above, it means all elements. 

Number of equations returned can be filtered with ``n`` argument. For example, if ``n`` is set to 1,
the function return only the first equation.

If one wants to ignore equations that have an infeasibility above a certain threshold, they can 
specify ``infeasibility_threshold`` argument. Any equation that has higher infeasibility than
infeasibility_threshold will be filtered out.

.. note::

  Length of the ``filters`` argument must be equal to the dimension of the equation.