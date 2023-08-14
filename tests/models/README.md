# GAMSpy

## Set, Alias, Parameter, Variable, Equation

You can create a Set, Alias, Parameter, Variable, and Equation in the same way you create a Gams Transfer symbol. You don't need to change anything. Now, these symbols can be used in creating expressions.

```Python
import gamspy as gp

m = gp.Container()
i = gp.Set(m, "i", records=['i1','i2'])
a = gp.Parameter(m, 'a', domain=[i], records=[['i1','1'], ['i1','1']])
```

### Model

A model is a list of equations. 

```Python
from gamspy import Model

m = gp.Container()
model1 = Model(m, name="model1", equations=[maxw, minw, etd]) # Defines equations explicitly as a list of equations
model2 = Model(m, name="model2", equations=m.getEquations())             # This includes all defined equations
```

### Sum/Product/Smin/Smax

Frequently used Gams operations which accept an index list and an expression.

```Python
from gamspy import Sum, Product, Smin, Smax

m = gp.Container()
i = gp.Set(m, "i", records=['i1','i2'])
a = gp.Parameter(m, 'a', domain=[i], records=[['i1','1'], ['i1','1']])

supply = gp.Equation(m, name="supply", domain=[i])
supply[i] = Sum(i, a[i]) <= a[i]
```

### Card/Ord

Python representation of Card and Ord operations.

```Python
from gamspy import Card, Ord

    m = Container()

    i = Set(m, name="i", records=[str(idx) for idx in range(0, 181)])
    step = Parameter(m, name="step", records=math.pi / 180)
    omega = Parameter(m, name="omega", domain=[i])
    omega[i] = (Ord(i) - 1) * step
```

### Domain

This class is exclusively for conditioning on a domain with more than one set.

```Python
equation = gp.Equation(name="equation", domain=[i,j])
equation[i,j] = Sum(Domain(i,j).where[i], a[i] + b[j]) # Equivalent to equation(i,j) = Sum((i,j)$(i), a(i) + b(j))
```

### Number

This is for conditions on numbers or yes/no statements.

```Python
from gamspy import Number

Number(1).where[sig[i] == 0] # Equivalent to 1$(sig(i) = 0). It is also equivalent to yes$(sig(i) = 0)
```

### math package

This package is for the mathematical operations of GAMS over a domain.

```
import gamspy.math as gams_math
import math

sigma = Variable(m, name="sigma", domain=[i, k], type="Positive")
sigma.l[i, k] = uniform(0.1, 1) # Generates a different value from uniform distribution for each element of the domain.
sigma.l[i, k] = math.uniform(0.1, 1) # This is not equivalent to the statement above. This generates only one value for the whole domain.
```

### Logical Operations

Since it is not possible in Python to overload keywords such as **and**, **or**, and **not**, you need to use bitwise operatiors **&**, **|**, and **~**.

Mapping:

- **and** -> &
- **or**  -> |
- **not** -> ~

```Python
error01[s1,s2] = (rt[s1,s2] != 0) & (lfr[s1,s2] == 0) | ((rt[s1,s2] == 0) & (lfr[s1,s2] != 0))
```
