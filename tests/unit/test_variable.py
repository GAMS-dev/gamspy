from __future__ import annotations

import os
import unittest

import pandas as pd

import gamspy._symbols.implicits as implicits
from gamspy import Container
from gamspy import Equation
from gamspy import Set
from gamspy import Variable
from gamspy import VariableType
from gamspy.exceptions import ValidationError


class VariableSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )

    def test_variable_creation(self):
        # no name
        self.assertRaises(TypeError, Variable, self.m)

        # non-str type name
        self.assertRaises(TypeError, Variable, self.m, 5)

        # no container
        self.assertRaises(TypeError, Variable)

        # non-container type container
        self.assertRaises(TypeError, Variable, 5, "j")

        # try to create a symbol with same name but different type
        _ = Set(self.m, "i")
        self.assertRaises(TypeError, Variable, self.m, "i")

        # get already created symbol
        j1 = Variable(self.m, "j")
        j2 = Variable(self.m, "j")
        self.assertEqual(id(j1), id(j2))

        # Variable and domain containers are different
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )
        set1 = Set(self.m, "set1")
        with self.assertRaises(ValidationError):
            _ = Variable(m, "var1", domain=[set1])

    def test_variable_string(self):
        # Check if the name is reserved
        self.assertRaises(ValidationError, Variable, self.m, "set")

        # Set
        i = Set(self.m, name="i", records=["bla", "damn"])
        j = Set(self.m, name="j", records=["test", "test2"])

        # Variable without data
        v4 = Variable(self.m, "v4")
        self.assertEqual(v4.gamsRepr(), "v4")
        self.assertEqual(
            v4.getStatement(),
            "free Variable v4;",
        )

        with self.assertRaises(TypeError):
            v4.records = 5

        # Variable without domain
        v0 = Variable(self.m, name="v0", description="some text")
        self.assertEqual(v0.gamsRepr(), "v0")
        self.assertEqual(
            v0.getStatement(),
            'free Variable v0 "some text";',
        )

        expression = -v0
        self.assertEqual(expression.name, "-v0")

        # Variable one domain
        v1 = Variable(self.m, name="v1", domain=[i])
        self.assertEqual(v1.gamsRepr(), "v1")
        self.assertEqual(
            v1.getStatement(),
            "free Variable v1(i);",
        )

        self.assertEqual((v1[i] == v1[i]).gamsRepr(), "v1(i) =e= v1(i)")

        # Variable two domain
        v2 = Variable(self.m, name="v2", domain=[i, j])
        self.assertEqual(v2.gamsRepr(), "v2")
        self.assertEqual(
            v2.getStatement(),
            "free Variable v2(i,j);",
        )

        # Scalar variable with records
        pi = Variable(
            self.m,
            "pi",
            records=pd.DataFrame(data=[3.14159], columns=["level"]),
        )
        self.assertEqual(
            pi.getStatement(),
            "free Variable pi;",
        )
        new_pi = -pi
        self.assertEqual(new_pi.gamsRepr(), "-pi")

        # 1D variable with records
        v = Variable(
            self.m,
            "v",
            "free",
            domain=["*"],
            records=pd.DataFrame(
                data=[("i" + str(i), i) for i in range(5)],
                columns=["domain", "marginal"],
            ),
        )
        self.assertEqual(
            v.getStatement(),
            "free Variable v(*);",
        )

        v3 = Variable(
            self.m,
            "v3",
            "positive",
            ["*", "*"],
            records=pd.DataFrame(
                [("seattle", "san-diego"), ("chicago", "madison")]
            ),
        )
        self.assertEqual(
            v3.getStatement(),
            "positive Variable v3(*,*);",
        )

    def test_variable_types(self):
        i = Set(self.m, "i", records=["1", "2"])

        v = Variable(self.m, name="v", type="Positive")
        self.assertEqual(
            v.getStatement(),
            "positive Variable v;",
        )

        v1 = Variable(self.m, name="v1", type="Negative")
        self.assertEqual(
            v1.getStatement(),
            "negative Variable v1;",
        )

        v2 = Variable(self.m, name="v2", type="Binary")
        self.assertEqual(
            v2.getStatement(),
            "binary Variable v2;",
        )

        v3 = Variable(self.m, name="v3", domain=[i], type="Integer")
        self.assertEqual(
            v3.getStatement(),
            "integer Variable v3(i);",
        )

        self.assertEqual(str(VariableType.FREE), "free")
        self.assertEqual(
            VariableType.values(),
            [
                "binary",
                "integer",
                "positive",
                "negative",
                "free",
                "sos1",
                "sos2",
                "semicont",
                "semiint",
            ],
        )

        v4 = Variable(
            self.m, name="v4", domain=[i], type=VariableType.POSITIVE
        )
        self.assertEqual(v4.type, "positive")

    def test_variable_attributes(self):
        m = Container()
        pi = Variable(
            m,
            "pi",
            records=pd.DataFrame(data=[3.14159], columns=["level"]),
        )

        self.assertTrue(
            hasattr(pi, "l") and isinstance(pi.l, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.l.gamsRepr(), "pi.l")

        self.assertTrue(
            hasattr(pi, "m") and isinstance(pi.m, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.m.gamsRepr(), "pi.m")

        self.assertTrue(
            hasattr(pi, "lo")
            and isinstance(pi.lo, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.lo.gamsRepr(), "pi.lo")

        self.assertTrue(
            hasattr(pi, "up")
            and isinstance(pi.up, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.up.gamsRepr(), "pi.up")

        self.assertTrue(
            hasattr(pi, "scale")
            and isinstance(pi.scale, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.scale.gamsRepr(), "pi.scale")

        self.assertTrue(
            hasattr(pi, "fx")
            and isinstance(pi.fx, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.fx.gamsRepr(), "pi.fx")

        self.assertTrue(
            hasattr(pi, "prior")
            and isinstance(pi.prior, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.prior.gamsRepr(), "pi.prior")

        self.assertTrue(
            hasattr(pi, "stage")
            and isinstance(pi.stage, implicits.ImplicitParameter)
        )
        self.assertEqual(pi.stage.gamsRepr(), "pi.stage")

        i = Set(m, name="i", records=["bla", "damn"])
        test = Variable(m, "test", domain=[i])
        self.assertTrue(
            hasattr(test, "l")
            and isinstance(test.l, implicits.ImplicitParameter)
        )
        self.assertTrue(
            hasattr(test, "m")
            and isinstance(test.m, implicits.ImplicitParameter)
        )
        self.assertTrue(
            hasattr(test, "lo")
            and isinstance(test.lo, implicits.ImplicitParameter)
        )
        self.assertTrue(
            hasattr(test, "up")
            and isinstance(test.up, implicits.ImplicitParameter)
        )
        self.assertTrue(
            hasattr(test, "scale")
            and isinstance(test.scale, implicits.ImplicitParameter)
        )
        self.assertTrue(
            hasattr(test, "fx")
            and isinstance(test.fx, implicits.ImplicitParameter)
        )
        self.assertTrue(
            hasattr(test, "prior")
            and isinstance(test.prior, implicits.ImplicitParameter)
        )
        self.assertTrue(
            hasattr(test, "stage")
            and isinstance(test.stage, implicits.ImplicitParameter)
        )

        k = Set(m, "k")
        x = Variable(m, "x", domain=[k])
        x.l[k] = 5

        self.assertEqual(
            x._assignment.gamsRepr(),
            "x.l(k) = 5;",
        )
        self.assertFalse(x._is_dirty)

    def test_scalar_attr_assignment(self):
        a = Variable(self.m, "a")
        b = Variable(self.m, "b", "binary")
        a.l = 5
        self.assertEqual(a._assignment.getStatement(), "a.l = 5;")

        a.m = 5
        self.assertEqual(a._assignment.getStatement(), "a.m = 5;")

        a.lo = 5
        self.assertEqual(a._assignment.getStatement(), "a.lo = 5;")

        a.up = 5
        self.assertEqual(a._assignment.getStatement(), "a.up = 5;")

        a.scale = 5
        self.assertEqual(a._assignment.getStatement(), "a.scale = 5;")

        a.fx = 5
        self.assertEqual(a._assignment.getStatement(), "a.fx = 5;")

        b.prior = 5
        self.assertEqual(b._assignment.getStatement(), "b.prior = 5;")

        a.stage = 5
        self.assertEqual(a._assignment.getStatement(), "a.stage = 5;")

    def test_implicit_variable(self):
        i = Set(self.m, "i", records=[f"i{i}" for i in range(10)])
        a = Variable(self.m, "a", "free", [i])
        a.generateRecords()
        self.assertTrue(a.isValid())

        expression = -a[i] * 5
        self.assertEqual(expression.gamsRepr(), "(-a(i) * 5)")

        self.assertTrue(
            hasattr(a[i], "l")
            and isinstance(a[i].l, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].l.gamsRepr(), "a(i).l")
        self.assertTrue(
            hasattr(a[i], "m")
            and isinstance(a[i].m, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].m.gamsRepr(), "a(i).m")
        self.assertTrue(
            hasattr(a[i], "lo")
            and isinstance(a[i].lo, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].lo.gamsRepr(), "a(i).lo")
        self.assertTrue(
            hasattr(a[i], "up")
            and isinstance(a[i].up, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].up.gamsRepr(), "a(i).up")
        self.assertTrue(
            hasattr(a[i], "scale")
            and isinstance(a[i].scale, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].scale.gamsRepr(), "a(i).scale")
        self.assertTrue(
            hasattr(a[i], "fx")
            and isinstance(a[i].fx, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].fx.gamsRepr(), "a(i).fx")
        self.assertTrue(
            hasattr(a[i], "prior")
            and isinstance(a[i].prior, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].prior.gamsRepr(), "a(i).prior")
        self.assertTrue(
            hasattr(a[i], "stage")
            and isinstance(a[i].stage, implicits.ImplicitParameter)
        )
        self.assertEqual(a[i].stage.gamsRepr(), "a(i).stage")

        a.l[...] = 5

        self.assertEqual(
            a.records.level.to_list(),
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        )

    def test_assignment_dimensionality(self):
        j1 = Set(self.m, "j1")
        j2 = Set(self.m, "j2")
        j3 = Variable(self.m, "j3", domain=[j1, j2])
        j4 = Set(self.m, "j4")

        e1 = Equation(self.m, "e1", domain=[j1, j2])

        with self.assertRaises(ValidationError):
            e1[j1, j2] = j3[j1, j2, j4] * 5 <= 5

    def test_type(self):
        gamma = Variable(self.m, "gamma")
        gamma.type = VariableType.BINARY
        self.assertEqual(gamma.type, "binary")

        var1 = Variable(self.m, "var1")
        var1.type = VariableType.FREE
        self.assertEqual(var1.type, "free")

        var2 = Variable(self.m, "var2")
        var2.type = VariableType.POSITIVE
        self.assertEqual(var2.type, "positive")

        var3 = Variable(self.m, "var3")
        var3.type = VariableType.NEGATIVE
        self.assertEqual(var3.type, "negative")

        var4 = Variable(self.m, "var4")
        var4.type = VariableType.NEGATIVE
        self.assertEqual(var4.type, "negative")

        var5 = Variable(self.m, "var5")
        var5.type = VariableType.SEMICONT
        self.assertEqual(var5.type, "semicont")

    def test_uels_on_axes(self):
        s = pd.Series(index=["a", "b", "c"], data=[i + 1 for i in range(3)])
        v = Variable(self.m, "v", domain=["*"], records=s, uels_on_axes=True)
        self.assertEqual(v.records.level.tolist(), [1, 2, 3])


def variable_suite():
    suite = unittest.TestSuite()
    tests = [
        VariableSuite(name)
        for name in dir(VariableSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(variable_suite())
