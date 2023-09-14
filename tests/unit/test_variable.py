import unittest

import pandas as pd

import gamspy._symbols.implicits as implicits
from gamspy import Container
from gamspy import Set
from gamspy import Variable
from gamspy import VariableType


class VariableSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_variable_string(self):
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
        pi = Variable(
            self.m,
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

        i = Set(self.m, name="i", records=["bla", "damn"])
        test = Variable(self.m, "test", domain=[i])
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

        k = Set(self.m, "k")
        x = Variable(self.m, "x", domain=[k])
        x.l[k] = 5
        self.assertEqual(
            list(self.m._statements_dict.values())[-1].gamsRepr(),
            "x.l(k) = 5;",
        )
        self.assertTrue(x._is_dirty)

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
