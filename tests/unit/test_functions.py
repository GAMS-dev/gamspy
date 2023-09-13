import unittest

from gamspy import Container
from gamspy import Equation
from gamspy import Parameter
from gamspy import Set
from gamspy import Variable
from gamspy.functions import beta
from gamspy.functions import entropy
from gamspy.functions import errorf
from gamspy.functions import gamma
from gamspy.functions import ifthen
from gamspy.functions import lse_max
from gamspy.functions import lse_max_sc
from gamspy.functions import lse_min
from gamspy.functions import lse_min_sc
from gamspy.functions import ncp_cm
from gamspy.functions import ncp_f
from gamspy.functions import ncpVUpow
from gamspy.functions import ncpVUsin
from gamspy.functions import poly
from gamspy.functions import rand_binomial
from gamspy.functions import rand_linear
from gamspy.functions import rand_triangle
from gamspy.functions import regularized_beta
from gamspy.functions import regularized_gamma
from gamspy.functions import slrec
from gamspy.functions import sqrec


class FunctionsSuite(unittest.TestCase):
    def test_logical(self):
        m = Container()

        o = Set(m, "o", records=[f"pos{idx}" for idx in range(1, 11)])
        p = Set(m, "p", records=[f"opt{idx}" for idx in range(1, 6)])
        sumc = Variable(m, "sumc", domain=[o, p])
        op = Variable(m, "op", domain=[o, p])
        defopLS = Equation(m, "defopLS", domain=[o, p])
        defopLS[o, p] = op[o, p] == ifthen(sumc[o, p] >= 0.5, 1, 0)
        self.assertEqual(
            list(m._statements_dict.values())[-1].gamsRepr(),
            "defopLS(o,p) .. op(o,p) =e= (ifthen(sumc(o,p) >= 0.5, 1, 0)  );",
        )

    def test_functions(self):
        m = Container()
        i = Set(m, "i", records=["1", "2"])
        a = Parameter(m, "a", domain=[i], records=[("1", 1), ("2", 2)])

        op1 = entropy(1)
        self.assertEqual(op1.gamsRepr(), "(entropy( 1 ))")

        op1 = beta(1, 2)
        self.assertEqual(op1.gamsRepr(), "(beta( 1,2 ))")

        op1 = regularized_beta(1, 2, 3)
        self.assertEqual(op1.gamsRepr(), "(betaReg( 1,2,3 ))")

        op1 = gamma(1, 2)
        self.assertEqual(op1.gamsRepr(), "(gamma( 1,2 ))")

        op1 = regularized_gamma(1, 2, 3)
        self.assertEqual(op1.gamsRepr(), "(gammaReg( 1,2,3 ))")

        op1 = lse_max(a[i])
        self.assertEqual(op1.gamsRepr(), "(lseMax( a(i) ))")

        op1 = lse_max_sc(a[i], a[i])
        self.assertEqual(op1.gamsRepr(), "(lseMaxSc( a(i),a(i) ))")

        op1 = lse_min(a[i])
        self.assertEqual(op1.gamsRepr(), "(lseMin( a(i) ))")

        op1 = lse_min_sc(a[i], a[i])
        self.assertEqual(op1.gamsRepr(), "(lseMinSc( a(i),a(i) ))")

        op1 = ncp_cm(a[i], a[i], 3)
        self.assertEqual(op1.gamsRepr(), "(ncpCM( a(i),a(i),3 ))")

        op1 = ncp_f(a[i], a[i])
        self.assertEqual(op1.gamsRepr(), "(ncpF( a(i),a(i),0 ))")

        op1 = ncpVUpow(a[i], a[i])
        self.assertEqual(op1.gamsRepr(), "(ncpVUpow( a(i),a(i),0 ))")

        op1 = ncpVUsin(a[i], a[i])
        self.assertEqual(op1.gamsRepr(), "(ncpVUsin( a(i),a(i),0 ))")

        op1 = poly(a[i], 3, 5)
        self.assertEqual(op1.gamsRepr(), "(poly( a(i),3,5 ))")

        op1 = rand_binomial(1, 2)
        self.assertEqual(op1.gamsRepr(), "(randBinomial( 1,2 ))")

        op1 = rand_linear(1, 2, 3)
        self.assertEqual(op1.gamsRepr(), "(randLinear( 1,2,3 ))")

        op1 = rand_triangle(1, 2, 3)
        self.assertEqual(op1.gamsRepr(), "(randTriangle( 1,2,3 ))")

        op1 = slrec(a[i])
        self.assertEqual(op1.gamsRepr(), "(slrec( a(i),1e-10 ))")

        op1 = sqrec(a[i])
        self.assertEqual(op1.gamsRepr(), "(sqrec( a(i),1e-10 ))")

        op1 = errorf(a[i])
        self.assertEqual(op1.gamsRepr(), "(errorf( a(i) ))")


def functions_suite():
    suite = unittest.TestSuite()
    tests = [
        FunctionsSuite(name)
        for name in dir(FunctionsSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(functions_suite())
