from __future__ import annotations

import os
import unittest

import pandas as pd

from gamspy import Alias
from gamspy import Card
from gamspy import Container
from gamspy import Ord
from gamspy import Parameter
from gamspy import Set
from gamspy import UniverseAlias
from gamspy.exceptions import ValidationError


class SetSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
        )

    def test_set_creation(self):
        # no name
        self.assertRaises((ValidationError, TypeError), Set, self.m)

        # non-str type name
        self.assertRaises(TypeError, Set, self.m, 5)

        # no container
        self.assertRaises((ValidationError, TypeError), Set)

        # non-container type container
        self.assertRaises(TypeError, Set, 5, "j")

        # try to create a symbol with same name but different type
        _ = Parameter(self.m, "i")
        self.assertRaises(TypeError, Set, self.m, "i")

        # get already created symbol
        j1 = Set(self.m, "j")
        j2 = Set(self.m, "j")
        self.assertEqual(id(j1), id(j2))

        # Set and domain containers are different
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
        )
        set1 = Set(self.m, "set1")
        with self.assertRaises(ValidationError):
            _ = Set(m, "set2", domain=[set1])

    def test_set_string(self):
        # Check if the name is reserved
        self.assertRaises(ValidationError, Set, self.m, "set")

        # Without records
        b = Set(self.m, "b")
        self.assertEqual(b.gamsRepr(), "b")
        self.assertEqual(b.getStatement(), "Set b(*);")

        # Without domain
        i = Set(
            self.m,
            "i",
            records=["seattle", "san-diego"],
            description="dummy set",
        )
        self.assertEqual(i.gamsRepr(), "i")
        self.assertEqual(
            i.getStatement(),
            'Set i(*) "dummy set";',
        )

        # With one domain
        j = Set(self.m, "j", records=["seattle", "san-diego", "california"])
        k = Set(self.m, "k", domain=[j], records=["seattle", "san-diego"])
        self.assertEqual(k.gamsRepr(), "k")
        self.assertEqual(k.getStatement(), "Set k(j);")

        # With two domain
        m = Set(self.m, "m", records=[f"i{i}" for i in range(2)])
        n = Set(self.m, "n", records=[f"j{i}" for i in range(2)])
        a = Set(self.m, "a", [m, n])
        a.generateRecords(density=1)
        self.assertEqual(a.gamsRepr(), "a")
        self.assertEqual(a.getStatement(), "Set a(m,n);")

        s = Set(self.m, "s", is_singleton=True)
        self.assertEqual(
            s.getStatement(),
            "Singleton Set s(*);",
        )

        self.assertRaises(
            ValidationError, Set, self.m, "s2", None, True, ["i1", "i2"]
        )

    def test_records_assignment(self):
        new_cont = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )
        i = Set(self.m, "i")
        j = Set(self.m, "j", domain=[i])
        k = Set(new_cont, "k")

        s = Set(self.m, "s", domain=[i])
        with self.assertRaises(TypeError):
            s.records = 5

        with self.assertRaises(ValidationError):
            j[k] = 5

    def test_set_operators(self):
        i = Set(self.m, "i", records=["seattle", "san-diego"])
        card = Card(i)
        self.assertEqual(card.gamsRepr(), "card(i)")

        ord = Ord(i)
        self.assertEqual(ord.gamsRepr(), "ord(i)")

    def test_implicit_sets(self):
        m = Container(
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False))
        )
        j = Set(m, "j", records=["seattle", "san-diego", "california"])
        k = Set(m, "k", domain=[j], records=["seattle", "san-diego"])

        expr = k[j] <= k[j]
        self.assertEqual(expr.gamsRepr(), "(k(j) <= k(j))")
        expr = k[j] >= k[j]
        self.assertEqual(expr.gamsRepr(), "(k(j) >= k(j))")

        k[j] = ~k[j]

        if m.delayed_execution:
            self.assertEqual(
                m._unsaved_statements[-1].gamsRepr(),
                "k(j) = ( not k(j));",
            )

    def test_set_operations(self):
        i = Set(self.m, "i", records=["seattle", "san-diego"])
        k = Set(self.m, "k", records=["seattle", "san-diego"])
        union = i + k
        self.assertEqual(union.gamsRepr(), "i + k")

        intersection = i * k
        self.assertEqual(intersection.gamsRepr(), "i * k")

        complement = ~i
        self.assertEqual(complement.gamsRepr(), "( not i)")

        difference = i - k
        self.assertEqual(difference.gamsRepr(), "i - k")

    def test_dynamic_sets(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
        )
        i = Set(m, name="i", records=[f"i{idx}" for idx in range(1, 4)])
        i["i1"] = False

        if m.delayed_execution:
            self.assertEqual(
                m._unsaved_statements[-1].getStatement(),
                'i("i1") = no;',
            )

        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
        )
        k = Set(m, name="k", records=[f"k{idx}" for idx in range(1, 4)])
        k["k1"] = False

        if m.delayed_execution:
            self.assertTrue(k._is_dirty)
        else:
            self.assertFalse(k._is_dirty)

    def test_lag_and_lead(self):
        set = Set(
            self.m, name="S", records=["a", "b", "c"], description="Test text"
        )
        alias = Alias(self.m, "A", alias_with=set)

        # Circular lag
        new_set = set.lag(n=5, type="circular")
        self.assertEqual(new_set.name, "S -- 5")
        new_set = alias.lag(n=5, type="circular")
        self.assertEqual(new_set.name, "A -- 5")

        # Circular lead
        new_set = set.lead(n=5, type="circular")
        self.assertEqual(new_set.name, "S ++ 5")
        new_set = alias.lead(n=5, type="circular")
        self.assertEqual(new_set.name, "A ++ 5")

        # Linear lag
        new_set = set.lag(n=5, type="linear")
        self.assertEqual(new_set.name, "S - 5")
        new_set = alias.lag(n=5, type="linear")
        self.assertEqual(new_set.name, "A - 5")

        # Linear lead
        new_set = set.lead(n=5, type="linear")
        self.assertEqual(new_set.name, "S + 5")
        new_set = alias.lead(n=5, type="linear")
        self.assertEqual(new_set.name, "A + 5")

        # Incorrect type
        self.assertRaises(ValueError, set.lead, 5, "bla")
        self.assertRaises(ValueError, alias.lead, 5, "bla")
        self.assertRaises(ValueError, set.lag, 5, "bla")
        self.assertRaises(ValueError, alias.lag, 5, "bla")

        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
        )
        s = Set(m, name="s", records=[f"s{i}" for i in range(1, 4)])
        t = Set(m, name="t", records=[f"t{i}" for i in range(1, 6)])

        sMinDown = Set(m, name="sMinDown", domain=[s, t])
        sMinDown[s, t.lead(Ord(t) - Ord(s))] = 1

        if m.delayed_execution:
            self.assertEqual(
                m._unsaved_statements[-1].gamsRepr(),
                "sMinDown(s,t + (ord(t) - ord(s))) = 1;",
            )

    def test_set_attributes(self):
        i = Set(self.m, "i")
        self.assertEqual(i.pos.gamsRepr(), "i.pos")
        self.assertEqual(i.ord.gamsRepr(), "i.ord")
        self.assertEqual(i.off.gamsRepr(), "i.off")
        self.assertEqual(i.rev.gamsRepr(), "i.rev")
        self.assertEqual(i.uel.gamsRepr(), "i.uel")
        self.assertEqual(i.len.gamsRepr(), "i.len")
        self.assertEqual(i.tlen.gamsRepr(), "i.tlen")
        self.assertEqual(i.val.gamsRepr(), "i.val")
        self.assertEqual(i.tval.gamsRepr(), "i.tval")
        self.assertEqual(i.first.gamsRepr(), "i.first")
        self.assertEqual(i.last.gamsRepr(), "i.last")

    def test_iterable(self):
        # Set with no records
        i = Set(self.m, "i")
        count = 0
        for _ in i:
            count += 1

        self.assertEqual(count, 0)

        # Set with records
        k = Set(self.m, "k", records=[str(idx) for idx in range(1, 3)])
        count = 0
        for _ in k:
            count += 1

        self.assertEqual(count, 2)

        # Alias with no records
        x = Set(self.m, "x")
        a = Alias(self.m, "a", x)
        count = 0
        for _ in a:
            count += 1

        self.assertEqual(count, 0)

        # Alias with records
        b = Set(self.m, "b", records=[str(idx) for idx in range(1, 3)])
        c = Alias(self.m, "c", b)

        count = 0
        for _ in c:
            count += 1

        self.assertEqual(count, 2)

        # UniverseAlias with no records
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
        )
        x = Set(m, "set1")
        a = UniverseAlias(m, "universe1")
        count = 0
        for _ in a:
            count += 1

        self.assertEqual(count, 0)

        # UniverseAlias with records
        b = Set(m, "set2", records=[str(idx) for idx in range(1, 3)])
        c = UniverseAlias(m, "universe2")

        count = 0
        for _ in c:
            count += 1

        self.assertEqual(count, 2)

    def test_sameas(self):
        i = Set(self.m, "i")
        j = Alias(self.m, "j", i)
        self.assertEqual(i.sameAs(j).gamsRepr(), "( sameAs(i,j) )")
        self.assertEqual(j.sameAs(i).gamsRepr(), "( sameAs(j,i) )")

        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
            delayed_execution=int(os.getenv("DELAYED_EXECUTION", False)),
        )
        i = Set(m, "i", records=["1", "2", "3"])
        p = Parameter(m, "p", [i])
        p[i] = i.sameAs("2")

        if m.delayed_execution:
            self.assertEqual(
                m._unsaved_statements[-1].getStatement(),
                "p(i) = (sameAs( i,'2' ));",
            )

    def test_assignment_dimensionality(self):
        j1 = Set(self.m, "j1")
        j2 = Set(self.m, "j2")
        j3 = Set(self.m, "j3", domain=[j1, j2])
        with self.assertRaises(ValidationError):
            j3["bla"] = 5

        j4 = Set(self.m, "j4")

        with self.assertRaises(ValidationError):
            j3[j1, j2, j4] = 5

        j5 = Set(self.m, "j5", domain=[j1, j2])
        j6 = Set(self.m, "j6", domain=[j1, j2])

        with self.assertRaises(ValidationError):
            j6[j1, j2] = j5[j1, j2, j3]

    def test_domain_verification(self):
        m = Container(
            system_directory=os.getenv("SYSTEM_DIRECTORY", None),
        )
        i1 = Set(m, "i1", records=["i1", "i2"])
        i2 = Set(m, "i2", records=["i1"], domain=i1)

        with self.assertRaises(ValidationError):
            i2["i3"] = True

    def test_uels_on_axes(self):
        s = pd.Series(index=["a", "b"])
        i = Set(self.m, "i", records=s, uels_on_axes=True)
        self.assertEqual(i.records["uni"].tolist(), ["a", "b"])


def set_suite():
    suite = unittest.TestSuite()
    tests = [
        SetSuite(name) for name in dir(SetSuite) if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(set_suite())
