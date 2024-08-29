from __future__ import annotations

import os
import unittest

import pandas as pd
from gamspy import (
    Alias,
    Container,
    Equation,
    Ord,
    Parameter,
    Set,
    UniverseAlias,
)
from gamspy.exceptions import ValidationError


class AliasSuite(unittest.TestCase):
    def setUp(self):
        self.m = Container()

    def test_alias_creation(self):
        i = Set(self.m, "i")
        self.m.addAlias(alias_with=i)

        a = Alias(self.m, alias_with=i)
        self.assertEqual(len(a), 0)

        # no alias
        self.assertRaises(TypeError, Alias, self.m)

        # non-str type name
        self.assertRaises(TypeError, Alias, self.m, 5, i)

        # no container
        self.assertRaises(TypeError, Alias)

        # non-container type container
        self.assertRaises(TypeError, Alias, 5, "j", i)

        # try to create a symbol with same name but different type
        self.assertRaises(TypeError, Alias, self.m, "i", i)

        # get already created symbol
        j1 = Alias(self.m, "j", i)
        j2 = Alias(self.m, "j", i)
        self.assertEqual(id(j1), id(j2))

        # len of Alias
        i2 = Set(self.m, records=["i1", "i2"])
        k2 = Alias(self.m, alias_with=i2)
        self.assertEqual(len(k2), 2)

        # synch
        with self.assertRaises(ValidationError):
            k2.synchronize = True

        with self.assertRaises(ValidationError):
            _ = k2.synchronize

    def test_alias_attributes(self):
        i = Set(self.m, "i")
        j = Alias(self.m, "j", alias_with=i)
        self.assertEqual(j.pos.gamsRepr(), "j.pos")
        self.assertEqual(j.ord.gamsRepr(), "j.ord")
        self.assertEqual(j.off.gamsRepr(), "j.off")
        self.assertEqual(j.rev.gamsRepr(), "j.rev")
        self.assertEqual(j.uel.gamsRepr(), "j.uel")
        self.assertEqual(j.len.gamsRepr(), "j.len")
        self.assertEqual(j.tlen.gamsRepr(), "j.tlen")
        self.assertEqual(j.val.gamsRepr(), "j.val")
        self.assertEqual(j.tval.gamsRepr(), "j.tval")
        self.assertEqual(j.first.gamsRepr(), "j.first")
        self.assertEqual(j.last.gamsRepr(), "j.last")

    def test_alias_string(self):
        # Set and Alias without domain
        i = Set(self.m, name="i", records=["a", "b", "c"])
        j = Alias(self.m, name="j", alias_with=i)
        self.assertEqual(j.gamsRepr(), "j")
        self.assertEqual(j.getDeclaration(), "Alias(i,j);")

        # Set and Alias with domain
        k = Set(self.m, name="k", domain=[i], records=["a", "b"])
        m = Alias(self.m, name="m", alias_with=k)
        self.assertEqual(m.gamsRepr(), "m")
        self.assertEqual(m.getDeclaration(), "Alias(k,m);")

        # Check if the name is reserved
        self.assertRaises(ValidationError, Alias, self.m, "set", i)

    def test_override(self):
        # Try to add the same Alias with non-Set alias_with
        u = Set(self.m, "u")
        v = Alias(self.m, "v", alias_with=u)
        eq = Equation(self.m, "eq", domain=[u, v])
        self.assertRaises(TypeError, self.m.addAlias, "v", eq)

        # Try to add the same alias
        self.assertRaises(TypeError, self.m.addAlias, "u", u)

    def test_universe_alias_creation(self):
        # non-str type name
        self.assertRaises(TypeError, UniverseAlias, self.m, 5)

        # no container
        self.assertRaises(TypeError, UniverseAlias)

        # non-container type container
        self.assertRaises(TypeError, UniverseAlias, 5, "j")

        # try to create a symbol with same name but different type
        _ = Set(self.m, "i")
        self.assertRaises(TypeError, UniverseAlias, self.m, "i")

        # get already created symbol
        j1 = UniverseAlias(self.m, "j")
        j2 = UniverseAlias(self.m, "j")
        self.assertEqual(id(j1), id(j2))

        u = UniverseAlias(self.m, name="u")
        p = Parameter(self.m, name="p", domain=u)
        p[u] = 2

    def test_universe_alias(self):
        gdx_path = os.path.join("tmp", "test.gdx")

        h = UniverseAlias(self.m, "h")
        self.assertEqual(len(h), 0)
        _ = Set(self.m, "i", records=["i1", "i2"])
        _ = Set(self.m, "j", records=["j1", "j2"])

        self.assertEqual(
            h.records.values.tolist(), [["i1"], ["i2"], ["j1"], ["j2"]]
        )

        self.assertEqual(len(h), 4)

        self.m.write(gdx_path)

        bla = Container()
        bla.read(gdx_path)
        self.assertEqual(
            bla.data["h"].records.values.tolist(), h.records.values.tolist()
        )

        m = Container()

        r = UniverseAlias(m, name="new_universe")
        k = Set(m, name="k", domain=r, records="Chicago")
        self.assertEqual(k.getDeclaration(), "Set k(*);")

    def test_alias_state(self):
        i = Set(self.m, name="i", records=["a", "b", "c"])
        j = Alias(self.m, name="j", alias_with=i)
        i.modified = False
        j.setRecords(["a", "b"])
        self.assertFalse(i.modified)

        i.modified = False
        j.records = pd.DataFrame([["a", "b"]])
        self.assertTrue(i.modified)

    def test_alias_modified_list(self):
        nodes = self.m.addSet("nodes", description="nodes", records=["s"])
        i = self.m.addAlias("i", nodes)
        _ = self.m.addSet(
            "s", domain=[i], description="sources", records=["s"]
        )
        modified_names = self.m._get_touched_symbol_names()
        self.assertEqual(modified_names, [])

    def test_indexing(self):
        row = Set(
            self.m, "row", records=[("r-" + str(i), i) for i in range(1, 11)]
        )
        col = Set(
            self.m, "col", records=[("c-" + str(i), i) for i in range(1, 11)]
        )

        r = Parameter(
            self.m,
            "r",
            domain=row,
            records=[
                [record, 4]
                if record in row.records["uni"][:7].values
                else [record, 5]
                for record in row.records["uni"]
            ],
        )
        c = Parameter(
            self.m,
            "c",
            domain=col,
            records=[
                [record, 3]
                if record in col.records["uni"][:5].values
                else [record, 2]
                for record in col.records["uni"]
            ],
        )

        a = Parameter(self.m, "a", domain=[row, col])

        dyn_col = Set(self.m, name="dyn_col", domain=[col])
        dyn_col_alias = Alias(self.m, name="dyn_col_alias", alias_with=dyn_col)
        dyn_col[col] = Ord(col) < 5

        a[row, dyn_col_alias[col]] = 13.2 + r[row] * c[dyn_col_alias]
        self.assertEqual(
            a.getAssignment(),
            "a(row,dyn_col_alias(col)) = (13.2 + (r(row) * c(dyn_col_alias)));",
        )

        dyn_col_alias["c-1"] = False
        self.assertEqual(
            dyn_col_alias.toList(), [f"c-{idx}" for idx in range(2, 5)]
        )


def alias_suite():
    suite = unittest.TestSuite()
    tests = [
        AliasSuite(name)
        for name in dir(AliasSuite)
        if name.startswith("test_")
    ]
    suite.addTests(tests)

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(alias_suite())
