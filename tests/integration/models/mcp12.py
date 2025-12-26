import gamspy as gp


def check_mod(mod: gp.Container):
    assert mod.solve_status == gp.SolveStatus.NormalCompletion, (
        f"bad solvestat for {mod.name}"
    )


def main():
    m = gp.Container()
    u = gp.Set(m, records=["a", "b", "c"])
    j1 = gp.Set(m, domain=u)
    j2 = gp.Set(m, domain=u)
    j3 = gp.Set(m, domain=u)

    XX = gp.Parameter(m, records=gp.SpecialValues.NA)
    x_ = gp.Parameter(m, records=0.5)
    v1_ = gp.Parameter(m, domain=u, description="value of v1 at solution")
    v2_ = gp.Parameter(m, domain=u, description="value of v2 at solution")
    v3_ = gp.Parameter(m, domain=u, description="value of v3 at solution")

    v1_[u] = 1 + (gp.Ord(u) - 1) / 4
    v2_[u] = 2 + (gp.Ord(u) - 1) / 4
    v3_[u] = 3 + (gp.Ord(u) - 1) / 4

    v1 = gp.Variable(m, domain=u)
    v2 = gp.Variable(m, domain=u)
    v3 = gp.Variable(m, domain=u)
    x = gp.Variable(m)
    e = gp.Equation(m, domain=u)
    f = gp.Equation(m, domain=u)

    e[u] = (
        v1[u].where[j1[u]]
        + v2[u].where[j2[u]]
        + v3[u].where[j3[u]]
        + x.where[XX]
        + v1_[u].where[~j1[u]]
        + v2_[u].where[~j2[u]]
        + v3_[u].where[~j3[u]]
        + x_.where[~XX]
        == v1_[u] + v2_[u] + v3_[u] + x_
    )
    f[u] = v1[u] + v2[u] + v3[u] == v1_[u] + v2_[u] + v3_[u]

    me = gp.Model(m, problem=gp.Problem.MCP, matches={e: [v1, v2, v3]})
    mf = gp.Model(m, problem=gp.Problem.MCP, matches={f: [v1, v2, v3]})

    """
------------------------------------------------------------------------
case: e.(v1,v2,v3) with an empty v-list for some tuple
rows:  e_a,  e_b,  e_c
cols: v1_a, v2_b       x
result: good match, assuming x is free
------------------------------------------------------------------------
    """
    j1[u] = u.sameAs("a")
    j2[u] = u.sameAs("b")
    j3[u] = False
    XX[...] = 1
    me.solve()
    check_mod(me)

    v1.fx["a"] = 1
    me.solve()
    check_mod(me)

    v1.lo["a"] = -float("inf")
    me.solve()
    check_mod(me)

    v1.up["a"] = 0.25
    me.solve()
    check_mod(me)

    v1.up["a"] = float("inf")

    """
------------------------------------------------------------------------
case: e.(v1,v2,v3) with a singleton v-list for every tuple
rows:  e_a,  e_b,  e_c
cols: v1_a, v2_b   v3_c
result: good match, independent of bounds on any variables
------------------------------------------------------------------------
    """
    j1[u] = u.sameAs("a")
    j2[u] = u.sameAs("b")
    j3[u] = u.sameAs("c")
    XX[...] = 0

    me.solve()
    check_mod(me)

    v3.fx["c"] = 1
    me.solve()
    check_mod(me)

    v3.lo["c"] = gp.SpecialValues.NEGINF
    me.solve()
    check_mod(me)

    v3.up["c"] = 0.25
    me.solve()
    check_mod(me)
    v3.up["c"] = gp.SpecialValues.POSINF

    """
------------------------------------------------------------------------
case: f.(v1,v2,v3) with a full v-list for every tuple
rows:  e_a,  e_b,  e_c
cols: v1_a, v1_b, v1_c
      v2_a, v2_b, v2_c
      v3_a, v3_b, v3_c
result: good match, if we fix 2 or more variables for each tuple
------------------------------------------------------------------------
    """
    # start with everything fixed
    v1.fx[u] = v1_[u]
    v2.fx[u] = v2_[u]
    v3.fx[u] = v3_[u]

    mf.solve()
    check_mod(mf)

    v1.fx["c"] = 8
    mf.solve()
    check_mod(mf)

    v1.lo["c"] = 0
    mf.solve()
    check_mod(mf)

    v2.lo["b"] = gp.SpecialValues.NEGINF
    v2.up["b"] = gp.SpecialValues.POSINF
    mf.solve()
    check_mod(mf)

    v1.lo["a"] = -1000
    v1.up["a"] = 1000
    mf.solve()
    check_mod(mf)

    """
------------------------------------------------------------------------
case: all scalar = fs.(vs1,vs2,vs3)
rows:  fs
cols: va1
      va2
      va3
result: good match, if we fix 2 or more variables
------------------------------------------------------------------------
    """

    vs1 = gp.Variable(m)
    vs2 = gp.Variable(m)
    vs3 = gp.Variable(m)
    fs = gp.Equation(m)
    fs[...] = vs1 + vs2 + vs3 == gp.Number(1)
    mfs = gp.Model(m, problem=gp.Problem.MCP, matches={fs: [vs1, vs2, vs3]})

    # start with everything fixed
    vs1.fx[...] = 1
    vs2.fx[...] = 1
    vs3.fx[...] = 1
    mfs.solve()
    check_mod(mfs)

    vs2.lo[...] = vs2.lo[...] - 10
    mfs.solve()
    check_mod(mfs)

    vs2.lo[...] = gp.SpecialValues.NEGINF
    mfs.solve()
    check_mod(mfs)

    vs2.up[...] = gp.SpecialValues.POSINF
    mfs.solve()
    check_mod(mfs)

    """
------------------------------------------------------------------------
case: fv.(vv1,vv2,vv3) with a full v-list for every tuple - with extra dimension
rows:  fv_a,  fv_b,  fv_c
cols: vv1_a, vv1_b, vv1_c
      vv2_a, vv2_b, vv2_c
      vv3_a, vv3_b, vv3_c
result: good match, if we fix 2 or more variables for each tuple
------------------------------------------------------------------------
    """

    v = gp.Set(m, records=["x", "y", "z"])
    vv1_ = gp.Parameter(m, domain=[u, v], description="value of vv1 at solution")
    vv2_ = gp.Parameter(m, domain=[u, v], description="value of vv2 at solution")
    vv3_ = gp.Parameter(m, domain=[u, v], description="value of vv3 at solution")

    vv1_[u, v] = 1 + (gp.Ord(u) - 1) / 4 + (gp.Ord(v) - 1) / 4
    vv2_[u, v] = 2 + (gp.Ord(u) - 1) / 4 + (gp.Ord(v) - 1) / 4
    vv3_[u, v] = 3 + (gp.Ord(u) - 1) / 4 + (gp.Ord(v) - 1) / 4

    vv1 = gp.Variable(m, domain=[u, v])
    vv2 = gp.Variable(m, domain=[u, v])
    vv3 = gp.Variable(m, domain=[u, v])
    fv = gp.Equation(m, domain=[u, v])
    fv[u, v] = vv1[u, v] + vv2[u, v] + vv3[u, v] == vv1_[u, v] + vv2_[u, v] + vv3_[u, v]
    mfv = gp.Model(m, problem=gp.Problem.MCP, matches={fv: [vv1, vv2, vv3]})

    # start with everything fixed
    vv1.fx[u, v] = vv1_[u, v]
    vv2.fx[u, v] = vv2_[u, v]
    vv3.fx[u, v] = vv3_[u, v]
    mfv.solve()
    check_mod(mfv)

    vv1.fx["c", v] = 8
    mfv.solve()
    check_mod(mfv)

    vv1.lo["c", v] = 0
    mfv.solve()
    check_mod(mfv)

    vv2.lo["b", v] = gp.SpecialValues.NEGINF
    vv2.up["b", v] = vv2.lo["b", v] * -1
    mfv.solve()
    check_mod(mfv)

    vv1.lo["a", v] = -1000
    vv1.up["a", v] = 1000
    mfv.solve()
    check_mod(mfv)


if __name__ == "__main__":
    main()
