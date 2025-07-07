# Domain violation test

import gamspy as gp

c = gp.Container()
i = gp.Set(c, "i")
p = gp.Parameter(
    c,
    "p",
    domain=i,
    records=[("i1", 1)],
    domain_forwarding=True,
    is_miro_input=True,
)
i.setRecords(["i1", "i2"])
p2 = gp.Parameter(c, "p2", domain=i, records=[("i2", 1)], is_miro_input=True)
