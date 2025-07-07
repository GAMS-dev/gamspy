import gamspy as gp

c = gp.Container(working_directory=".")
acad_yr = gp.Set(c, "acad_yr", records=["2026"])
i = gp.Set(c, "i", records=["i1"])
completed_df = gp.Set(
    c,
    "completed_df",
    domain=[acad_yr, i],
    records=[("2026", "i1")] if not c.in_miro else None,
    is_miro_input=True,
)
completed = gp.Set(c, "completed", domain=[i])

completed[...] = gp.Sum(acad_yr, completed_df[...])
print(completed.toList())
assert completed.toList() == ["i1"], completed.toList()
