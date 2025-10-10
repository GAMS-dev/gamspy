import gamspy as gp

c = gp.Container()

par = gp.Parameter(c, "test", domain=["*"], is_miro_input=True, is_miro_table=True)
