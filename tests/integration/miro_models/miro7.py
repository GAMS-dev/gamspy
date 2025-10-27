import gamspy as gp

m = gp.Container()
i = gp.Set(m, name="i", records=range(10))
j = gp.Alias(m, name="j", alias_with=i)

letters = gp.Set(
    m,
    name="letters",
    records=list("abcdefghijklmnopqrstuvwxyz"),
    is_miro_input=True,
)
