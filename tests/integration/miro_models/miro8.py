import gamspy as gp

m = gp.Container()
i = gp.Set(m, name="i", records=range(10))
j = gp.Alias(m, name="j", alias_with=i)

letters = gp.Set(
    m,
    name="letters",
    records=list("abcdefghijklmnopqrstuvwxyz"),
)

pick_letter = gp.Set(m, name="pick_letter", domain=[letters], is_miro_input=True)
