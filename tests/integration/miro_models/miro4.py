from gamspy import (
    Container,
    Parameter,
    Set,
)


def main():
    m = Container()
    i = Set(m, name="i")
    p = Parameter(
        m,
        name="p",
        domain=i,
        records=[("i2", 2)],
        domain_forwarding=True,
        is_miro_input=True,
    )

    assert i.records.values.tolist() == [["i1", ""]]
    assert p.records.values.tolist() == [["i1", 1.0]]

    _ = Set(m, name="j", records=["j1"])
    _ = Parameter(m, name="p2", records=5)


if __name__ == "__main__":
    main()
