from gamspy import (
    Container,
    Parameter,
)


def main():
    m = Container()

    f = Parameter(
        m,
        name="f",
        description="supply of commodity at plant i (in cases)",
        records=5 if not m.in_miro else None,
        is_miro_input=True,
    )

    print(f.records)

    if m.in_miro:
        assert f.toValue() == 120
    else:
        assert f.toValue() == 5


if __name__ == "__main__":
    main()
