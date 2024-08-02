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
        records=5 if not m.miro_in else None,
        is_miro_input=True,
    )

    if m.miro_in:
        assert f.toValue() == 0
    else:
        assert f.toValue() == 5


if __name__ == "__main__":
    main()
