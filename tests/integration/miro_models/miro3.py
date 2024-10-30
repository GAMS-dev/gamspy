import numpy as np

from gamspy import (
    Container,
    Parameter,
    Set,
)


def main():
    m = Container()

    # Set
    col = Set(m, "col", records=[("col" + str(i), i) for i in range(1, 10)])
    row = Set(m, "row", records=[("row" + str(i), i) for i in range(1, 10)])
    initial_state_data = np.array(
        [
            [0, 0, 0, 0, 8, 6, 0, 0, 0],
            [0, 7, 0, 9, 0, 2, 0, 0, 0],
            [6, 9, 0, 0, 0, 0, 2, 0, 8],
            [8, 0, 0, 0, 9, 0, 7, 0, 2],
            [4, 0, 0, 0, 0, 0, 0, 0, 3],
            [2, 0, 9, 0, 1, 0, 0, 0, 4],
            [5, 0, 3, 0, 0, 0, 0, 7, 6],
            [0, 0, 0, 5, 0, 8, 0, 2, 0],
            [0, 0, 0, 3, 7, 0, 0, 0, 0],
        ],
    )

    initial_state = Parameter(
        m,
        "initial_state",
        domain=[row, col],
        is_miro_input=True,
        is_miro_table=True,
        records=initial_state_data,
    )
    assert initial_state.records.columns.tolist() == ["row", "col", "value"]


if __name__ == "__main__":
    main()
