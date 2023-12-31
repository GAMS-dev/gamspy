import urllib.error
import urllib.request
from pathlib import Path
from pathlib import PurePosixPath

import pandas as pd


def open_url(request):
    try:
        return urllib.request.urlopen(request)
    except urllib.error.HTTPError as e:
        return e


files = list(Path("tests/integration/models").glob("*.py"))

csv = {"Model": [], "GAMSPy": [], "GAMS": [], "Data": []}

for f in files:
    name = f.stem
    title = f.read_text().split("\n")[1]

    # Find data files
    data_file = ""
    data = False
    matching_files = list(Path(f"tests/integration/models").glob(f"{name}.*"))
    if len(matching_files) > 1:
        data = True
        for m in matching_files:
            if m.suffix != ".py":
                data_file = m
                data_name = m.name

    # configure .rst file
    rst_head = (
        ":orphan:\n\n"
        f".. _{name}:\n\n{title}\n{'=' * len(title)}\n\n"
        f":download:`{f.name} <{PurePosixPath('../../..', f)}>` "
    )

    rst_foot = (
        "\n\n.. literalinclude::"
        f" {PurePosixPath(f'../../../tests/integration/models/{name}.py')}\n"
    )

    if data:
        data_link = (
            f":download:`{data_name} <{PurePosixPath('../..', data_file)}>`"
        )
        rst_data = (
            f"|{data_name}|\n\n"
            f".. |{data_name}| replace::\n"
            f"   :download:`{data_name} <{PurePosixPath('../../..', data_file)}>`\n\n"
        )
        rst_str = rst_head + rst_data + rst_foot
    else:
        data_link = ""
        rst_str = rst_head + rst_foot

    # model libraries to check
    links = [
        f"https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_{name}.html",
        f"https://www.gams.com/latest/finlib_ml/libhtml/finlib_{name}.html",
        f"https://www.gams.com/latest/noalib_ml/libhtml/noalib_{name}.html",
        f"https://www.gams.com/latest/psoptlib_ml/libhtml/psoptlib_{name}.html",
    ]

    # check in which lib the model is
    for link in links:
        found = False
        if open_url(link).status == 200:
            found = True
            # Table
            csv["Model"].append(name)
            csv["GAMSPy"].append(f":ref:`GAMSPy <{name}>`")
            csv["GAMS"].append(f"`GAMS <{link}>`__")
            csv["Data"].append(data_link)
            # write rst
            with open(f"docs/examples/model_lib/{name}.rst", "w") as rst_file:
                rst_file.write(rst_str)
            break
    if not found:
        print(f"{name} not found in lib")

# write model lib table
pd.DataFrame(csv).sort_values(
    by="Model", key=lambda col: col.str.lower()
).to_csv(Path("docs/examples/model_lib/table.csv"), index=False)
