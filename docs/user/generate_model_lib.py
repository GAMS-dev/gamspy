import os
import urllib.error
import urllib.request

import pandas as pd


def open_url(request):
    try:
        return urllib.request.urlopen(request)
    except urllib.error.HTTPError as e:
        return e


files = os.listdir("tests/integration/models")

csv = {"Model": [], "GAMSPy": [], "GAMS": []}

for f in files:
    if "py" in f:
        name = f.split(".")[0]

        file_str = (
            f":orphan:\n\n"
            f".. _{name}:\n\n{name}\n{'=' * len(name)}\n\n.."
            f" literalinclude:: ../../../tests/integration/models/{name}.py\n"
        )

        links = [
            f"https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_{name}.html",
            f"https://www.gams.com/latest/finlib_ml/libhtml/finlib_{name}.html",
            f"https://www.gams.com/latest/noalib_ml/libhtml/noalib_{name}.html",
        ]

        for link in links:
            if open_url(link).status == 200:
                # Table
                csv["Model"].append(name)
                csv["GAMSPy"].append(f":ref:`GAMSPy <{name}>`")
                csv["GAMS"].append(f"`GAMS <{link}>`__")
                # rst
                with open(f"docs/examples/model_lib/{name}.rst", "w") as rst:
                    rst.write(file_str)
                break

pd.DataFrame(csv).to_csv("docs/examples/model_lib/table.csv", index=False)
