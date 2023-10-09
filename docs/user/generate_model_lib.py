import os
import urllib.error
import urllib.request
import glob
import pandas as pd


def open_url(request):
    try:
        return urllib.request.urlopen(request)
    except urllib.error.HTTPError as e:
        return e


files = os.listdir("tests/integration/models")

csv = {"Model": [], "GAMSPy": [], "GAMS": [], "Data": []}

for f in files:
    if "py" in f:
        name = f.split(".")[0]

        data = ""
        matching_files = glob.glob(f"tests/integration/models/{name}*")
        for m in matching_files:
            if not 'py' in m:
                data = m

        file_str = (
            f":orphan:\n\n"
            f".. _{name}:\n\n{name}\n===========================================\n\n.."
            f" literalinclude:: ../../../tests/integration/models/{name}.py\n"
        )

        links = [
            f"https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_{name}.html",
            f"https://www.gams.com/latest/finlib_ml/libhtml/finlib_{name}.html",
            f"https://www.gams.com/latest/noalib_ml/libhtml/noalib_{name}.html",
            f"https://www.gams.com/latest/psoptlib_ml/libhtml/psoptlib_{name}.html"
        ]

        for link in links:
            found = False
            if open_url(link).status == 200:
                found = True
                # Table
                csv["Model"].append(name)
                csv["GAMSPy"].append(f":ref:`GAMSPy <{name}>`")
                csv["GAMS"].append(f"`GAMS <{link}>`__")
                csv["Data"].append(data)
                # rst
                with open(f"docs/examples/model_lib/{name}.rst", "w") as rst:
                    rst.write(file_str)
                break
        if not found:
            print(f'{name} not found in lib')

pd.DataFrame(csv).sort_values(by='Model', key=lambda col: col.str.lower()).to_csv("docs/examples/model_lib/table.csv", index=False)
