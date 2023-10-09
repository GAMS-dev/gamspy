import urllib.error
import urllib.request
import pandas as pd
from pathlib import Path, PurePosixPath


def open_url(request):
    try:
        return urllib.request.urlopen(request)
    except urllib.error.HTTPError as e:
        return e


files = list(Path('tests/integration/models').glob('*.py'))

csv = {"Model": [], "GAMSPy": [], "GAMS": [], "Data": []}

for f in files:
    name = f.stem

    # Find data files
    data_file = ""
    data = False
    matching_files = list(Path(f"tests/integration/models").glob(f'{name}.*'))
    if len(matching_files) > 1:
        data = True
        for m in matching_files:
            if m.suffix != ".py":
                data_file = m
                data_name = m.name

        # configure .rst file
        rst_head = (
            f":orphan:\n\n"
            f".. _{name}:\n\n{name}\n===========================================\n\n"
        )

        rst_foot = f".. literalinclude:: {PurePosixPath(f'../../../tests/integration/models/{name}.py')}\n"

        if data:
                rst_data = (
                f"Data |{data_name}|\n\n"
                f".. |{data_name}| replace::\n"
                f"   :download:`{data_name} <{PurePosixPath('../../..', data_file)}>`\n\n"
            )

        if data:
            rst_str = rst_head + rst_data + rst_foot
        else:
            rst_str = rst_head + rst_foot

        # model libraries to check
        links = [
            f"https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_{name}.html",
            f"https://www.gams.com/latest/finlib_ml/libhtml/finlib_{name}.html",
            f"https://www.gams.com/latest/noalib_ml/libhtml/noalib_{name}.html",
            f"https://www.gams.com/latest/psoptlib_ml/libhtml/psoptlib_{name}.html"
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
                csv["Data"].append(data_file)
                # write rst
                Path(f"docs/examples/model_lib/{name}.rst").write_text(rst_str)
                break
        if not found:
            print(f"{name} not found in lib")

# write model lib table
pd.DataFrame(csv).sort_values(by="Model", key=lambda col: col.str.lower()).to_csv(
    Path("docs/examples/model_lib/table.csv"), index=False
)
