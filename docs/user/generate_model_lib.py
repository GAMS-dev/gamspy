from __future__ import annotations

from pathlib import Path
from pathlib import PurePosixPath

import pandas as pd
import subprocess


models = list(Path("tests/integration/models").glob("*.py"))

csv = {
    "Model": [],
    "Data": [],
    "Model Type": [],
    "License": [],
}


for model in models:
    content = model.read_text().split("\n")[1:10]
    model_info = model_info = [
        line
        for line in content
        if line.startswith("##") and not line.startswith("## KEYWORDS:")
    ]

    # {'GAMSSOURCE': 'https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_aircraft.html', 'LICENSETYPE': 'Demo', 'MODELTYPE': 'LP'}
    model_info_dict = {
        line.split(": ")[0][2:].strip(): line.split(": ")[1].strip()
        for line in model_info
    }

    ## FETCHING DATA FROM MODEL FILES

    # Model's name
    name = model.stem

    # Model's title
    title = next(
        (line for line in content if not line.startswith("##") and line != ""),
        None,
    )

    # Model's GAMSPy source
    gamspy_source = f":ref:`{name} <{name}>`"

    # Model's GAMS source
    gams_source = model_info_dict.get("GAMSSOURCE", "")
    gams_source = f"`GAMSSOURCE <{gams_source}>`__" if gams_source else ""

    # Model's license type
    license_type = model_info_dict.get("LICENSETYPE")

    # Model's type
    model_type = model_info_dict.get("MODELTYPE")

    # Model's data files (if any)
    data_files = model_info_dict.get("DATAFILES", "")
    data_files_link = (
        f":download:`{data_files} <{PurePosixPath('../../tests/integration/models/', data_files)}>`"
        if data_files
        else ""
    )

    ## APPENDING DATA TO .csv FILE
    csv["Model"].append(f"{gamspy_source}   {gams_source}")
    csv["Data"].append(data_files_link)
    csv["License"].append(license_type)
    csv["Model Type"].append(model_type)

    ## CONFIGURE .rst FILES
    rst_head = (
        ":orphan:\n\n"
        f".. _{name}:\n\n{title}\n{'=' * len(title)}\n\n"
        f":download:`{model.name} <{PurePosixPath('../../..', model)}>` "
    )

    rst_foot = (
        "\n\n.. literalinclude::"
        f" {PurePosixPath(f'../../../tests/integration/models/{name}.py')}\n"
    )

    if data_files:
        rst_data = (
            f"|{data_files}|\n\n"
            f".. |{data_files}| replace::\n"
            f"   :download:`{data_files} <{PurePosixPath('../../../tests/integration/models/', data_files)}>`\n\n"
        )
        rst_str = rst_head + rst_data + rst_foot
    else:
        data_link = ""
        rst_str = rst_head + rst_foot

    with open(f"docs/examples/model_lib/{name}.rst", "w") as rst_file:
        rst_file.write(rst_str)


pd.DataFrame(csv).sort_values(
    by="Model", key=lambda col: col.str.lower()
).to_csv(Path("docs/examples/model_lib/table.csv"), index=False)


# Go to docs directory for the remaining part
directory = Path("docs").resolve()

command = ["make", "html"]

process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=Path("docs").resolve(),
)
stdout, stderr = process.communicate()

if process.returncode != 0:
    print(f"An error occurred: {stderr.decode('utf-8')}")
else:
    print(stdout.decode("utf-8"))

content = (
    (directory / "_build" / "html" / "user" / "model_library.html")
    .read_text()
    .splitlines()
)

content_n = []

for line in content:
    if "GAMSSOURCE" in line:
        line_n = line.replace(
            "GAMSSOURCE",
            "<img src='../_static/gams.svg' class='icon-link-table' alt='GAMS'/>",
        )
        content_n.append(line_n)

    else:
        content_n.append(line)

(directory / "_build" / "html" / "user" / "model_library.html").write_text(
    "\n".join(content_n)
)
