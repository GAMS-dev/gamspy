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

    # Find data files
        for m in matching_files:
            if not 'py' in m:
                data = m

        # configure .rst file
            f":orphan:\n\n"
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
                # write rst
                break
        if not found:
            print(f'{name} not found in lib')

# write model lib table
