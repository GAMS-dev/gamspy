# GAMSPy Documentation

This folder contains the documentation of GAMSPy.

## Install Dependencies

```sh
pip install -r requirements.txt ----extra-index-url https://test.pypi.org/simple/
```

## Build

```sh
make clean
make html
```

## Generate Docs

```sh
sphinx-apidoc src/gamspy/ --separate -o bla --private
```