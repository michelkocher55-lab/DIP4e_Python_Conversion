# DIP4e_Python_Conversion

# Architecture

## The `processing` directory
It contains the main code for chapters 2 to 13 plus a `RunAllChpaters.py` as a script to run all the chapters in one go. 

Each chapter contains a set of files to generate the corresponding figure. A `RunAll.py` file is also included in each chapter to run all the files in one go.

All Python file can be run optionally with the `--noshow` flag to avoid showing the generated figure. This is useful when running all the chapters in one go.  The figure still will be generated and saved in the `output` directory.

## The `output` directory
It contains the generated figures for each chapter. 

## The `ia870` directory
It contains a copy of the ia870 library, to be copied in the virtual environment directory after its creation in order to use the adequate functions. 

## The `libDIP` directory
It contains the main class `DIP`. 

This file must be reworked to be more modular (next step...).

# Installation

## Create a virtual environment

From the project root:

```bash
python3 -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate
python3 -c "import site; print(site.getsitepackages()[0])"
echo "$PWD" > "$(python3 -c 'import site; print(site.getsitepackages()[0])')/dip4e_root.pth"
```


Install dependencies:

```bash
pip3 install -r requirements.txt
```

Copy the ia870 library to the virtual environment:

```bash
cp -r ia870 "$(python3 -c 'import site; print(site.getsitepackages()[0])')/ia870"
```