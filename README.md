# DIP4e_Python_Conversion

# Architecture

## The `processing` directory
It contains the main Python's runners for chapters 2 to 13 plus a `RunAllChapters.py` as a script to run all the chapters in one go. 

Each chapter contains a set of files to generate the corresponding figure. A `RunAll.py` file is also included in each chapter to run all the files in one go.

All Python files can be run optionally with the `--noshow` flag to avoid showing the generated figure. This is useful when running all the chapters in one go.  The figure still will be generated and saved in the `output` directory.

## The `libDIP` directory
It contains the main class `DIP`. 

This file must be reworked to be more modular (next steps...).

## The `helpers` directory
It  contains the helper functions for the processing of the data.

## The `AllDataFiles` directory
It contains the input data for each chapter. 

## The `output` directory
It contains the generated figures for each chapter. 

## The `ia870` directory
It contains a copy of the ia870 library, to be copied in the virtual environment directory after its creation in order to use the adequate functions. 

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

## Run the code

Activate the virtual environment if not already done:

```bash# macOS / Linux
source .venv/bin/activate
```

To run all the chapters in one go:

```bash
cd processing
python3 RunAllChapters.py --noshow
```

To run a specific chapter, for example chapter 2:

```bash
cd processing/Chapter02
python3 RunAll.py --noshow
``` 

To run a specific file, for example `Fig2_01.py`:

```bash
cd processing/Chapter02
python3 Figure223.py
```

# Adding a new Figure

1. Change directory to the processing/Chapter02 directory.
2. Copy an existing file, for example `Figure223.py`, and rename it to the new figure name, for example `Figure224.py`.
3. Edit the new file to generate the desired figure.
4. Change directory to the LibDIP directory and edit the `DIP.py` file to add the new function for the new figure (copy an existing function and rename it).