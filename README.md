# DIP4e_Python_Conversion

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
