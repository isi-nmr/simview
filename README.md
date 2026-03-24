# Sim View

Simple tool to visualize outputs from Paravision pulse sequence simulation 

## Installation with poetry
- Install python (mainly concerns windows)
- Make sure at least Python3.12 exists on the system and can be invoked by python --version
- Install poetry https://python-poetry.org/docs/#installing-with-the-official-installer
- run poetry install
- run the simView app in the created venv

## Installation manual
- Install python (mainly concerns windows)
- Make sure at least Python3.12 exists on the system and can be invoked by python --version
- Create venv
- install packages from pyproject.toml using pip
- run the simView app in the created venv

## Usage
- Upon opening the window use file -> Open Folder and open folder with the simulation output (contianing either bruker output or NMRScopeB output)
- Or run from command line as python3 simView {path}

- Or by using the function
```python
from simView import show_graphs_from_dict

import json

path = "/home/vitous/Documents/seqSim/scopeBNew/pulse_seq.json"
with open(path) as f:
    data = json.load(f)
    
show_graphs_from_dict(data)    

```

## Testing
- Run the full test suite with `pytest`

```bash
poetry run pytest
```

- Or, if you are already using the local virtualenv directly:

```bash
.venv/bin/python -m pytest
```

- The current tests cover:
  - core derived-signal logic in `simView.py`
  - Bruker parser loading against fixture data in `testData/mrScanSim`
