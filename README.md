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
- For ParaVision integration, use the wrapper script:
  - `./install_venv.sh` to create and populate a project-local `.venv`
  - `/path/to/script/run_simview.sh` as the viewer command in ParaVision options menu (card MRI)
  - wrapper stderr is logged to `simview_viewer_errors.log` in the target simulation folder
  - optional override: `SIMVIEW_ERROR_LOG=/path/to/file /path/to/script/run_simview.sh`

- Or by using the function
```python
from simView import show_graphs_from_dict

import json

path = "/home/vitous/Documents/seqSim/scopeBNew/pulse_seq.json"
with open(path) as f:
    data = json.load(f)
    
show_graphs_from_dict(data)    

```

## AlmaLinux / Qt Notes
- On AlmaLinux (remote connection), Qt may fail to start with an error about `xcb` / `libxcb-cursor.so.0`.
- Install the missing system package:

```bash
sudo dnf install -y xcb-util-cursor
```

- If needed, install additional runtime libs:

```bash
sudo dnf install -y libxkbcommon-x11 mesa-libGL
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
