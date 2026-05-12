# Sim View

Sim View is a lightweight desktop viewer for pulse-sequence simulation outputs from Bruker ParaVision-style simulations and NMRScopeB exports.

It is intended for quick inspection of RF, gradient, and derived channels during pulse-sequence development.

## Features
- Load Bruker simulation folders and NMRScopeB output folders
- Inspect RF and gradient channels in a synchronized time view
- Show derived gradient signals such as slew rate, trajectory, and duty cycle
- Export measurements to Excel
- Use either the GUI file picker or launch directly from the command line

## Requirements
- Python 3.12 or newer
- A working Qt desktop environment for `PyQt6`
- `poetry` for the recommended installation flow

## Installation with Poetry
1. Install Python 3.12+ and confirm it is available with `python --version`
2. Install Poetry from <https://python-poetry.org/docs/#installing-with-the-official-installer>
3. Install dependencies:

```bash
poetry install
```

4. Start the app:

```bash
poetry run python simView.py
```

## Manual Installation
1. Install Python 3.12+
2. Create and activate a virtual environment
3. Install the dependencies listed in `pyproject.toml`
4. Start the app from that environment:

```bash
python simView.py
```

## Usage
- Open the application and choose `File -> Open Folder`
- Select a simulation output folder containing either Bruker output or NMRScopeB output
- You can also launch the viewer directly for a specific folder:

```bash
python simView.py /path/to/simulation_folder
```

### Loading JSON Data Programmatically
```python
from simView import show_graphs_from_dict

import json

path = "/home/vitous/Documents/seqSim/scopeBNew/pulse_seq.json"
with open(path) as f:
    data = json.load(f)

show_graphs_from_dict(data)
```

## ParaVision Integration
For ParaVision integration, use the wrapper script included in this repository.

1. Run `./install_venv.sh` to create and populate a project-local `.venv`
2. Set `/path/to/script/run_simview.sh` as the viewer command in the ParaVision options menu
3. Wrapper stderr is logged to `simview_viewer_errors.log` inside the target simulation folder
4. Optional override:

```bash
SIMVIEW_ERROR_LOG=/path/to/file /path/to/script/run_simview.sh
```

## AlmaLinux / Qt Notes
- On AlmaLinux, especially over remote connections, Qt may fail to start with an `xcb` or `libxcb-cursor.so.0` error
- Install the missing system package:

```bash
sudo dnf install -y xcb-util-cursor
```

- If needed, also install:

```bash
sudo dnf install -y libxkbcommon-x11 mesa-libGL
```

## Testing
Run the full test suite with:

```bash
poetry run pytest
```

If you are using the local virtual environment directly:

```bash
.venv/bin/python -m pytest
```

Current tests cover:
- Core derived-signal logic
- Bruker parser loading against fixture data in `testData/mrScanSim`

## Acknowledgment
If Sim View contributes to your pulse-sequence development or related research, please consider acknowledging the infrastructure support that helped make this work possible:

`Czech-BioImaging LM2023050, funded by the Ministry of Education, Youth and Sports of the Czech Republic.`

We would also be glad to hear about publications, presentations, or internal use cases where the tool proved helpful.

## Contact
Questions, feedback, and collaboration notes are welcome at `vitous@isibrno.cz`.
