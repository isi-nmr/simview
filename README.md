# Sim View

Simple tool to visualize outputs from Paravision pulse sequence simulation 

## Installation
- Install python (mainly concerns windows)
- Make sure at least Python3.12 exists on the system and can be invoked by python --version
- Install poetry https://python-poetry.org/docs/#installing-with-the-official-installer
- run poetry install
- run the simView app in the created venv

## Usage
- Upon opening the window use file -> Open Folder and open folder with the simulation output (contianing either bruker output or NMRScopeB output)
- Or run from command line as python3 simView {path}