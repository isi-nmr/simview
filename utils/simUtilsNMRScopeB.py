import json
import re
from collections.abc import Mapping

import numpy as np
from PyQt6.QtWidgets import QMainWindow, QProgressDialog


def _extract_numeric_series(payload: object) -> np.ndarray:
    if isinstance(payload, Mapping):
        for key in ("val", "values", "data"):
            if key in payload:
                return np.asarray(payload[key], dtype=float)
        return np.asarray([], dtype=float)
    return np.asarray(payload, dtype=float)


def _extract_units(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return ""
    for key in ("units", "unit"):
        if key in payload:
            return str(payload[key])
    return ""


def _extract_show(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return True
    show_value = payload.get("show", True)
    if isinstance(show_value, str):
        return show_value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(show_value)


def readNMRScopeBChannels(path_data:str|dict,progress:QProgressDialog,app:QMainWindow)->dict:

    if type(path_data) is str:
        with open(path_data+"/pulse_seq.json") as f:
            data = json.load(f)

        app.setWindowTitle(f"{path_data}")
    else:
        data = path_data

    channels = []

    if not isinstance(data, Mapping):
        raise TypeError("NMRScopeB payload must be a dictionary-like object")

    # Move gradients to the end...
    end_keys = ["gx", "gy", "gz"]

    # Rebuild the dictionary
    new_data = {k: v for k, v in data.items() if k not in end_keys}
    for k in end_keys:
        if k in data:
            new_data[k] = data[k]




    time = _extract_numeric_series(data.get("time", [])) * 1e-3

    for channelName in new_data:
        if channelName == "time":
            continue

        plotType = "phase" if "phase" in channelName or "_p" in channelName else "mag"

        payload = new_data[channelName]
        dataNpy = _extract_numeric_series(payload)

        if dataNpy.size != time.size:
            print (f"Array length does not match the time vector for channel {channelName}. Found {dataNpy.size} \
                samples, time vector has {time.size} samples skipping...")
            continue

        units = _extract_units(payload)

        unitLabel = f"({units})" if units != '' else "(-)"

        channelDes = {
            "chanLabel": channelName +" "+unitLabel,
            "label": channelName,
            "type": "grads" if re.match(r"g\w",channelName) else "NCO",
            "ind": str(0),
            "key": channelName,
            "plotType": plotType,
            "units": units,
            "raw_units": units,
            "t": time,
            "data":dataNpy,
            "raw_data": dataNpy.copy(),
            "show": _extract_show(payload),
        }
        channelDes["annotations"] = []
        channels.append([channelDes])

    return channels
