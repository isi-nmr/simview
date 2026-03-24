import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xmltodict
from PyQt6.QtWidgets import QMainWindow, QProgressDialog


def getGradEvents(dict: dict) -> tuple[np.ndarray, np.ndarray]:
    time = np.zeros(len(dict["pulseprogram"]["ev"]))

    tUnit = float(dict["pulseprogram"]["@timeunit"])

    grads = np.zeros((3, len(dict["pulseprogram"]["ev"])))
    ind = 0
    for event in dict["pulseprogram"]["ev"]:
        if "@g1" not in event:
            continue

        time[ind] = float(event["@t"]) * tUnit
        grads[:, ind] = [float(event["@g1"]), float(event["@g2"]), float(event["@g3"])]
        ind = ind + 1

    return time[:ind], grads[:, :ind]


def readGrads(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path + "/" + "_GCube.xml") as f:
        gCube = xmltodict.parse(f.read())

    time, grads = getGradEvents(gCube)

    return time, grads


def initNco(nEvents: int) -> dict:
    return {
        "t": np.zeros(nEvents),
        "p0": np.zeros(nEvents),
        "p1": np.zeros(nEvents),
        "p2": np.zeros(nEvents),
        "am": np.zeros(nEvents),
        "pw": np.zeros(nEvents),
        "sf": np.zeros(nEvents),
        "rgp": np.zeros(nEvents),
    }


def turnOffNco(ncos: dict, ncoNumber: int, t: np.ndarray, ind: int) -> None:
    ncos[ncoNumber]["t"][ind] = t
    ncos[ncoNumber]["p0"][ind] = float(ncos[ncoNumber]["p0"][ind - 1]) if ind > 0 else 0
    ncos[ncoNumber]["p1"][ind] = float(ncos[ncoNumber]["p1"][ind - 1]) if ind > 0 else 0
    ncos[ncoNumber]["p2"][ind] = float(ncos[ncoNumber]["p2"][ind - 1]) if ind > 0 else 0
    ncos[ncoNumber]["am"][ind] = 0
    ncos[ncoNumber]["pw"][ind] = 0
    ncos[ncoNumber]["sf"][ind] = float(ncos[ncoNumber]["sf"][ind - 1]) if ind > 0 else 0
    ncos[ncoNumber]["rgp"][ind] = 0


def getRFEvents(dict: dict) -> tuple[dict, dict]:
    ncos = {}

    pulse_program = dict["pulseprogram"]
    info = {"pulProg": pulse_program["@path"].split("/")[-1]}
    ind = 0
    tUnit = float(pulse_program["@timeunit"])

    indMaxes = {}
    line_mapping: dict[int, dict[str, object]] = {}
    line_event_times: list[float] = []
    line_event_numbers: list[int] = []
    parsed_events: list[dict[str, object]] = []

    line_mapping_entries = pulse_program.get("linemapping", {}).get("map", [])
    if isinstance(line_mapping_entries, Mapping):
        line_mapping_entries = [line_mapping_entries]
    for entry in line_mapping_entries:
        if "@ln" not in entry:
            continue
        try:
            ln_value = int(entry["@ln"])
        except (TypeError, ValueError):
            continue

        source_name = (
            entry.get("@file")
            or entry.get("@cpd")
            or entry.get("@intern")
            or entry.get("@macro")
            or "unknown"
        )
        source_line_value = entry.get("@line", entry.get("@ml"))
        try:
            source_line_number = int(source_line_value) if source_line_value is not None else None
        except (TypeError, ValueError):
            source_line_number = None

        line_mapping[ln_value] = {
            "source": str(source_name),
            "line": source_line_number,
        }

    ncoNumber = None

    for event in pulse_program["ev"]:
        if "@t" not in event:
            print("Skip event")
            continue

        event_time = float(event["@t"]) * tUnit
        event_record: dict[str, object] = {"t": event_time}
        for raw_key, raw_value in event.items():
            if not raw_key.startswith("@"):
                continue
            key = raw_key[1:]
            if key == "t":
                continue
            event_record[key] = raw_value
        parsed_events.append(event_record)

        if "@ln" in event:
            try:
                line_event_times.append(event_time)
                line_event_numbers.append(int(event["@ln"]))
            except (TypeError, ValueError):
                pass

        if "@nco" in event:
            if event["@nco"] == "0":
                turnOffNco(
                    ncos,
                    ncoNumber,
                    event_time,
                    indMaxes[str(ncoNumber)],
                )
                indMaxes[ncoNumber] += 1
                continue
            ncoNumber = event["@nco"]
        else:
            ncoNumber = str(1)

        if str(ncoNumber) not in ncos:
            ncos[str(ncoNumber)] = initNco(len(dict["pulseprogram"]["ev"]))
            indMaxes[str(ncoNumber)] = 0

        ind = indMaxes[str(ncoNumber)]

        ncos[ncoNumber]["t"][ind] = event_time

        if "@p0" in event:
            ncos[ncoNumber]["p0"][ind] = float(event["@p0"])
        else:
            ncos[ncoNumber]["p0"][ind] = ncos[ncoNumber]["p0"][ind - 1] if ind > 0 else 0

        if "@p1" in event:
            ncos[ncoNumber]["p1"][ind] = float(event["@p1"])
        else:
            ncos[ncoNumber]["p1"][ind] = ncos[ncoNumber]["p1"][ind - 1] if ind > 0 else 0

        if "@p2" in event:
            ncos[ncoNumber]["p2"][ind] = float(event["@p2"])
        else:
            ncos[ncoNumber]["p2"][ind] = ncos[ncoNumber]["p2"][ind - 1] if ind > 0 else 0

        if "@pw" in event:
            ncos[ncoNumber]["pw"][ind] = float(event["@pw"])
        else:
            ncos[ncoNumber]["pw"][ind] = ncos[ncoNumber]["pw"][ind - 1] if ind > 0 else 0

        if "@am" in event:
            ncos[ncoNumber]["am"][ind] = float(event["@am"])
        else:
            ncos[ncoNumber]["am"][ind] = ncos[ncoNumber]["am"][ind - 1] if ind > 0 else 0

        if "@sf" in event:
            ncos[ncoNumber]["sf"][ind] = float(event["@sf"])
        else:
            ncos[ncoNumber]["sf"][ind] = ncos[ncoNumber]["sf"][ind - 1] if ind > 0 else 0

        if "@rgp" in event:
            ncos[ncoNumber]["rgp"][ind] = 1 if "0--0" in event["@rgp"] else 0

        elif "@ln" in event:
            if event["@ln"] == "8000001":
                ncos[ncoNumber]["rgp"][ind] = 0
            else:
                ncos[ncoNumber]["rgp"][ind] = ncos[ncoNumber]["rgp"][ind - 1] if ind > 0 else 0
        else:
            ncos[ncoNumber]["rgp"][ind] = ncos[ncoNumber]["rgp"][ind - 1] if ind > 0 else 0

        indMaxes[str(ncoNumber)] += 1

    for ncoKey, subDict in ncos.items():
        for key, value in subDict.items():
            if isinstance(value, np.ndarray):
                subDict[key] = value[: indMaxes[ncoKey]]

    info["lineMapping"] = line_mapping
    info["lineEventTimes"] = np.asarray(line_event_times, dtype=float)
    info["lineEventNumbers"] = np.asarray(line_event_numbers, dtype=int)
    info["events"] = parsed_events

    return ncos, info


def readRFEvents(path: str) -> tuple[dict, dict]:
    with open(path + "/" + "_FCube1.xml") as f:
        gCube = xmltodict.parse(f.read())

    ncos, info = getRFEvents(gCube)

    return ncos, info


def read_all_fcube_event_infos(path: str) -> list[dict[str, object]]:
    event_infos: list[dict[str, object]] = []
    fcube_paths = sorted(Path(path).glob("_FCube*.xml"))
    for fcube_path in fcube_paths:
        with fcube_path.open() as handle:
            parsed = xmltodict.parse(handle.read())
        _ncos, info = getRFEvents(parsed)
        info["fcube"] = fcube_path.stem
        event_infos.append(info)
    return event_infos


def build_pulse_program_event_annotations(event_infos: list[dict[str, object]]) -> list[dict]:
    grouped_events: dict[tuple[float, tuple[str, ...]], dict[str, object]] = {}
    for info in event_infos:
        source = str(info.get("fcube", ""))
        parsed_events = info.get("events", [])
        if not isinstance(parsed_events, list):
            continue
        for event in parsed_events:
            if "t" not in event:
                continue

            parts: list[str] = []
            if "seq" in event:
                parts.append(f"seq={event['seq']}")
            if "tr" in event:
                parts.append(f"tr={event['tr']}")
            if "scancmd" in event:
                parts.append(f"scan={event['scancmd']}")
            if "wl" in event:
                parts.append(f"wl={event['wl']}")
            if "wr" in event:
                parts.append(f"wr={event['wr']}")
            if "if" in event:
                parts.append(f"if={event['if']}")
            if "df" in event:
                parts.append(f"df={event['df']}")
            if "rf" in event:
                parts.append(f"rf={event['rf']}")

            if not parts:
                continue
            event_time = float(event["t"])
            signature = (round(event_time, 12), tuple(parts))
            if signature not in grouped_events:
                grouped_events[signature] = {
                    "time": event_time,
                    "parts": tuple(parts),
                    "sources": set(),
                    "count": 0,
                }
            if source:
                grouped_events[signature]["sources"].add(source)
            grouped_events[signature]["count"] += 1

    if not grouped_events:
        return []

    event_times: list[float] = []
    event_texts: list[str] = []
    for event_data in sorted(grouped_events.values(), key=lambda item: float(item["time"])):
        parts = list(event_data["parts"])
        label = " | ".join(parts)
        event_times.append(float(event_data["time"]))
        event_texts.append(label)

    return [
        {
            "name": "ppg_events",
            "t": np.asarray(event_times, dtype=float),
            "texts": event_texts,
            "color": "y",
        },
    ]


def readBrkrChannels(path: str, progress: QProgressDialog, app: QMainWindow)->dict:
    progress.setLabelText("Reading RF Events")

    if progress.wasCanceled():
        return None

    ncos, info = readRFEvents(path)

    app.setWindowTitle(f"{path} originPPG: {info['pulProg']}")
    app.pulseProgramSource = info.get("pulProg")
    app.pulseProgramTimeline = (
        info.get("lineEventTimes"),
        info.get("lineEventNumbers"),
    )
    app.pulseProgramLineMapping = info.get("lineMapping", {})

    if progress.wasCanceled():
        return None
    progress.setValue(40)
    progress.setLabelText("Reading gradients")

    gradTime, grads = readGrads(path)

    progress.setValue(50)

    progress.setLabelText("Preparing plots gradients")

    channels = []
    event_infos = read_all_fcube_event_infos(path)
    event_annotations = build_pulse_program_event_annotations(event_infos)

    for nco in ncos:
        for key in ncos[nco]:
            if key in {"t", "sf"}:
                continue

            if re.match(r"p\d", key):
                plotType = "phase"
            elif key == "pw":
                plotType = "power"
            else:
                plotType = "mag"

            channelDes = {
                "chanLabel": "NCO_" + nco + "_" + key,
                "label": "NCO_" + nco + "_" + key,
                "type": "NCO",
                "ind": nco,
                "key": key,
                "plotType": plotType,
                "t": ncos[nco]["t"],
                "data": ncos[nco][key],
            }

            channelDes["annotations"] = list(event_annotations)

            if key == "am":
                sf = ncos[nco]["sf"]
                t = ncos[nco]["t"]

                # Compute differences to find where frequency changes
                dsf = sf - sf[np.where(sf > 0)[0][0]]

                whenChange = np.abs(np.diff(dsf, prepend=0)) > 0

                # Extract change values and corresponding time points
                sfChanges = dsf[whenChange]
                tsfChanges = t[whenChange]
                if len(sfChanges > 0):
                    channelDes["annotations"].append(
                        {
                            "name": "sf",
                            "t": tsfChanges,
                            "vals": sfChanges * 1e3,
                            "units": "kHz",
                        }
                    )

            channels.append([channelDes])

    channels.append(
        [
            {
                "chanLabel": "Gradients",
                "label": "Gx",
                "type": "grads",
                "ind": str(0),
                "key": "Gx",
                "plotType": "mag",
                "t": gradTime,
                "data": grads[0],
                "raw_data": grads[0].copy(),
                "units": "%",
                "raw_units": "%",
                "annotations": list(event_annotations),
                "pen": "g",
            },
            {
                "label": "Gy",
                "type": "grads",
                "ind": str(1),
                "key": "Gy",
                "plotType": "mag",
                "t": gradTime,
                "data": grads[1],
                "raw_data": grads[1].copy(),
                "units": "%",
                "raw_units": "%",
                "annotations": [],
                "pen": "r",
            },
            {
                "label": "Gz",
                "type": "grads",
                "ind": str(2),
                "key": "Gz",
                "plotType": "mag",
                "t": gradTime,
                "data": grads[2],
                "raw_data": grads[2].copy(),
                "units": "%",
                "raw_units": "%",
                "annotations": [],
                "pen": "b",
            },
        ]
    )

    return channels
