import re
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import xmltodict


def getGradEvents(dict: dict) -> tuple[np.ndarray, np.ndarray]:
    events = dict["pulseprogram"]["ev"]
    if isinstance(events, Mapping):
        events = [events]

    time = np.zeros(len(events))

    tUnit = float(dict["pulseprogram"]["@timeunit"])

    grads = np.zeros((3, len(events)))
    ind = 0
    for event in events:
        if "@g1" not in event:
            continue

        time[ind] = float(event["@t"]) * tUnit
        grads[:, ind] = [float(event["@g1"]), float(event["@g2"]), float(event["@g3"])]
        ind = ind + 1

    return time[:ind], grads[:, :ind]


def readGrads(
    path: str,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_label: str = "Parsing gradient events",
) -> tuple[np.ndarray, np.ndarray]:
    with open(path + "/" + "_GCube.xml") as f:
        gCube = xmltodict.parse(f.read())

    events = gCube["pulseprogram"]["ev"]
    if isinstance(events, Mapping):
        events = [events]
    total_events = len(events)
    progress_step = max(total_events // 200, 1) if total_events > 0 else 1

    time = np.zeros(total_events)
    grads = np.zeros((3, total_events))
    tUnit = float(gCube["pulseprogram"]["@timeunit"])
    ind = 0
    for event_index, event in enumerate(events):
        if progress_callback is not None and (
            event_index % progress_step == 0 or event_index == total_events - 1
        ):
            progress_callback((event_index + 1) / max(total_events, 1), progress_label)

        if "@g1" not in event:
            continue
        time[ind] = float(event["@t"]) * tUnit
        grads[:, ind] = [float(event["@g1"]), float(event["@g2"]), float(event["@g3"])]
        ind += 1

    return time[:ind], grads[:, :ind]


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


def getRFEvents(
    dict: dict,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_label: str = "Parsing RF events",
) -> tuple[dict, dict]:
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

    events = pulse_program["ev"]
    if isinstance(events, Mapping):
        events = [events]
    total_events = len(events)
    progress_step = max(total_events // 200, 1) if total_events > 0 else 1

    for event_index, event in enumerate(events):
        if progress_callback is not None and (
            event_index % progress_step == 0 or event_index == total_events - 1
        ):
            progress_callback((event_index + 1) / max(total_events, 1), progress_label)

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


def readRFEvents(
    path: str,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_label: str = "Parsing RF events",
) -> tuple[dict, dict]:
    with open(path + "/" + "_FCube1.xml") as f:
        gCube = xmltodict.parse(f.read())

    ncos, info = getRFEvents(gCube, progress_callback=progress_callback, progress_label=progress_label)

    return ncos, info


def read_all_fcube_event_infos(
    path: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[dict[str, object]]:
    event_infos: list[dict[str, object]] = []
    fcube_paths = sorted(Path(path).glob("_FCube*.xml"))
    total_files = len(fcube_paths)
    if total_files == 0:
        return event_infos

    for file_index, fcube_path in enumerate(fcube_paths):
        def _per_file_progress(file_fraction: float, label: str) -> None:
            if progress_callback is None:
                return
            combined = (file_index + file_fraction) / total_files
            progress_callback(combined, label)

        with fcube_path.open() as handle:
            parsed = xmltodict.parse(handle.read())
        _ncos, info = getRFEvents(
            parsed,
            progress_callback=_per_file_progress,
            progress_label=f"Parsing {fcube_path.name}",
        )
        info["fcube"] = fcube_path.stem
        event_infos.append(info)
    if progress_callback is not None:
        progress_callback(1.0, "Parsed FCube files")
    return event_infos


def build_pulse_program_event_annotations(
    event_infos: list[dict[str, object]],
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[dict]:
    grouped_events: dict[tuple[float, tuple[str, ...]], dict[str, object]] = {}
    total_events = 0
    for info in event_infos:
        parsed_events = info.get("events", [])
        if isinstance(parsed_events, list):
            total_events += len(parsed_events)
    processed_events = 0
    progress_step = max(total_events // 200, 1) if total_events > 0 else 1

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

            processed_events += 1
            if progress_callback is not None and (
                processed_events % progress_step == 0 or processed_events == total_events
            ):
                progress_callback(
                    processed_events / max(total_events, 1),
                    "Building event annotations",
                )

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
        if progress_callback is not None:
            progress_callback(1.0, "Building event annotations")
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


def parseBrkrChannels(
    path: str,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[list[list[dict]], dict[str, object]]:
    last_progress_value = 0

    def emit_progress(value: int, label: str) -> None:
        nonlocal last_progress_value
        clamped_value = max(0, min(99, int(value)))
        if clamped_value < last_progress_value:
            clamped_value = last_progress_value
        last_progress_value = clamped_value
        if progress_callback is not None:
            progress_callback(clamped_value, label)

    emit_progress(10, "Reading RF events")
    ncos, info = readRFEvents(
        path,
        progress_callback=lambda fraction, label: emit_progress(10 + int(35 * fraction), label),
        progress_label="Parsing _FCube1.xml events",
    )

    emit_progress(46, "Reading gradients")
    gradTime, grads = readGrads(
        path,
        progress_callback=lambda fraction, label: emit_progress(46 + int(8 * fraction), label),
        progress_label="Parsing _GCube.xml events",
    )

    emit_progress(55, "Parsing FCube event files")
    channels = []
    event_infos = read_all_fcube_event_infos(
        path,
        progress_callback=lambda fraction, label: emit_progress(55 + int(20 * fraction), label),
    )
    emit_progress(76, "Building event annotations")
    event_annotations = build_pulse_program_event_annotations(
        event_infos,
        progress_callback=lambda fraction, label: emit_progress(76 + int(7 * fraction), label),
    )

    # Build NCO channels and emit coarse-grained progress for long runs.
    total_nco_channels = 0
    for nco_key in ncos:
        total_nco_channels += sum(1 for key in ncos[nco_key] if key not in {"t", "sf"})
    built_nco_channels = 0

    emit_progress(83, "Preparing NCO channels")
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
                "annotations": [],
            }

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
            built_nco_channels += 1
            if total_nco_channels > 0:
                progress_value = 83 + int(12 * built_nco_channels / total_nco_channels)
                emit_progress(min(progress_value, 95), "Preparing NCO channels")

    if event_annotations:
        event_times = np.asarray(event_annotations[0].get("t", np.asarray([], dtype=float)), dtype=float)
        channels.append(
            [
                {
                    "chanLabel": "NCO_0_events",
                    "label": "NCO_0_events",
                    "type": "NCO",
                    "ind": "0",
                    "key": "events",
                    "plotType": "mag",
                    "units": "",
                    "t": event_times,
                    "data": np.zeros(event_times.shape, dtype=float),
                    "annotations": list(event_annotations),
                },
            ]
        )

    emit_progress(96, "Preparing gradient channels")
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
                "annotations": [],
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

    emit_progress(98, "Finalizing Bruker channels")
    return channels, info


def readBrkrChannels(path: str, progress=None, app=None) -> dict:
    if progress is not None:
        progress.setLabelText("Reading RF Events")
        if hasattr(progress, "wasCanceled") and progress.wasCanceled():
            return None

    def _progress(value: int, label: str) -> None:
        if progress is None:
            return
        progress.setValue(value)
        progress.setLabelText(label)

    channels, info = parseBrkrChannels(path, progress_callback=_progress)

    if app is not None:
        app.setWindowTitle(f"{path} originPPG: {info['pulProg']}")
        app.pulseProgramSource = info.get("pulProg")
        app.pulseProgramTimeline = (
            info.get("lineEventTimes"),
            info.get("lineEventNumbers"),
        )
        app.pulseProgramLineMapping = info.get("lineMapping", {})

    if progress is not None:
        progress.setValue(98)
        progress.setLabelText("Preparing plots gradients")

    return channels
