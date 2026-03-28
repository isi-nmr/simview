import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import xmltodict

try:
    from lxml import etree as LET
except ImportError:  # pragma: no cover - optional dependency
    LET = None


def _get_bruker_pw_reference_watts() -> float:
    raw = os.getenv("SIMVIEW_BRUKER_PW_REF_W", "1.0").strip()
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    if value <= 0:
        return 1.0
    return value


def bruker_pw_attenuation_db_to_watts(attenuation_db: np.ndarray | float, reference_watts: float) -> np.ndarray:
    att = np.asarray(attenuation_db, dtype=float)
    # Bruker "pw" is attenuation in dB, so linear power scales by 10^(-dB/10).
    return reference_watts * np.power(10.0, -att / 10.0)


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


def _count_event_tags(xml_path: str) -> int:
    count = 0
    needle = b"<ev"
    with open(xml_path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            count += chunk.count(needle)
    return count


def _is_tag(element_tag: object, name: str) -> bool:
    if not isinstance(element_tag, str):
        return False
    return element_tag == name or element_tag.endswith("}" + name)


def _find_xml_attr(line: bytes, key: bytes) -> bytes | None:
    start = line.find(key)
    if start < 0:
        return None
    start += len(key)
    end = line.find(b'"', start)
    if end < 0:
        return None
    return line[start:end]


def _read_timeunit_from_xml_bytes(xml_path: str) -> float:
    # Read only the beginning of the file where <pulseprogram ... timeunit=...> is expected.
    with open(xml_path, "rb") as handle:
        head = handle.read(2 * 1024 * 1024)

    patterns = [
        rb'timeunit="([^"]+)"',
        rb"timeunit='([^']+)'",
    ]
    for pattern in patterns:
        match = re.search(pattern, head)
        if match is not None:
            return float(match.group(1))

    raise ValueError("Could not detect pulseprogram timeunit in _GCube.xml")


def _readGrads_fast_line(
    gcube_path: str,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_label: str = "Parsing gradient events",
) -> tuple[np.ndarray, np.ndarray]:
    line_ev_count = 0
    token_ev_count = 0
    with open(gcube_path, "rb") as handle:
        for raw_line in handle:
            stripped = raw_line.lstrip()
            token_ev_count += stripped.count(b"<ev")
            if stripped.startswith(b"<ev"):
                line_ev_count += 1

    if line_ev_count == 0:
        return np.zeros(0, dtype=float), np.zeros((3, 0), dtype=float)

    # Guardrail: if many <ev tokens are not line-oriented, fast parser may miss data.
    if token_ev_count > line_ev_count * 1.05:
        raise ValueError("GCube XML is not line-oriented enough for fast parser")

    t_unit = _read_timeunit_from_xml_bytes(gcube_path)
    t = np.zeros(line_ev_count, dtype=float)
    grads = np.zeros((3, line_ev_count), dtype=float)

    progress_step = max(line_ev_count // 200, 1)
    parsed_lines = 0
    grad_index = 0
    with open(gcube_path, "rb") as handle:
        for raw_line in handle:
            stripped = raw_line.lstrip()
            if not stripped.startswith(b"<ev"):
                continue

            if progress_callback is not None and (
                parsed_lines % progress_step == 0 or parsed_lines == line_ev_count - 1
            ):
                progress_callback((parsed_lines + 1) / line_ev_count, progress_label)

            parsed_lines += 1

            g1 = _find_xml_attr(stripped, b'g1="')
            if g1 is None:
                continue
            g2 = _find_xml_attr(stripped, b'g2="')
            g3 = _find_xml_attr(stripped, b'g3="')
            tv = _find_xml_attr(stripped, b't="')
            if g2 is None or g3 is None or tv is None:
                continue

            t[grad_index] = float(tv) * t_unit
            grads[:, grad_index] = [float(g1), float(g2), float(g3)]
            grad_index += 1

    return t[:grad_index], grads[:, :grad_index]


def readGrads(
    path: str,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_label: str = "Parsing gradient events",
) -> tuple[np.ndarray, np.ndarray]:
    gcube_path = path + "/" + "_GCube.xml"
    grad_parser = os.environ.get("SIMVIEW_GRAD_PARSER", "auto").lower()
    if grad_parser not in {"auto", "fast", "xml"}:
        grad_parser = "auto"

    if grad_parser in {"auto", "fast"}:
        try:
            return _readGrads_fast_line(
                gcube_path,
                progress_callback=progress_callback,
                progress_label=progress_label,
            )
        except Exception:
            if grad_parser == "fast":
                raise

    total_events = _count_event_tags(gcube_path)
    progress_step = max(total_events // 200, 1) if total_events > 0 else 1

    time = np.zeros(max(total_events, 1), dtype=float)
    grads = np.zeros((3, max(total_events, 1)), dtype=float)

    t_unit = 1.0
    event_index = 0
    grad_index = 0

    xml_backend = os.environ.get("SIMVIEW_XML_BACKEND", "et").lower()
    parser = LET if xml_backend == "lxml" and LET is not None else ET
    context = parser.iterparse(gcube_path, events=("start", "end"))
    for parse_event, element in context:
        if parse_event == "start" and _is_tag(element.tag, "pulseprogram"):
            raw_timeunit = element.attrib.get("timeunit")
            if raw_timeunit is not None:
                t_unit = float(raw_timeunit)
            continue

        if parse_event != "end" or not _is_tag(element.tag, "ev"):
            continue

        if progress_callback is not None and (
            event_index % progress_step == 0 or event_index == total_events - 1
        ):
            progress_callback((event_index + 1) / max(total_events, 1), progress_label)

        attrs = element.attrib
        event_index += 1
        if "g1" in attrs and "g2" in attrs and "g3" in attrs and "t" in attrs:
            if grad_index >= time.size:
                new_size = max(time.size * 2, grad_index + 1)
                new_time = np.zeros(new_size, dtype=float)
                new_grads = np.zeros((3, new_size), dtype=float)
                new_time[:grad_index] = time[:grad_index]
                new_grads[:, :grad_index] = grads[:, :grad_index]
                time = new_time
                grads = new_grads
            time[grad_index] = float(attrs["t"]) * t_unit
            grads[:, grad_index] = [float(attrs["g1"]), float(attrs["g2"]), float(attrs["g3"])]
            grad_index += 1
        element.clear()

    return time[:grad_index], grads[:, :grad_index]


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

        event_time: float | None = None
        event_record: dict[str, object] = {}
        if "@t" in event:
            event_time = float(event["@t"]) * tUnit
            event_record["t"] = event_time
        for raw_key, raw_value in event.items():
            if not raw_key.startswith("@"):
                continue
            key = raw_key[1:]
            if key == "t":
                continue
            event_record[key] = raw_value
        parsed_events.append(event_record)

        if event_time is None:
            continue

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
    skip_fcube_stems: set[str] | None = None,
) -> list[dict[str, object]]:
    event_infos: list[dict[str, object]] = []
    skipped_stems = skip_fcube_stems if skip_fcube_stems is not None else set()
    fcube_paths = sorted(Path(path).glob("_FCube*.xml"))
    filtered_paths = [fcube_path for fcube_path in fcube_paths if fcube_path.stem not in skipped_stems]
    fcube_paths = filtered_paths
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
            "color": "#8a7600",
        },
    ]


def parseBrkrChannels(
    path: str,
    progress_callback: Callable[[int, str], None] | None = None,
    pw_reference_watts: float | None = None,
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
    if pw_reference_watts is None or pw_reference_watts <= 0:
        pw_reference_watts = _get_bruker_pw_reference_watts()
    event_infos = [dict(info, fcube="_FCube1")]
    event_infos.extend(read_all_fcube_event_infos(
        path,
        progress_callback=lambda fraction, label: emit_progress(55 + int(20 * fraction), label),
        skip_fcube_stems={"_FCube1"},
    ))
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
            if key == "pw":
                raw_pw_db = np.asarray(ncos[nco][key], dtype=float)
                channelDes["raw_data"] = raw_pw_db.copy()
                channelDes["raw_units"] = "dB"
                channelDes["units"] = "W"
                channelDes["data"] = bruker_pw_attenuation_db_to_watts(raw_pw_db, pw_reference_watts)

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


def readBrkrChannels(path: str, progress=None, app=None, pw_reference_watts: float | None = None) -> dict:
    if progress is not None:
        progress.setLabelText("Reading RF Events")
        if hasattr(progress, "wasCanceled") and progress.wasCanceled():
            return None

    def _progress(value: int, label: str) -> None:
        if progress is None:
            return
        progress.setValue(value)
        progress.setLabelText(label)

    if pw_reference_watts is None and app is not None and hasattr(app, "brukerPwReferenceWatts"):
        try:
            candidate = float(getattr(app, "brukerPwReferenceWatts"))
            if candidate > 0:
                pw_reference_watts = candidate
        except (TypeError, ValueError):
            pw_reference_watts = None

    channels, info = parseBrkrChannels(
        path,
        progress_callback=_progress,
        pw_reference_watts=pw_reference_watts,
    )

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
