import os
import re
from collections.abc import Callable, Mapping
from functools import cache
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


def find_origin_acquisition_path(sim_path: str) -> Path | None:
    local_path = Path(sim_path)
    if (local_path / "method").is_file() and (local_path / "acqp").is_file():
        return local_path
    for output_path in sorted(local_path.glob("*.output")):
        try:
            output_text = output_path.read_text(errors="replace")
        except OSError:
            continue
        match = re.search(r"pulse program:\s*(.+?)/pulseprogram\.precomp\s*$", output_text, re.MULTILINE)
        if match is not None:
            acquisition_path = Path(match.group(1))
            if (acquisition_path / "method").is_file() and (acquisition_path / "acqp").is_file():
                return acquisition_path
    return None


def read_origin_rf_pulse_calibrations(sim_path: str) -> tuple[str | None, list[dict[str, object]]]:
    acquisition_path = find_origin_acquisition_path(sim_path)
    if acquisition_path is None:
        return None, []
    try:
        method_text = (acquisition_path / "method").read_text(errors="replace")
    except OSError:
        return str(acquisition_path), []
    calibrations: list[dict[str, object]] = []
    # RF structures use both ``Pul`` (ExcPul, FovSatPul) and ``Pulse``
    # (RefPulse1, DwRfcPulse1) naming across ParaVision methods.
    pattern = re.compile(r"^##\$(\w*(?:Pul|Pulse)\d*)=\(([^)]*)\)", re.MULTILINE)
    for match in pattern.finditer(method_text):
        fields = [field.strip() for field in match.group(2).split(",")]
        if len(fields) < 3:
            continue
        try:
            duration_seconds, flip_angle = float(fields[0]) * 1e-3, float(fields[2])
        except ValueError:
            continue
        if duration_seconds > 0 and flip_angle > 0:
            calibrations.append({"name": match.group(1), "duration": duration_seconds, "flip_angle": flip_angle})
    return str(acquisition_path), calibrations


def read_pulse_program_structure(
    sim_path: str,
    pulse_program: str | None,
    pulse_program_path: str | None = None,
    *,
    include_recorded_gradients: bool = True,
) -> dict[str, object]:
    """Extract the exact precompiled PPG used by the simulation and loop counts."""
    local_path = Path(sim_path)
    output_path = local_path / "_FCube1.output"
    try:
        text = output_path.read_text(errors="replace")
    except OSError:
        return {}
    main_source = Path(pulse_program or "").name
    precomp_match = re.search(r"^pulse program:\s*(.+?)\s*$", text, re.MULTILINE)
    precomp_path = Path(precomp_match.group(1)) if precomp_match else None
    local_precomp = local_path / "pulseprogram.precomp"
    if local_precomp.is_file():
        precomp_path = local_precomp
    records: list[dict[str, object]] = []
    if precomp_path is not None:
        try:
            precomp_lines = precomp_path.read_text(errors="replace").splitlines()
        except OSError:
            precomp_lines = []
        current_source = main_source
        current_source_path = str(precomp_path)
        current_line = 1
        directive_pattern = re.compile(r'^\s*#\s+(\d+)\s+"([^"]+)"(?:\s+\d+)*\s*$')
        for internal_line, raw_line in enumerate(precomp_lines, start=1):
            directive = directive_pattern.match(raw_line)
            if directive:
                current_line = int(directive.group(1))
                current_source_path = directive.group(2)
                current_source = Path(current_source_path).name
                if not main_source or current_source.endswith(".ppg"):
                    main_source = current_source
                continue
            records.append({
                "source": current_source,
                "source_path": current_source_path,
                "line": current_line,
                "internal_line": internal_line,
                "text": raw_line.strip(),
            })
            current_line += 1

    # Older output formats may not retain or expose pulseprogram.precomp.
    record_pattern = re.compile(r'^\s*(\d+)\s+"\'([^\']+)\': line (\d+)--->([^\r\n]*)$', re.MULTILINE)
    records_by_line: dict[int, dict[str, object]] = {}
    for internal_line, source, line, source_text in record_pattern.findall(text):
        # An expanded subroutine can repeat a source location at a new compiler line.
        records_by_line[int(internal_line)] = {
            "source": source, "line": int(line), "text": source_text.removesuffix('"').strip(),
            "internal_line": int(internal_line),
        }
    if not records:
        records = [records_by_line[line] for line in sorted(records_by_line)]
    if not records and pulse_program_path:
        source_path = Path(pulse_program_path)
        try:
            source_lines = source_path.read_text(errors="replace").splitlines()
        except OSError:
            source_lines = []
        if source_lines:
            records = [record for record in records if record["source"] != main_source]
            records.extend(
                {"source": main_source, "line": line_number, "text": source_text.strip()}
                for line_number, source_text in enumerate(source_lines, start=1)
            )

    loop_match = re.search(r"loopcounters:\s*256 items\s*=+\s*(.*?)(?:\n\s*\n\s*delay increments)", text, re.DOTALL)
    loop_array = [int(value) for value in re.findall(r"\b\d+\b", loop_match.group(1))] if loop_match else []
    loop_values = {f"l{index}": value for index, value in enumerate(loop_array[:256])}
    for name, index in re.findall(r"^\s*([A-Za-z_]\w*):\s+14\s+(\d+)\b", text, re.MULTILINE):
        array_index = int(index)
        if array_index < len(loop_array):
            loop_values[name] = loop_array[array_index]
    parameters: dict[str, object] = {"ACQ_scan_type": "Scan_Experiment"}
    delay_match = re.search(
        r"delays \[sec\]:\s*256 items\s*=+\s*(.*?)(?=\n\s*pulses \[usec\])",
        text, re.DOTALL,
    )
    delay_values = [float(value) for value in re.findall(
        r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", delay_match.group(1),
    )] if delay_match else []
    parameters.update({f"d{index}": value for index, value in enumerate(delay_values[:256])})
    pulse_match = re.search(
        r"pulses \[usec\]:\s*256 items\s*=+\s*(.*?)(?=\n\s*loopcounters:)",
        text, re.DOTALL,
    )
    pulse_values = [float(value) * 1e-6 for value in re.findall(
        r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", pulse_match.group(1),
    )] if pulse_match else []
    parameters.update({f"p{index}": value for index, value in enumerate(pulse_values[:256])})
    named_delays = re.findall(r"^\s*([A-Za-z_]\w*):\s+2\s+(\d+)\b", text, re.MULTILINE)
    compiled_delays = {}
    for name, array_index in named_delays:
        index = int(array_index)
        if index < len(delay_values):
            compiled_delays[name.casefold()] = parameters[name] = delay_values[index]
    for name, array_index in re.findall(r"^\s*([A-Za-z_]\w*):\s+4\s+(\d+)\b", text, re.MULTILINE):
        index = int(array_index)
        if index < len(pulse_values):
            parameters[name] = pulse_values[index]
    origin_path = find_origin_acquisition_path(sim_path)
    if origin_path is not None:
        try:
            method_text = (origin_path / "method").read_text(errors="replace")
        except OSError:
            method_text = ""
        for name, raw_value in re.findall(r"^##\$(\w+)=([^\r\n]*)", method_text, re.MULTILINE):
            value = raw_value.strip().strip("<>")
            try:
                parameters[name] = float(value)
            except ValueError:
                parameters[name] = value
        parameters.update(_read_jcamp_numeric_parameters(origin_path / "method"))
        acqp_parameters = _read_jcamp_numeric_parameters(origin_path / "acqp")
        parameters.update(acqp_parameters)
        # ACQP D registers use seconds. Prefer the compiled register table
        # for values that the simulator had available.
        if isinstance(acqp_parameters.get("D"), list):
            for index, value in enumerate(acqp_parameters["D"]):
                parameters.setdefault(f"d{index}", value)
        # The PPG symbol ``de`` is the ACQP DE hardware delay, stored in µs.
        if isinstance(acqp_parameters.get("DE"), (int, float)):
            parameters["de"] = float(acqp_parameters["DE"]) * 1e-6
        gradient_registers = acqp_parameters.get("ACQ_gradient_amplitude", [])
        if isinstance(gradient_registers, list):
            parameters.update({f"g{index}": value for index, value in enumerate(gradient_registers)})
    spatial_dimension = re.fullmatch(r"([123])D", str(parameters.get("PVM_SpatDimEnum", "")), re.IGNORECASE)
    if spatial_dimension:
        parameters.setdefault("PVM_SpatDim", int(spatial_dimension[1]))
    for symbol, parameter in (("PVM_VpTrim", "PVM_VpSpoilerStrength"), ("PVM_ChTrim", "PVM_ChSpoilerStrength")):
        values = parameters.get(parameter)
        if isinstance(values, list):
            parameters.setdefault(symbol, values)
    delay_lists = {}
    for match in re.finditer(
        r'^(Delay|Loopcounter) list "(\w+)"[^\n]*:\s*(\d+) items?\s*\n=+\n([^\n]*(?:\n[ \t]+[^\n]+)*)',
        text, re.MULTILINE,
    ):
        kind, name, count, data = match.groups()
        values = [float(value) for value in re.findall(_GCON_NUMBER, data)]
        if len(values) != int(count):
            continue
        if kind == "Delay":
            parameters[name] = values
            delay_lists[name.casefold()] = values
        elif len(values) == 1 and values[0].is_integer():
            loop_values[name] = int(values[0])
    gradient_ramps: list[list[float]] = []
    gradient_output_path = local_path / "_GCube.output"
    try:
        gradient_output = gradient_output_path.read_text(errors="replace")
    except OSError:
        gradient_output = ""
    gcon_sections = _gcon_sections(gradient_output)
    gradient_function = _gcon_gradient_function(gcon_sections)
    ramped_targets, sampled_gradients = _read_gcon_gradient_points(gradient_output, gcon_sections, gradient_function)
    ramp_matches = list(re.finditer(
        r"^ramp\s+(\d+):\s+(\d+)\s+items\s*\n-+\s*\n(.*?)(?=\n\s*(?:ramp\s+\d+:|pulse increments|\Z))",
        gradient_output, re.MULTILINE | re.DOTALL,
    ))
    for match in ramp_matches:
        ramp_index, expected_count = int(match.group(1)), int(match.group(2))
        values = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", match.group(3))]
        if len(values) < expected_count:
            continue
        while len(gradient_ramps) <= ramp_index:
            gradient_ramps.append([])
        # Include exact endpoints; the output table stores intermediate DAC
        # fractions only for the default constant-time ramps.
        gradient_ramps[ramp_index] = [0.0, *values[:expected_count], 1.0]
    configured_ramp_ms = parameters.get("PVM_RampTime")
    configured_ramp_seconds = (
        float(configured_ramp_ms) * 1e-3
        if isinstance(configured_ramp_ms, (int, float)) and configured_ramp_ms > 0 else None
    )
    gradient_commands = {
        record["internal_line"] for record in records
        if "internal_line" in record and re.search(
            r"\b(?:grad_(?:ramp|shape|off)|groff)\b", str(record["text"]).partition(";")[0], re.IGNORECASE,
        )
    }
    compiled_lines = {
        int(number)
        for listing in re.findall(
            r"^[ \t]*\w+:[ \t]+\d+(?:[ \t]+\d+){3}[ \t]+in lines:[ \t]+([\d \t]+)$",
            text, re.MULTILINE,
        )
        for number in listing.split()
    }
    return {
        "source": main_source, "records": records, "loop_values": loop_values,
        "delay_names": [name for name, _ in named_delays] + list(delay_lists),
        "compiled_delay_lists": delay_lists,
        "compiled_delays": compiled_delays, "compiled_lines": compiled_lines,
        "parameters": parameters, "output": str(output_path),
        "precomp": str(precomp_path) if precomp_path is not None else None,
        "rf_shapes": _read_compiled_rf_shapes(text),
        "rf_shape_list_instructions": _read_compiled_rf_shape_list_instructions(text),
        "rf_phase_lists": _read_compiled_phase_lists(text),
        "rf_phase_aliases": {
            name.casefold(): f"ph{register}" for name, register in re.findall(
                r'^Phase list "(\w+)"[^\n]*:\s*1 item\s*\n=+\s*\n[ \t]*(\d+)[ \t]*$',
                text, re.MULTILINE,
            )
        },
        "frequency_lists_hz": _read_compiled_frequency_lists(text),
        "adc_gate_durations": _read_compiled_dwell_times(text),
        "gradient_ramps": gradient_ramps,
        "gradient_vectors": _read_shaped_gradient_vectors(gcon_sections, gradient_function),
        "ramped_gradient_targets": ramped_targets,
        "sampled_gradient_points": sampled_gradients,
        "recorded_gradient_profiles": read_schematic_gradient_profiles(
            local_path / "_GCube.xml", gradient_commands,
            expected_samples=max((len(ramp) - 1 for ramp in gradient_ramps), default=0),
        ) if include_recorded_gradients else {},
        # Ramp tables define normalized interpolation only. ParaVision's
        # physical transition time is configured independently in milliseconds.
        "gradient_ramp_durations": [
            configured_ramp_seconds if configured_ramp_seconds is not None else max(0, len(ramp) - 1) * 1e-6
            for ramp in gradient_ramps
        ],
        "gradient_ramp_time_source": "PVM_RampTime" if configured_ramp_seconds is not None else "Gcon raster fallback",
    }


_GCON_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def _read_compiled_dwell_times(output: str) -> dict[str, float]:
    """Resolve compiled CPD pointers to their sampling gates."""
    pointers = re.split(r"^[ \t]*\[(\d+)\]:[^\n]*\n", output, flags=re.MULTILINE)
    durations = {}
    for pointer, block in zip(pointers[1::2], pointers[2::2], strict=True):
        if not re.search(r"^[ \t]*cpd program:[ \t]+DWELL-PROGRAM\b", block, re.MULTILINE):
            continue
        gates = re.findall(rf"fixdel\(({_GCON_NUMBER}) usec\)[^\n]*\bset=Dwen[AB]\b", block)
        if len(gates) == 1:
            durations[pointer] = float(gates[0]) * 1e-6
    candidates: dict[str, set[float]] = {}
    for delay, pointer in re.findall(
        r"^[ \t]*(\w+)\s+deccmd[^\n]*\bCPDSNG\b[^\n]*\bvia ptr (\d+)\b", output, re.MULTILINE,
    ):
        if pointer in durations:
            candidates.setdefault(delay.casefold(), set()).add(durations[pointer])
    return {delay: next(iter(values)) for delay, values in candidates.items() if len(values) == 1}


def _read_compiled_frequency_lists(output: str) -> dict[str, np.ndarray]:
    headers = re.finditer(
        r'^[ \t]*Frequency list "([^"]+)"[^\n]*?\s*:\s*(\d+) items?\s*\n=+[^\S\n]*\n',
        output, re.MULTILINE | re.IGNORECASE,
    )
    frequency_lists = {}
    for header in headers:
        count = int(header[2])
        values = []
        for line in output[header.end():].splitlines():
            tokens = line.split()
            if not all(re.fullmatch(_GCON_NUMBER, token) for token in tokens):
                break
            values.extend(float(token) for token in tokens)
            if len(values) >= count:
                break
        if count and len(values) == count:
            offsets = np.asarray(values) * 1e6  # The compiler stores frequency offsets in MHz.
            if np.isfinite(offsets).all():
                frequency_lists[header[1].casefold()] = offsets
    return frequency_lists


def _read_compiled_rf_table(block: str, label: str) -> np.ndarray | None:
    header = re.search(
        rf"^[ \t]*{re.escape(label)}:[ \t]*(\d+)[ \t]+items?[ \t]*\n[ \t]*-+[ \t]*$",
        block, flags=re.MULTILINE | re.IGNORECASE,
    )
    if header is None:
        return None
    count = int(header[1])
    values = []
    for line in block[header.end():].splitlines():
        tokens = line.split()
        if not all(re.fullmatch(_GCON_NUMBER, token) for token in tokens):
            break
        values.extend(float(token) for token in tokens)
        if len(values) > count:
            return None
    if not values or len(values) != count:
        return None
    samples = np.asarray(values, dtype=float)
    return samples if np.isfinite(samples).all() else None


def _read_compiled_rf_shapes(output: str) -> dict[tuple[str, int], list[dict[str, object]]]:
    headers = list(re.finditer(
        r"^[ \t]*shape(?:[ \t]+list[ \t]*\[(\d+)\]:|"
        r"[ \t]*\[\d+\]:[ \t]*(\w+)[ \t]+on[ \t]+NCO[ \t]*(\d+)\b)",
        output, flags=re.MULTILINE | re.IGNORECASE,
    ))
    power_channels = {
        (int(pointer), int(nco)): int(channel)
        for pointer, channel, nco in re.findall(
            r"^[ \t]*\[(\d+)\]:[ \t]*Power for f(\d+) on NCO(\d+):",
            output, flags=re.MULTILINE | re.IGNORECASE,
        )
    }
    shapes: dict[tuple[str, int], list[dict[str, object]]] = {}
    shape_list = None
    for index, header in enumerate(headers):
        if header[1] is not None:
            shape_list = int(header[1])
            continue
        end = headers[index + 1].start() if index + 1 < len(headers) else len(output)
        block = output[header.end():end]
        duration_match = re.search(rf"\bdur[ \t]*=[ \t]*({_GCON_NUMBER})[ \t]+us\b", block, re.IGNORECASE)
        if duration_match is None:
            continue
        duration = float(duration_match[1]) * 1e-6
        amplitude = _read_compiled_rf_table(block, "amplitudes")
        if amplitude is None or not np.isfinite(duration) or duration <= 0.0:
            continue
        peak = float(np.max(np.abs(amplitude)))
        if peak:
            amplitude /= peak
        phase_words = _read_compiled_rf_table(block, "phases")
        phase = None
        if phase_words is not None and phase_words.shape == amplitude.shape and np.all(
            (phase_words >= 0) & (phase_words < 65536) & (phase_words == np.floor(phase_words))
        ):
            # The shape registry stores the phase of each slice as a 16-bit turn.
            phase = phase_words * (360.0 / 65536)
        nco = int(header[3])
        shape: dict[str, object] = {"duration": duration, "amplitude": amplitude, "phase": phase}
        offset = re.search(rf"\boffsFreq[ \t]*=[ \t]*({_GCON_NUMBER})", block.partition("\n")[0])
        if offset is not None and np.isfinite(hz := float(offset[1]) * 1e6):
            shape["offset_hz"] = hz
        if shape_list is not None:
            shape["shape_list"] = shape_list
            shape["shape_index"] = int(re.search(r"\[(\d+)\]", header[0])[1])
        power = re.search(rf"\bpower\s*=\s*({_GCON_NUMBER})\s*W\b", block.partition("\n")[0])
        if power is not None:
            shape["power_watts"] = float(power[1])
        pointer = re.search(r"\bpoino[ \t]*=[ \t]*(\d+)\b", block, re.IGNORECASE)
        channel = power_channels.get((int(pointer[1]), nco)) if pointer is not None else None
        if channel is not None:
            shape["channel"] = channel
        shapes.setdefault((header[2].lower(), nco), []).append(shape)
    return shapes


def _read_compiled_rf_shape_list_instructions(output: str) -> dict[int, int]:
    """Bind shape-list pointers to the source pulse that emitted the instruction."""
    blocks = re.split(r"^[ \t]*(\d+)[ \t]+\"'[^\n]+?': line \d+--->", output, flags=re.MULTILINE)
    instructions = {}
    for line, block in zip(blocks[1::2], blocks[2::2], strict=True):
        source, _, emitted = block.partition("\n")
        pulse = re.search(r"\(\s*(\w+)\s*:", source)
        if pulse:
            pointer = re.search(
                rf"^[ \t]*{re.escape(pulse[1])}\s+Shape Pulse[^\n]*\bshape pulse from list via ptr (\d+)\b",
                emitted, re.MULTILINE | re.IGNORECASE,
            )
            if pointer:
                instructions[int(line)] = int(pointer[1])
    return instructions


def _read_compiled_phase_lists(output: str) -> dict[tuple[str, int], np.ndarray]:
    headers = list(re.finditer(
        r"^[ \t]*Phaselist[ \t]+#\d+[ \t]+(\w+)[ \t]+on\b",
        output, flags=re.MULTILINE | re.IGNORECASE,
    ))
    phase_lists: dict[tuple[str, int], np.ndarray] = {}
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(output)
        block = output[header.end():end]
        nco = re.search(r"\bNCO[ \t]*=[ \t]*(\d+)\b", block, flags=re.IGNORECASE)
        phases = _read_compiled_rf_table(block, "original list")
        if nco is not None and phases is not None:
            phase_lists.setdefault((header[1].lower(), int(nco[1])), phases)
    return phase_lists


def _gcon_sections(output: str) -> dict[str, str]:
    parts = re.split(r"^Gcon:[ \t]*([^\r\n]+)\r?\n-+[ \t]*$", output, flags=re.MULTILINE)
    return {name.strip(): body for name, body in zip(parts[1::2], parts[2::2], strict=True)}


def _gcon_gradient_function(sections: dict[str, str]) -> Callable[..., np.ndarray | None]:
    """Resolve a point's function relation, including chained multiplicative modifiers."""
    shapes = {}
    shape_section = sections.get("Gradient Shape Values", "")
    for match in re.finditer(
        r"^shape +([0-9]+): +(\d+) +items?\s*\n-+\s*\n(.*?)(?=^shape +\d+|\Z)",
        shape_section, re.MULTILINE | re.DOTALL,
    ):
        index, sample_count = map(int, match.groups()[:2])
        samples = np.array([float(value) for value in re.findall(_GCON_NUMBER, match[3])])
        if samples.size == sample_count and np.isfinite(samples).all():
            shapes[index] = samples
    relations = {
        int(index): (int(shape), float(scale), int(modifier))
        for index, shape, scale, modifier in re.findall(
            rf"^[ \t]*(\d+):[ \t]+\d+[ \t]+(\d+)[ \t]+({_GCON_NUMBER})[ \t]+(-?\d+)[ \t]*$",
            sections.get("Gradient Points to Function Relation", ""), re.MULTILINE,
        )
    }
    @cache
    def function_values(index: int, sample_count: int | None = None) -> np.ndarray | None:
        values = np.ones((1, 1)) if sample_count else np.ones(1)
        visited = set()
        while index >= 0:
            if index not in relations or index in visited:
                return None
            visited.add(index)
            shape, scale, index = relations[index]
            samples = shapes.get(shape)
            if samples is None:
                return None
            if sample_count:
                # Modifiers with a different length select encodes, while
                # waveform samples run along the time axis.
                samples = samples[:, None] if samples.size not in {1, sample_count} else samples[None, :]
                if values.shape[0] != 1 and samples.shape[0] not in {1, values.shape[0]}:
                    return None
            elif values.size != 1 and samples.size not in {1, values.size}:
                return None
            values = values * (scale * samples)
        return values if np.isfinite(values).all() else None

    return function_values


def _read_shaped_gradient_vectors(
    sections: dict[str, str], function_values: Callable[[int], np.ndarray | None],
) -> list[np.ndarray]:
    """Read the compiled XYZ samples for shaped Gcon vector points, in program order."""
    # The XYZB0 vector in this table names three gradient channels and B0.
    axes = re.search(r"\b0:\s*0,\s*1,\s*2,\s*3\b", sections.get("GradientVectors", ""))
    if not axes:
        return []

    vectors = []
    points = sections.get("GradientVectorPoints", "")
    for match in re.finditer(
        r"^[ \t]*\d+:[ \t]+gvec_idx=[ \t]*(\d+)\b(.*?)(?=^[ \t]*\d+:[ \t]+gvec_idx=|\Z)",
        points, re.MULTILINE | re.DOTALL,
    ):
        if int(match[1]) != 0:
            continue
        rows = re.findall(
            rf"^[ \t]*Shaped[ \t]+({_GCON_NUMBER}),[ \t]*(\d+)[ \t]+(\d+):[ \t]+"
            rf"({_GCON_NUMBER})[ \t]+(\d+)[ \t]+(\d+)[ \t]+(\d+)[ \t]+(\d+)[ \t]*$",
            match[2], re.MULTILINE,
        )
        if len(rows) != 4:
            return []
        sample_count = int(rows[0][2])
        if not sample_count or any(int(row[2]) != sample_count for row in rows):
            return []
        vector = np.zeros((3, sample_count))
        for axis, (delay, delay_index, _, base, first, count, first_sample, sample_relations) in enumerate(rows[:3]):
            if float(delay) or int(delay_index):
                return []
            values = vector[axis]
            values[:] = float(base)
            for relation_start, relation_count in ((first, count), (first_sample, sample_relations)):
                for index in range(int(relation_start), int(relation_start) + int(relation_count)):
                    samples = function_values(index)
                    if samples is None or samples.size not in {1, sample_count}:
                        return []
                    values += samples
        vectors.append(vector * 100.0)
    return vectors


def _read_gcon_gradient_points(
    output: str, sections: dict[str, str], function_values: Callable[..., np.ndarray | None],
) -> tuple[list[tuple[np.ndarray, ...] | None], dict[int, np.ndarray]]:
    """Decode ramp targets and sampled shapes from the first slice of the XYZ point table."""
    header = re.search(r"\bn_gpoints\s*=\s*(\d+)[^\n]*\bn_slices\s*=\s*(\d+)", output)
    axes = re.search(r"\bdigsgradchanmap\s*=\s*0,\s*1,\s*2\b", output)
    section = sections.get("Gradient Points")
    if header is None or axes is None or section is None:
        return [], {}
    point_count, slice_count = map(int, header.groups())
    if not point_count or not slice_count:
        return [], {}

    rows = re.findall(
        rf"^[ \t]*(\w+)\s+({_GCON_NUMBER}),\s*(\d+)\s+(\d+):\s+"
        rf"({_GCON_NUMBER})\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)[ \t]*$",
        section, re.MULTILINE,
    )
    axis_stride = point_count * slice_count
    if len(rows) != 3 * axis_stride:
        return [], {}

    def axis_values(row: tuple[str, ...], sample_count: int | None) -> np.ndarray | None:
        kind, delay, delay_index, count, base, first, functions, first_sample, sample_functions = row
        if float(delay) != 0 or delay_index != "0":
            return None
        if sample_count is None:
            if kind != "Ramped" or count != "1" or sample_functions != "0":
                return None
        elif kind != "Shaped" or int(count) != sample_count:
            return None
        values = np.full((1, sample_count), float(base)) if sample_count else np.array([float(base)])
        for start, length in ((first, functions), (first_sample, sample_functions)):
            for index in range(int(start), int(start) + int(length)):
                samples = function_values(index, sample_count)
                if samples is None or (values.shape[0] != 1 and samples.shape[0] not in {1, values.shape[0]}):
                    return None
                values = values + samples
        if not np.isfinite(values).all():
            return None
        return values[:1] if sample_count is None and np.all(values == values[0]) else values

    targets = []
    sampled = {}
    for point in range(point_count):
        xyz = [rows[axis * axis_stride + point] for axis in range(3)]
        sample_count = int(xyz[0][3]) if xyz[0][0] == "Shaped" else None
        values = tuple(axis_values(row, sample_count) for row in xyz)
        complete = all(value is not None for value in values)
        targets.append(values if complete and sample_count is None else None)
        if complete and sample_count:
            variants = {value.shape[0] for value in values if value.shape[0] > 1}
            if len(variants) <= 1:
                waveform = np.stack(np.broadcast_arrays(*values)) * 100.0
                sampled[point] = waveform[:, 0] if waveform.shape[1] == 1 else waveform
    return targets, sampled


def build_compiled_epi_waveform(
    structure: Mapping[str, object], records: list[dict],
) -> dict[str, object] | None:
    """Reconstruct a shaped EPI readout from compiler tables and method timing."""
    vectors = structure.get("gradient_vectors", [])
    if not vectors:
        return None
    parameters = structure.get("parameters", {})
    codes = [str(record.get("text", "")).partition(";")[0].strip() for record in records]
    job = re.search(r"\badc_job(\d+)\b", codes[0], re.IGNORECASE) if codes else None
    if not job:
        return None
    gate = f"dwgen_job{job[1]}"
    gate_indices = [index for index, code in enumerate(codes) if re.search(rf"\b{gate}\b", code, re.IGNORECASE)]
    if len(gate_indices) != 1:
        return None

    control_index = next(
        (index for index, code in enumerate(codes) if re.fullmatch(r"0u\s+gc_control", code, re.IGNORECASE)),
        None,
    )
    loop_match = None
    loop_index = None
    gradient_indices = []
    delay_names = []
    for index, code in enumerate(codes):
        loop = re.fullmatch(r"loop\s+(\w+)", code, re.IGNORECASE)
        if loop:
            if loop_match is not None:
                return None
            loop_match, loop_index = loop, index
        if re.search(r"\bgrad_shape\b", code, re.IGNORECASE):
            shape = re.match(r"(\w+)\s+grad_shape\s*<\s*xyzb0\s*>", code, re.IGNORECASE)
            if not shape:
                return None
            gradient_indices.append(index)
            delay_names.append(shape[1])
    if control_index is None or loop_match is None or len(gradient_indices) != len(vectors):
        return None
    loop_close = next((index for index in range(loop_index + 1, len(codes)) if codes[index] == "}"), None)
    if loop_close is None or not (control_index < gradient_indices[0] <= loop_index < loop_close < gate_indices[0]):
        return None

    timing_names = (
        "PVM_EpiGradDwellTime", "PVM_EpiPlateau", "PVM_EpiModuleTime",
        "EpiD0", "EpiD1", "EpiD2", "EpiD3", loop_match[1],
    )
    timing = [parameters.get(name) for name in timing_names]
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0 for value in timing):
        return None
    dwell_us, plateau_ms, module_ms, tail, before_sampling, sampling, after_sampling, repeat_count = timing
    dwell, plateau, duration = dwell_us * 1e-6, plateau_ms * 1e-3, module_ms * 1e-3
    if min(dwell, plateau, sampling, repeat_count) <= 0 or int(repeat_count) != repeat_count:
        return None
    # Gcon runs in parallel with the FCube delays following gc_control.
    # Their sum excludes the setup at the beginning of the EPI module.
    setup = duration - (before_sampling + sampling + after_sampling + tail)
    if setup < 0 or not all(control_index < index < gate_indices[0] for index in gradient_indices):
        return None

    segments = []
    shape_durations = {}
    for index, delay_name, values in zip(gradient_indices, delay_names, vectors, strict=True):
        delay = parameters.get(delay_name)
        if not isinstance(delay, (int, float)):
            delay = plateau if delay_name.lower() == "epid11" else values.shape[1] * dwell
        if not np.isfinite(delay) or delay <= 0:
            return None
        segments.append((index, delay, values))
        shape_durations[delay_name] = delay
    prefix = [(delay, values) for index, delay, values in segments if index < loop_index]
    loop_body = [(delay, values) for index, delay, values in segments if loop_index < index < loop_close]
    suffix = [(delay, values) for index, delay, values in segments if loop_close < index]
    if not loop_body:
        return None
    loop_start = setup + sum(delay for delay, _ in prefix)
    loop_end = loop_start + repeat_count * sum(delay for delay, _ in loop_body)
    train_end = loop_end + sum(delay for delay, _ in suffix)
    if train_end > duration + 1e-10:
        return None

    cursor = setup
    times = [np.array([0.0])] if setup > 0 else []
    values = [np.zeros((3, 1))] if setup > 0 else []
    for delay, samples in prefix + loop_body * int(repeat_count) + suffix:
        sample_count = samples.shape[1]
        times.append(cursor + np.arange(sample_count) * (delay / sample_count))
        values.append(samples)
        cursor += delay
    times.append(np.array([duration]))
    values.append(values[-1][:, -1:])
    return {
        "duration": duration,
        "gradient_time": np.concatenate(times),
        "gradient_values": np.concatenate(values, axis=1),
        "gradient_shape_durations": shape_durations,
        "gradient_train_duration": train_end - setup,
        "acquisition_windows": [(setup + before_sampling, setup + before_sampling + sampling)],
        "loop_bounds": (loop_start, loop_end),
    }


def _read_jcamp_numeric_parameters(path: Path) -> dict[str, object]:
    """Read scalar and numeric-array values from a Bruker JCAMP parameter file."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {}
    entries = list(re.finditer(r"^##\$(\w+)=([^\r\n]*)", text, re.MULTILINE))
    result: dict[str, object] = {}
    for index, match in enumerate(entries):
        name, header = match.group(1), match.group(2).strip()
        if header.startswith("("):
            body_end = entries[index + 1].start() if index + 1 < len(entries) else len(text)
            body = text[match.end():body_end]
            values: list[float] = []
            for token in re.findall(r"@\d+\*\([-+]?\d+(?:\.\d+)?\)|[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", body):
                repeated = re.fullmatch(r"@(\d+)\*\(([-+]?\d+(?:\.\d+)?)\)", token)
                if repeated:
                    values.extend([float(repeated.group(2))] * int(repeated.group(1)))
                else:
                    values.append(float(token))
            if values:
                result[name] = values
            continue
        try:
            result[name] = float(header.strip("<>"))
        except ValueError:
            pass
    return result


def bruker_pw_attenuation_db_to_watts(attenuation_db: np.ndarray | float, reference_watts: float) -> np.ndarray:
    att = np.asarray(attenuation_db, dtype=float)
    # Bruker "pw" is attenuation in dB, so linear power scales by 10^(-dB/10).
    return reference_watts * np.power(10.0, -att / 10.0)


def _is_rgp_active(value: object) -> bool:
    text = str(value).strip().lower()
    if not text:
        return False

    if "x" in text:
        try:
            return int(text.rsplit("x", 1)[1], 16) != 0
        except ValueError:
            return False

    normalized = re.sub(r"[^0-9a-f]", "", text)
    if not normalized:
        return False
    try:
        return int(normalized, 16) != 0
    except ValueError:
        return False


def _is_acquisition_end_event(event: Mapping[str, object]) -> bool:
    return any(key in event for key in ("wr", "if", "df", "rf"))


def _dwell_program_line(event: Mapping[str, object]) -> tuple[str, str] | None:
    # Bruker encodes the DWELL-PROGRAM number followed by a six-digit
    # line number. Both seven- and eight-digit markers occur in simulations.
    number = str(event.get("ln", ""))
    program, line = number[:-6], number[-6:]
    if line in {"000000", "000001"} and program.isdecimal():
        return program, line[-1]
    return None


def _is_sample_acquisition_end_event(event: Mapping[str, object]) -> bool:
    dwell_line = _dwell_program_line(event)
    return dwell_line is not None and dwell_line[1] == "1"


def _read_fcube_job_markers(path: str, fcube_stem: str) -> tuple[dict[str, str], set[str]]:
    output_path = Path(path) / f"{fcube_stem}.output"
    job_line_map: dict[str, str] = {}
    eos_lines: set[str] = set()
    try:
        handle = output_path.open(errors="replace")
    except OSError:
        return job_line_map, eos_lines
    # Long records may omit their closing quote in the compiled listing.
    line_pattern = re.compile(r'^\s*(\d+)\s+"[^\n]*?--->(.*)')
    acquisition_pattern = re.compile(r"\b(adc|dwgen|eos[a-z0-9]*)_(\w+)\b", re.IGNORECASE)
    numeric_pattern = re.compile(r"\b\w+_job(\d+)\b", re.IGNORECASE)
    with handle:
        for line in handle:
            match = line_pattern.search(line)
            if match is None:
                continue
            line_number, code = match.groups()
            code = code.partition(";")[0]
            acquisition = acquisition_pattern.search(code)
            if acquisition is not None:
                opcode, job = acquisition.groups()
                numeric_job = re.fullmatch(r"job(\d+)", job, re.IGNORECASE)
                job_line_map[line_number] = numeric_job[1] if numeric_job is not None else job
                if opcode.lower().startswith("eos"):
                    eos_lines.add(line_number)
            elif numeric_job := numeric_pattern.search(code):
                job_line_map[line_number] = numeric_job[1]
    return job_line_map, eos_lines


def annotate_fcube_event_jobs(
    info: dict[str, object], job_line_map: Mapping[str, str], eos_lines: set[str] | None = None,
) -> None:
    events = info.get("events", [])
    if not isinstance(events, list) or not job_line_map:
        return
    if eos_lines:
        info["acquisitionEosLines"] = eos_lines

    current_job: str | None = None
    dwell_jobs: dict[str, str] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        line_number = str(event.get("ln", ""))
        if line_number in job_line_map:
            current_job = job_line_map[line_number]
            event["job"] = current_job
        dwell_line = _dwell_program_line(event)
        if dwell_line is not None:
            program, line = dwell_line
            if line == "0" and current_job is not None and _is_rgp_active(event.get("rgp", "")):
                dwell_jobs[program] = current_job
            job = dwell_jobs.get(program, current_job)
            if job is not None:
                event["job"] = job
        elif current_job is not None and "rgp" in event:
            event["job"] = current_job


def _extract_windows_from_events(
    events: list[dict[str, object]],
    end_event_predicate: Callable[[Mapping[str, object]], bool],
) -> dict[str, list[tuple[float, float]]]:
    starts_by_key: dict[tuple[str, str | None], list[float]] = {}
    end_times_by_key: dict[tuple[str | None], list[float]] = {}
    saw_job = False

    for event in events:
        if "t" not in event:
            continue
        event_time = float(event["t"])
        job = str(event["job"]) if "job" in event else None
        if job is not None:
            saw_job = True
        if "rgp" in event and _is_rgp_active(event["rgp"]):
            nco = str(event.get("nco", "1"))
            starts_by_key.setdefault((nco, job), []).append(event_time)
        if end_event_predicate(event):
            end_times_by_key.setdefault((job,), []).append(event_time)

    if not starts_by_key or not end_times_by_key:
        return {}

    windows_by_nco: dict[str, list[tuple[float, float]]] = {}
    for (nco, job), starts in starts_by_key.items():
        starts = sorted(starts)
        sorted_ends = sorted(end_times_by_key.get((job,), []))
        if not sorted_ends and not saw_job:
            sorted_ends = sorted(end_time for end_times in end_times_by_key.values() for end_time in end_times)
        if not sorted_ends:
            continue
        end_index = 0
        windows: list[tuple[float, float]] = []
        durations: list[float] = []

        for start_index, start_time in enumerate(starts):
            next_start = starts[start_index + 1] if start_index + 1 < len(starts) else None
            while end_index < len(sorted_ends) and sorted_ends[end_index] <= start_time:
                end_index += 1

            if end_index < len(sorted_ends) and (next_start is None or sorted_ends[end_index] < next_start):
                end_time = sorted_ends[end_index]
                windows.append((start_time, end_time))
                durations.append(end_time - start_time)
                end_index += 1
            elif durations:
                windows.append((start_time, start_time + float(np.median(durations))))

        if windows:
            windows_by_nco.setdefault(nco, []).extend(windows)

    for nco, windows in windows_by_nco.items():
        windows_by_nco[nco] = sorted(windows)

    return windows_by_nco


def extract_acquisition_windows(event_infos: list[dict[str, object]]) -> dict[str, list[tuple[float, float]]]:
    return extract_best_windows(event_infos, _is_acquisition_end_event)


def extract_sample_acquisition_windows(event_infos: list[dict[str, object]]) -> dict[str, list[tuple[float, float]]]:
    return _group_sample_windows_by_nco(extract_sample_acquisition_window_details(event_infos))


def _group_sample_windows_by_nco(details: list[dict[str, object]]) -> dict[str, list[tuple[float, float]]]:
    windows: dict[str, list[tuple[float, float]]] = {}
    for detail in details:
        windows.setdefault(str(detail["nco"]), []).append((float(detail["start"]), float(detail["end"])))
    return windows


def extract_sample_acquisition_window_details(event_infos: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return de-duplicated sample windows with the originating job-line type."""
    details: dict[tuple[str, float], tuple[bool, dict[str, object]]] = {}
    for info in event_infos:
        starts: dict[tuple[str, str | None, str], float] = {}
        eos_lines = info.get("acquisitionEosLines", ())
        for event in info.get("events", []):
            if not isinstance(event, dict) or "t" not in event:
                continue
            event_time = float(event["t"])
            dwell_line = _dwell_program_line(event)
            program = dwell_line[0] if dwell_line is not None else None
            job = str(event.get("job", ""))
            nco = str(event.get("nco", ""))
            if "rgp" in event and _is_rgp_active(event["rgp"]):
                starts[(nco or "1", program, job)] = event_time
                continue
            exact = dwell_line is not None and dwell_line[1] == "1"
            if exact:
                matches = (
                    (key, time) for key, time in starts.items()
                    if time < event_time and nco in {"", "0", key[0]}
                    and (key[1] == program or (key[1] is None and (not job or job == key[2])))
                )
                match = max(matches, key=lambda pair: (pair[0][1] == program, pair[1]), default=None)
            elif job and str(event.get("ln", "")) in eos_lines and _is_acquisition_end_event(event):
                matches = (
                    (key, time) for key, time in starts.items()
                    if time < event_time and key[2] == job
                )
                match = max(matches, key=lambda pair: pair[1], default=None)
            else:
                continue
            if match is None:
                continue
            (receive_nco, start_program, start_job), start_time = match
            identity = receive_nco, start_time
            # The same start can occur in multiple FCubes. Prefer a dwell end
            # over a later EOS write and never infer a duration from another job.
            if identity in details and (details[identity][0] or not exact):
                continue
            job_type = start_job if start_job and not start_job.isdecimal() else (
                f"job {start_program}" if start_program is not None else "ADC"
            )
            details[identity] = exact, {
                "nco": receive_nco, "start": start_time, "end": event_time, "job_type": job_type,
            }
    return [details[key][1] for key in sorted(details, key=lambda key: (key[1], key[0]))]


def extract_best_windows(
    event_infos: list[dict[str, object]],
    end_event_predicate: Callable[[Mapping[str, object]], bool],
) -> dict[str, list[tuple[float, float]]]:
    best_windows: dict[str, list[tuple[float, float]]] = {}
    best_count = 0

    for info in event_infos:
        events = info.get("events", [])
        if not isinstance(events, list):
            continue
        windows = _extract_windows_from_events(events, end_event_predicate)
        window_count = sum(len(nco_windows) for nco_windows in windows.values())
        if window_count > best_count:
            best_windows = windows
            best_count = window_count

    return best_windows


def apply_windows_to_nco_gate(
    ncos: dict,
    windows_by_nco: dict[str, list[tuple[float, float]]],
    key: str,
) -> None:
    for nco, windows in windows_by_nco.items():
        if nco not in ncos or not windows:
            continue

        times: list[float] = []
        values: list[float] = []
        for start_time, end_time in windows:
            if end_time <= start_time:
                continue
            times.extend([float(start_time), float(end_time)])
            values.extend([1.0, 0.0])

        if not times:
            continue

        ncos[nco][f"{key}_t"] = np.asarray(times, dtype=float)
        ncos[nco][key] = np.asarray(values, dtype=float)


def apply_acquisition_windows_to_rgp(ncos: dict, acquisition_windows: dict[str, list[tuple[float, float]]]) -> None:
    apply_windows_to_nco_gate(ncos, acquisition_windows, "rgp")


def _gradient_attr_sort_key(attr_name: str) -> int:
    match = re.fullmatch(r"g(\d+)", attr_name)
    if match is None:
        return 10**9
    return int(match.group(1))


def _gradient_attr_names_from_mapping(attrs: Mapping[str, object], prefix: str = "") -> list[str]:
    return sorted(
        [
            key[len(prefix):]
            for key in attrs
            if isinstance(key, str) and re.fullmatch(rf"{re.escape(prefix)}g\d+", key) is not None
        ],
        key=_gradient_attr_sort_key,
    )


def _gradient_display_name(attr_name: str) -> str:
    return {
        "g1": "Gx",
        "g2": "Gy",
        "g3": "Gz",
        "g4": "B0",
    }.get(attr_name, attr_name)


def getGradEvents(dict: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    events = dict["pulseprogram"]["ev"]
    if isinstance(events, Mapping):
        events = [events]

    time = np.zeros(len(events))

    tUnit = float(dict["pulseprogram"]["@timeunit"])

    gradient_attr_names: list[str] = []
    for event in events:
        gradient_attr_names.extend(_gradient_attr_names_from_mapping(event, prefix="@"))
    gradient_attr_names = sorted(set(gradient_attr_names), key=_gradient_attr_sort_key)

    grads = np.zeros((len(gradient_attr_names), len(events)))
    ind = 0
    for event in events:
        if "@t" not in event or not gradient_attr_names:
            continue

        time[ind] = float(event["@t"]) * tUnit
        grads[:, ind] = [float(event.get(f"@{attr_name}", 0.0)) for attr_name in gradient_attr_names]
        ind = ind + 1

    return time[:ind], gradient_attr_names, grads[:, :ind]


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


def _find_gradient_attrs_in_xml_line(line: bytes) -> dict[str, float]:
    gradient_attrs: dict[str, float] = {}
    for match in re.finditer(rb'g(\d+)="([^"]+)"', line):
        gradient_attrs[f"g{int(match.group(1))}"] = float(match.group(2))
    return gradient_attrs


def read_schematic_gradient_profiles(
    path: Path, commands: set[int], *, expected_samples: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Keep the first complete GCube transition for each compiled gradient command."""
    if expected_samples < 2 or not commands or not path.is_file():
        return {}
    time_unit = _read_timeunit_from_xml_bytes(str(path))
    profiles = {}
    active_line = None
    samples = []
    levels = {"g1": 0.0, "g2": 0.0, "g3": 0.0}

    def finish_profile() -> None:
        if active_line not in commands or active_line in profiles or len(samples) < expected_samples:
            return
        first_scan = np.asarray(samples[:expected_samples], dtype=float)
        profiles[active_line] = (
            (first_scan[:, 0] - first_scan[0, 0]) * time_unit, first_scan[:, 1:].T,
        )

    with path.open("rb") as stream:
        for line in stream:
            if not line.lstrip().startswith(b"<ev "):
                continue
            number = _find_xml_attr(line, b' ln="')
            line_id = int(number) if number and number.isdigit() else None
            if line_id is not None and line_id != active_line:
                finish_profile()
                active_line, samples = line_id, []
            gradient = _find_gradient_attrs_in_xml_line(line)
            if not gradient:
                continue
            for axis in levels.keys() & gradient.keys():
                levels[axis] = gradient[axis]
            if line_id not in commands or line_id in profiles or len(samples) >= expected_samples:
                continue
            timestamp = _find_xml_attr(line, b' t="')
            if timestamp is not None:
                samples.append((float(timestamp), *levels.values()))
    finish_profile()
    return profiles


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
) -> tuple[np.ndarray, list[str], np.ndarray]:
    line_ev_count = 0
    token_ev_count = 0
    gradient_attr_names: set[str] = set()
    with open(gcube_path, "rb") as handle:
        for raw_line in handle:
            stripped = raw_line.lstrip()
            token_ev_count += stripped.count(b"<ev")
            if stripped.startswith(b"<ev"):
                line_ev_count += 1
                gradient_attr_names.update(_find_gradient_attrs_in_xml_line(stripped))

    if line_ev_count == 0:
        return np.zeros(0, dtype=float), [], np.zeros((0, 0), dtype=float)

    # Guardrail: if many <ev tokens are not line-oriented, fast parser may miss data.
    if token_ev_count > line_ev_count * 1.05:
        raise ValueError("GCube XML is not line-oriented enough for fast parser")

    ordered_gradient_attr_names = sorted(gradient_attr_names, key=_gradient_attr_sort_key)
    t_unit = _read_timeunit_from_xml_bytes(gcube_path)
    t = np.zeros(line_ev_count, dtype=float)
    grads = np.zeros((len(ordered_gradient_attr_names), line_ev_count), dtype=float)

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

            gradient_values = _find_gradient_attrs_in_xml_line(stripped)
            if not gradient_values:
                continue
            tv = _find_xml_attr(stripped, b't="')
            if tv is None:
                continue

            t[grad_index] = float(tv) * t_unit
            grads[:, grad_index] = [gradient_values.get(attr_name, 0.0) for attr_name in ordered_gradient_attr_names]
            grad_index += 1

    return t[:grad_index], ordered_gradient_attr_names, grads[:, :grad_index]


def readGrads(
    path: str,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_label: str = "Parsing gradient events",
) -> tuple[np.ndarray, list[str], np.ndarray]:
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

    gradient_attr_names: list[str] = []
    time = np.zeros(max(total_events, 1), dtype=float)
    grads = np.zeros((0, max(total_events, 1)), dtype=float)

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
        event_gradient_attr_names = _gradient_attr_names_from_mapping(attrs)
        if event_gradient_attr_names and "t" in attrs:
            merged_gradient_attr_names = sorted(
                set(gradient_attr_names).union(event_gradient_attr_names),
                key=_gradient_attr_sort_key,
            )
            if merged_gradient_attr_names != gradient_attr_names:
                new_grads = np.zeros((len(merged_gradient_attr_names), grads.shape[1]), dtype=float)
                old_index_map = {name: index for index, name in enumerate(gradient_attr_names)}
                for new_index, attr_name in enumerate(merged_gradient_attr_names):
                    old_index = old_index_map.get(attr_name)
                    if old_index is not None:
                        new_grads[new_index, :] = grads[old_index, :]
                grads = new_grads
                gradient_attr_names = merged_gradient_attr_names
            if grad_index >= time.size:
                new_size = max(time.size * 2, grad_index + 1)
                new_time = np.zeros(new_size, dtype=float)
                new_grads = np.zeros((len(gradient_attr_names), new_size), dtype=float)
                new_time[:grad_index] = time[:grad_index]
                new_grads[:, :grad_index] = grads[:, :grad_index]
                time = new_time
                grads = new_grads
            time[grad_index] = float(attrs["t"]) * t_unit
            grads[:, grad_index] = [float(attrs.get(attr_name, 0.0)) for attr_name in gradient_attr_names]
            grad_index += 1
        element.clear()

    return time[:grad_index], gradient_attr_names, grads[:, :grad_index]


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
    info = {
        "pulProg": pulse_program["@path"].split("/")[-1],
        "pulProgPath": pulse_program["@path"],
    }
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
            ncos[ncoNumber]["rgp"][ind] = 1 if _is_rgp_active(event["@rgp"]) else 0

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
    annotate_fcube_event_jobs(info, *_read_fcube_job_markers(path, "_FCube1"))

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
        annotate_fcube_event_jobs(info, *_read_fcube_job_markers(path, fcube_path.stem))
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
            if "rgp" in event:
                rgp_state = "on" if _is_rgp_active(event["rgp"]) else "off"
                parts.append(f"rgp={rgp_state}")
            if _is_sample_acquisition_end_event(event):
                parts.append("acq=end")
            if parts and "job" in event:
                parts.append(f"job={event['job']}")

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
    origin_path, rf_pulse_calibrations = read_origin_rf_pulse_calibrations(path)

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
    gradTime, gradient_attr_names, grads = readGrads(
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
    apply_acquisition_windows_to_rgp(ncos, extract_acquisition_windows(event_infos))
    sample_windows = extract_sample_acquisition_window_details(event_infos)
    apply_windows_to_nco_gate(ncos, _group_sample_windows_by_nco(sample_windows), "acq")
    info["sampleAcquisitionWindows"] = sample_windows
    emit_progress(76, "Building event annotations")
    event_annotations = build_pulse_program_event_annotations(
        event_infos,
        progress_callback=lambda fraction, label: emit_progress(76 + int(7 * fraction), label),
    )

    # Build NCO channels and emit coarse-grained progress for long runs.
    total_nco_channels = 0
    for nco_key in ncos:
        total_nco_channels += sum(1 for key in ncos[nco_key] if key not in {"t", "sf", "rgp_t", "acq_t"})
    built_nco_channels = 0

    emit_progress(83, "Preparing NCO channels")
    for nco in ncos:
        for key in ncos[nco]:
            if key in {"t", "sf", "rgp_t", "acq_t"}:
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
                "t": ncos[nco].get(f"{key}_t", ncos[nco]["t"]),
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
    gradient_pen_map = {
        "g1": "g",
        "g2": "r",
        "g3": "b",
        "g4": "y",
    }
    gradient_channel: list[dict[str, object]] = []
    for grad_index, attr_name in enumerate(gradient_attr_names):
        line = {
            "label": _gradient_display_name(attr_name),
            "type": "grads",
            "ind": str(grad_index),
            "key": _gradient_display_name(attr_name),
            "source_attr": attr_name,
            "plotType": "mag",
            "t": gradTime,
            "data": grads[grad_index],
            "raw_data": grads[grad_index].copy(),
            "units": "%",
            "raw_units": "%",
            "annotations": [],
            "pen": gradient_pen_map.get(attr_name),
        }
        if grad_index == 0:
            line["chanLabel"] = "Gradients"
        gradient_channel.append(line)

    if gradient_channel:
        channels.append(gradient_channel)

    emit_progress(98, "Finalizing Bruker channels")
    info["originAcquisitionPath"] = origin_path
    info["rfPulseCalibrations"] = rf_pulse_calibrations
    info["pulseProgramStructure"] = read_pulse_program_structure(
        path, str(info.get("pulProg", "")), str(info.get("pulProgPath", "")),
    )
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
