"""RF sample and phase handling for the sequence schematic."""

from collections.abc import Mapping

import numpy as np


def schematic_rf_channels(channels: list[list[dict]]) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    signals: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for channel in channels:
        for line in channel:
            key = str(line.get("key", "")).lower()
            if str(line.get("type", "")).upper() != "NCO" or key not in {"am", "p0", "p1", "p2"}:
                continue
            times = np.asarray(line.get("t", []), dtype=float)
            values = np.asarray(line.get("data", []), dtype=float)
            if times.size and times.shape == values.shape:
                signals.setdefault(str(line.get("ind", "1")), {})[key] = (times, values)
    return signals


def recorded_rf_phase(signals: Mapping[str, tuple[np.ndarray, np.ndarray]], times: np.ndarray) -> np.ndarray | None:
    """Sample the recorded transmitter phase registers with a zero-order hold."""
    components = [signals[key] for key in ("p0", "p1", "p2") if key in signals]
    if not components:
        return None
    phase = np.zeros_like(times, dtype=float)
    for timestamps, values in components:
        indices = timestamps.searchsorted(times, side="right") - 1
        valid = indices >= 0
        phase[valid] += values[indices[valid]]
    return np.mod(phase, 360.0)


def compiled_rf_shape(event: dict, structure: dict) -> tuple[int, dict] | None:
    """Find a unique registered RF shape for a program instruction."""
    shape_name = event.get("rf_shape")
    channel = event.get("rf_channel")
    candidates = [
        (nco, variant)
        for (name, nco), variants in structure.get("rf_shapes", {}).items()
        if name == shape_name
        for variant in variants
        if variant.get("channel") in (None, channel)
    ]
    matching_channel = [(nco, shape) for nco, shape in candidates if shape.get("channel", nco) == channel]
    candidates = matching_channel or candidates
    duration = event.get("duration")
    if len(candidates) > 1 and isinstance(duration, (int, float)):
        candidates = [
            (nco, shape) for nco, shape in candidates if np.isclose(duration, shape["duration"], atol=1e-7, rtol=1e-4)
        ]
    return candidates[0] if len(candidates) == 1 else None


def recorded_rf_envelopes(
    events: list[dict], timeline: tuple | None, signals: dict[str, dict],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Find complete recorded RF profiles by their precompiled instruction number."""
    expected = {
        event["internal_line"]: event for event in events
        if event.get("internal_line") is not None and (event.get("duration") or 0) > 0
    }
    if not expected or not signals or timeline is None:
        return {}
    times, lines = map(np.asarray, timeline)
    if times.ndim != 1 or times.size < 2 or times.shape != lines.shape:
        return {}
    boundaries = np.r_[0, np.flatnonzero(lines[1:] != lines[:-1]) + 1, times.size - 1]
    envelopes = {}
    for first, last in zip(boundaries[:-1], boundaries[1:], strict=True):
        line = int(lines[first])
        if line not in expected or line in envelopes:
            continue
        event = expected[line]
        start = float(times[first])
        duration = float(event["duration"])
        if float(times[last]) - start < duration * 0.98:
            continue
        end = min(float(times[last]), start + duration)
        candidates = []
        for nco, channels in signals.items():
            if "am" not in channels:
                continue
            timestamps, amplitude = channels["am"]
            left, right = timestamps.searchsorted([start, end])
            samples = amplitude[left:right]
            if samples.size and (peak := np.max(np.abs(samples))) > 0:
                candidates.append((nco, timestamps[left:right], samples / peak))
        channel_matches = [candidate for candidate in candidates if candidate[0] == str(event.get("rf_channel"))]
        candidates = channel_matches or candidates
        if len(candidates) != 1:
            continue
        nco, timestamps, amplitude = candidates[0]
        sample_times = np.r_[start, timestamps, end]
        magnitude = np.r_[0.0, amplitude, amplitude[-1]]
        phase = recorded_rf_phase(signals[nco], sample_times)
        envelopes[line] = ((sample_times - start) / duration, magnitude, phase)
        if len(envelopes) == len(expected):
            break
    return envelopes
