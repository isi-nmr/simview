import os
import re
import ast
from bisect import bisect_left
from collections.abc import Collection
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QListWidgetItem
from PyQt6.QtWidgets import QApplication

from utils import dialog
from utils.simUtilsBrkr import build_compiled_epi_waveform

from .constants import _UNSET
from .ppg_loops import collect_ppg_loops, ppg_frequency_offsets, resolve_loop_count
from .schematic_rf import compiled_rf_shape, recorded_rf_envelopes, recorded_rf_phase, schematic_rf_channels


_PPG_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_PPG_TIME = re.compile(rf"{_PPG_NUMBER}[nums]?", re.IGNORECASE)
_PPG_TIME_FACTORS = {"n": 1e-9, "u": 1e-6, "m": 1e-3, "s": 1.0}
_PPG_PLACEHOLDER_COLOR = "#a45309"
_PPG_TIMING_ROW_STEP = 0.52
_PPG_LOOP_ROW_STEP = 0.60
_PPG_MAX_GRADIENT_VARIANTS = 9
_PPG_SECONDS_PER_UNIT = 50e-6
_PPG_PREPARATION_LABELS = {
    "BlBlood": "Black blood",
    "DriftComp": "Drift compensation",
    "Evolution": "Evolution",
    "FatSup": "Fat suppression",
    "FlowSat": "Inflow saturation",
    "FovSat": "FOV saturation",
    "SatTrans": "Saturation transfer",
    "SliceSelIr": "Selective inversion",
    "Tagging": "Tagging",
    "WsOvs": "Water suppression and OVS",
}
_RF_LANE = 5.0
_RF_PHASE_LANE = 4.0
_RF_PHASE_HEIGHT = 0.75
_RF_PHASE_COLOR = "#0f766e"
_SCHEMATIC_LANES = {"RF": _RF_LANE, "RF phase 0°": _RF_PHASE_LANE, "Gx": 3.0, "Gy": 2.0, "Gz": 1.0, "ADC": 0.0}
_PPG_CHANNELS = ("RF", "RF phase", "Gx", "Gy", "Gz", "ADC")
_PPG_GRADIENTS = ("Gx", "Gy", "Gz")
_PPG_GRADIENT_COLORS = ("#15803d", "#dc2626", "#2563eb")
_SCHEMATIC_TICKS = [
    *((position, label) for label, position in _SCHEMATIC_LANES.items()),
    (_RF_PHASE_LANE + _RF_PHASE_HEIGHT / 2, "180°"), (_RF_PHASE_LANE + _RF_PHASE_HEIGHT, "360°"),
]


class SchematicViewBox(pg.ViewBox):
    resetPerformed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.separate_zoom = False
        self.fit_range = None

    def wheelEvent(self, event, axis: int | None = None) -> None:
        if self.separate_zoom and axis is None:
            axis = 1 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 0
        sensitivity = self.state["wheelScaleFactor"]
        if self.separate_zoom and axis == 1:
            self.state["wheelScaleFactor"] = sensitivity * 0.35
        try:
            super().wheelEvent(event, axis)
        finally:
            self.state["wheelScaleFactor"] = sensitivity

    def mouseClickEvent(self, event) -> None:
        if event.double() and event.button() == Qt.MouseButton.LeftButton and self.fit_range is not None:
            x_range, y_range = self.fit_range
            self.setRange(xRange=x_range, yRange=y_range, padding=0)
            self.resetPerformed.emit()
            event.accept()
            return
        super().mouseClickEvent(event)


class InteractionMixin:
    def build_sequence_outline(self) -> list[dict]:
        """Build a compact cycle/module/event hierarchy from simulator timing."""
        timeline = getattr(self, "pulseProgramTimeline", None)
        if timeline is None or len(timeline) != 2:
            return []
        raw_times = np.asarray(timeline[0], dtype=float)
        raw_lines = np.asarray(timeline[1], dtype=int)
        if raw_times.size == 0 or raw_lines.size == 0:
            return []
        count = min(raw_times.size, raw_lines.size)
        order = np.argsort(raw_times[:count], kind="stable")
        raw_times, raw_lines = raw_times[:count][order], raw_lines[:count][order]

        # Collapse repeated XML samples which point to the same source line.
        ppg_events: list[dict] = []
        previous_signature: tuple[str, int] | None = None
        for time_value, internal_line in zip(raw_times, raw_lines, strict=True):
            mapping = getattr(self, "pulseProgramLineMapping", {}).get(int(internal_line), {})
            source = str(mapping.get("source", "PPG"))
            source_line = int(mapping.get("line", 0))
            signature = (source, source_line)
            if signature == previous_signature:
                continue
            previous_signature = signature
            ppg_events.append({
                "time": float(time_value), "source": source, "line": source_line,
                "internal_line": int(internal_line), "gradients": [],
            })

        gradient_channel = next(
            (channel for channel in getattr(self, "channels", []) if channel and channel[0].get("type") == "grads"),
            [],
        )
        gradient_events: list[tuple[float, str]] = []
        if gradient_channel:
            common_time = np.unique(np.concatenate([
                np.asarray(line.get("t", []), dtype=float) for line in gradient_channel
                if np.asarray(line.get("t", []), dtype=float).size
            ]))
            previous = np.zeros(len(gradient_channel), dtype=float)
            for time_value in common_time:
                values = np.asarray([
                    float(np.interp(time_value, np.asarray(line.get("t", []), dtype=float),
                                    np.asarray(line.get("raw_data", line.get("data", [])), dtype=float)))
                    for line in gradient_channel
                ])
                changed = ~np.isclose(values, previous, atol=1e-12, rtol=0.0)
                if np.any(changed):
                    parts = [
                        f"{gradient_channel[index].get('key', f'G{index + 1}')}={values[index]:.4g}%"
                        for index in np.flatnonzero(changed)
                    ]
                    gradient_events.append((float(time_value), ", ".join(parts)))
                previous = values

        event_times = np.asarray([event["time"] for event in ppg_events], dtype=float)
        for grad_time, description in gradient_events:
            index = int(np.searchsorted(event_times, grad_time, side="right") - 1)
            if index >= 0:
                ppg_events[index]["gradients"].append({"time": grad_time, "description": description})

        extent = float(max(raw_times[-1], max((time for time, _ in gradient_events), default=raw_times[-1])))
        excitations = self.get_trajectory_excitation_times(np.asarray([float(raw_times[0]), extent]))
        boundaries = np.concatenate(([float(raw_times[0])], excitations, [extent + 1e-15]))
        cycles: list[dict] = []
        for boundary_index in range(len(boundaries) - 1):
            start, end = float(boundaries[boundary_index]), float(boundaries[boundary_index + 1])
            events = [event for event in ppg_events if start <= float(event["time"]) < end]
            if not events:
                continue
            label = "Preparation" if boundary_index == 0 and excitations.size else f"Cycle {boundary_index if excitations.size else 1}"
            modules: list[dict] = []
            for event_index, event in enumerate(events):
                delay = max(
                    (float(events[event_index + 1]["time"]) if event_index + 1 < len(events) else end)
                    - float(event["time"]),
                    0.0,
                )
                event["delay"] = delay
                if not modules or modules[-1]["source"] != event["source"]:
                    modules.append({"source": event["source"], "events": []})
                modules[-1]["events"].append(event)
            cycles.append({"label": label, "start": start, "end": end, "modules": modules})
        return cycles

    def refresh_sequence_tree(self) -> None:
        tree = self.__dict__.get("sequenceTreeWidget")
        if tree is None:
            return
        tree.clear()
        for cycle in self.build_sequence_outline():
            cycle_item = QtWidgets.QTreeWidgetItem([
                str(cycle["label"]),
                f"{self.format_time(float(cycle['start']))} – {self.format_time(float(cycle['end']))}",
            ])
            cycle_item.setData(0, Qt.ItemDataRole.UserRole, float(cycle["start"]))
            tree.addTopLevelItem(cycle_item)
            for module in cycle["modules"]:
                events = module["events"]
                module_item = QtWidgets.QTreeWidgetItem([
                    str(module["source"]),
                    f"{self.format_time(float(events[0]['time']))} – {self.format_time(float(events[-1]['time']))}",
                ])
                module_item.setData(0, Qt.ItemDataRole.UserRole, float(events[0]["time"]))
                cycle_item.addChild(module_item)
                for event in events:
                    delay = float(event["delay"])
                    text = f"line {event['line']}"
                    event_item = QtWidgets.QTreeWidgetItem([
                        text,
                        f"{self.format_time(float(event['time']))}  Δ {self.format_time(delay)}",
                    ])
                    event_item.setData(0, Qt.ItemDataRole.UserRole, float(event["time"]))
                    event_item.setToolTip(0, f"Internal line {event['internal_line']}")
                    module_item.addChild(event_item)
                    gradients = event["gradients"]
                    if gradients:
                        examples = list(dict.fromkeys(str(item["description"]) for item in gradients))[:3]
                        suffix = " …" if len(gradients) > len(examples) else ""
                        gradient_item = QtWidgets.QTreeWidgetItem([
                            f"Gradient changes ({len(gradients)}): " + "; ".join(examples) + suffix,
                            (
                                self.format_time(float(gradients[0]["time"]))
                                if len(gradients) == 1 else
                                f"{self.format_time(float(gradients[0]['time']))} – "
                                f"{self.format_time(float(gradients[-1]['time']))}"
                            ),
                        ])
                        gradient_item.setData(0, Qt.ItemDataRole.UserRole, float(gradients[0]["time"]))
                        event_item.addChild(gradient_item)
        tree.resizeColumnToContents(0)

    def jump_to_sequence_tree_item(self, item: QtWidgets.QTreeWidgetItem, _column: int = 0) -> None:
        time_value = item.data(0, Qt.ItemDataRole.UserRole)
        if time_value is not None:
            self.jump_to_rf_pulse_time(float(time_value))

    def close_sequence_schematic(self) -> None:
        dialog = getattr(self, "_sequenceSchematicDialog", None)
        if dialog is not None:
            dialog.close()

    def show_sequence_schematic(self) -> None:
        dialog = getattr(self, "_sequenceSchematicDialog", None)
        if dialog is not None:
            if dialog.isMinimized():
                dialog.setWindowState(dialog.windowState() & ~Qt.WindowState.WindowMinimized)
            dialog.raise_()
            dialog.activateWindow()
            return

        outline = self.build_sequence_outline()
        structure = getattr(self, "pulseProgramStructure", {})
        if not outline and not structure:
            QtWidgets.QMessageBox.information(self, "Pulse Sequence", "No timed PPG data is loaded.")
            return
        dialog = QtWidgets.QDialog(None, Qt.WindowType.Window)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._sequenceSchematicDialog = dialog
        dialog.finished.connect(lambda: setattr(self, "_sequenceSchematicDialog", None))
        dialog.setWindowTitle("Pulse Sequence Schematic")
        dialog.resize(1200, 720)
        layout = QtWidgets.QVBoxLayout(dialog)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Section:"))
        section_combo = QtWidgets.QComboBox()
        if structure:
            section_combo.addItem("Full PPG — all loops", "structure")
        for index, cycle in enumerate(outline):
            section_combo.addItem(str(cycle["label"]), ("executed", index))
        controls.addWidget(section_combo)
        delay_scale_label = QtWidgets.QLabel("Idle delays")
        delay_scale = QtWidgets.QComboBox()
        delay_scale.addItem("Proportional", "proportional")
        delay_scale.addItem("Compressed", "compressed")
        delay_scale.addItem("Uniform", "uniform")
        delay_scale.setCurrentIndex(1)
        delay_scale_label.setBuddy(delay_scale)
        controls.addWidget(delay_scale_label)
        controls.addWidget(delay_scale)
        controls.addStretch(1)
        navigation = QtWidgets.QLabel()
        controls.addWidget(navigation)
        layout.addLayout(controls)
        channel_controls = QtWidgets.QWidget()
        channel_layout = QtWidgets.QHBoxLayout(channel_controls)
        channel_layout.setContentsMargins(0, 0, 0, 0)
        channel_layout.addWidget(QtWidgets.QLabel("Channels"))
        channel_boxes = {name: QtWidgets.QCheckBox(name) for name in _PPG_CHANNELS}
        for checkbox in channel_boxes.values():
            checkbox.setChecked(True)
            channel_layout.addWidget(checkbox)
        combine_box = QtWidgets.QCheckBox("Combine gradients")
        channel_layout.addWidget(combine_box)
        channel_layout.addStretch()
        layout.addWidget(channel_controls)
        view_box = SchematicViewBox()
        schematic = pg.PlotWidget(background="#f3f3f3", viewBox=view_box)
        schematic.showGrid(x=True, y=False, alpha=0.15)
        schematic.setLabel("bottom", "Time", units="s")
        schematic.getPlotItem().hideButtons()
        schematic_scroll = QtWidgets.QScrollArea()
        schematic_scroll.setWidgetResizable(True)
        schematic_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        schematic_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        schematic_scroll.setWidget(schematic)
        view_box.resetPerformed.connect(lambda: schematic_scroll.verticalScrollBar().setValue(0))
        layout.addWidget(schematic_scroll, stretch=1)
        delay_key = QtWidgets.QLabel()
        delay_key.setWordWrap(True)
        delay_key.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(delay_key)
        details = QtWidgets.QLabel()
        details.setWordWrap(True)
        layout.addWidget(details)

        cursor_time = getattr(self, "currentCursorTime", None)
        initial_index = 0
        if cursor_time is not None and not structure:
            for index, cycle in enumerate(outline):
                if float(cycle["start"]) <= float(cursor_time) <= float(cycle["end"]):
                    initial_index = index
                    break

        def render(combo_index: int, *, keep_x: bool = False) -> None:
            selection = section_combo.itemData(combo_index)
            full_ppg = selection == "structure"
            x_range = view_box.viewRange()[0] if keep_x else None
            view_box.separate_zoom = full_ppg
            view_box.fit_range = None
            delay_scale_label.setVisible(full_ppg)
            delay_scale.setVisible(full_ppg)
            channel_controls.setVisible(full_ppg)
            navigation.setText(
                ("Wheel: horizontal   Shift+wheel: vertical" if full_ppg else "Wheel: zoom")
                + "   Drag: pan   Double-click: reset"
            )
            if full_ppg:
                lengths = self.render_full_ppg_schematic(
                    schematic, delay_scale=delay_scale.currentData(),
                    shown_channels={name for name, box in channel_boxes.items() if box.isChecked()},
                    combine_gradients=combine_box.isChecked(),
                )
                delay_key.setText("    ".join(
                    f"{token} = {self.format_ppg_duration(duration)}" for token, duration in lengths.items()
                ))
                delay_key.setVisible(bool(lengths))
                details.setText(
                    "Brackets show the precompiled loop structure and counts. EPI acquisition trains show "
                    "their repeated gradient and ADC waveforms. Dashed amber traces indicate placeholders. "
                    "RF phase uses the first programmed phase-cycle entry when available. "
                    "Scroll down for more timing and frequency labels."
                )
            else:
                delay_key.hide()
                cycle = outline[int(selection[1])]
                self.render_sequence_schematic(schematic, cycle)
                details.setText(
                    f"Executed {cycle['label']}: {self.format_time(float(cycle['start']))} – "
                    f"{self.format_time(float(cycle['end']))}. Exact RF and gradient shapes from the simulated XML timeline."
                )
            view_box.fit_range = view_box.viewRange()
            if x_range is not None:
                view_box.setXRange(*x_range, padding=0)
            schematic_scroll.verticalScrollBar().setValue(0)

        section_combo.currentIndexChanged.connect(render)
        delay_scale.currentIndexChanged.connect(lambda: render(section_combo.currentIndex()))
        for box in (*channel_boxes.values(), combine_box):
            box.toggled.connect(lambda: render(section_combo.currentIndex(), keep_x=True))
        section_combo.setCurrentIndex(initial_index)
        render(initial_index)
        close_buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        close_buttons.rejected.connect(dialog.reject)
        layout.addWidget(close_buttons)
        dialog.show()

    def build_full_ppg_schematic_events(self) -> list[dict]:
        structure = getattr(self, "pulseProgramStructure", {})
        # pulseprogram.precomp is already in compiler/execution order. Do not
        # sort it by source line: included module bodies deliberately jump
        # between files and may revisit the same source location.
        records = list(structure.get("records", []))
        active_records = self.select_active_ppg_records(
            records, {**structure.get("parameters", {}), **structure.get("loop_values", {})},
            structure.get("compiled_lines", ()),
        )
        loop_aliases = {}
        delay_aliases = {}
        delay_names = {"de", *(name.casefold() for name in structure.get("delay_names", []))}
        for record in records:
            code = str(record.get("text", "")).partition(";")[0].strip()
            alias = re.match(
                r"define\s+loopcounter\s+(\w+)\s*=\s*\{\s*\$?(\w+)\s*\}",
                code, re.IGNORECASE,
            )
            if alias:
                loop_aliases[alias.group(1)] = alias.group(2)
            declaration = re.fullmatch(r"define\s+delay\s+(\w+)(?:\s*=\s*(.+))?", code, re.IGNORECASE)
            if declaration:
                name, expression = declaration.groups()
                delay_names.add(name.casefold())
                if expression and "$" not in expression:
                    delay_aliases[name.casefold()] = expression.strip("{} ")
            delay_alias = re.match(
                r'"?\s*([A-Za-z_]\w*)\s*=\s*([^";]+)"?$',
                code,
            )
            if delay_alias:
                name, expression = delay_alias.groups()
                if not re.fullmatch(r"ph\d+", name, re.IGNORECASE):
                    delay_aliases[name.casefold()] = expression.strip()
        loops = collect_ppg_loops(active_records, structure, loop_aliases)
        frequency_offsets = ppg_frequency_offsets(active_records, loops, structure.get("frequency_lists_hz", {}))

        def is_delay(token: str | None) -> bool:
            if token is None:
                return False
            name = token.partition("[")[0].casefold()
            return bool(_PPG_TIME.fullmatch(name) or re.fullmatch(r"d\d+", name)
                        or name in delay_names or name in delay_aliases)

        def loop_event(index: int, *, expanded: bool = False) -> dict:
            loop = loops[index]
            record = active_records[loop["source_record_index"]]
            text = str(record["text"])
            token = self.get_ppg_duration_token(text)
            duration = self.resolve_ppg_duration(token, delay_aliases) if token else None
            return {
                **loop, "kind": "loop", "record_index": index, "expanded": expanded,
                "line": record["line"], "source": record.get("source", ""), "text": text,
                "duration": duration,
                "duration_token": token if is_delay(token) else None,
            }

        events: list[dict] = []
        module_stack: list[str] = []
        gradient_order = 0
        epi_readouts: dict[int, tuple[int, dict | None]] = {}
        for index, record in enumerate(active_records):
            if Path(str(record.get("source", ""))).name.lower() != "epi.mod":
                continue
            adc_job = re.search(r"\badc_job(\d+)\b", str(record["text"]).partition(";")[0], re.IGNORECASE)
            if adc_job:
                readout = self.build_ppg_epi_readout(active_records, index, adc_job.group(1), delay_aliases, loop_aliases)
                if readout is not None:
                    epi_readouts[index] = readout
        # Without the original method, both ramp branches may be selected.
        # A compiled waveform or a complete XML readout identifies the selected branch.
        use_resolved_epi_branch = (
            "PVM_EpiRampMode" not in structure.get("parameters", {})
            and any(event is not None for _, event in epi_readouts.values())
        )
        epi_readout_end = -1
        for record_index, record in enumerate(active_records):
            if record_index <= epi_readout_end:
                continue
            if record_index in epi_readouts:
                end_index, readout = epi_readouts[record_index]
                if readout is not None or use_resolved_epi_branch:
                    epi_readout_end = end_index
                    if readout is not None:
                        readout.update(record_index=record_index, end_record_index=end_index)
                        events.append(readout)
                        events.extend(
                            loop_event(index, expanded=True)
                            for index in range(record_index + 1, end_index + 1) if index in loops
                        )
                    continue
            if record_index in loops:
                events.append(loop_event(record_index))
                continue
            text = str(record.get("text", "")).strip()
            lowered = text.lower()
            exec_begin = re.match(r";\s*exec_begin\s+([A-Za-z_]\w*)", text, re.IGNORECASE)
            exec_end = re.match(r";\s*exec_end\s+([A-Za-z_]\w*)", text, re.IGNORECASE)
            if exec_begin:
                module_stack.append(exec_begin.group(1))
                continue
            if exec_end:
                if module_stack:
                    module_stack.pop()
                continue
            if not text or text.startswith(";") or lowered.startswith("define ") or text in {"{", "}"}:
                continue
            if lowered.startswith(("if", "else", '"')) or re.match(r"\w+\s*=", text):
                continue
            label_match = re.match(r"([A-Za-z_]\w*)\s*,", text)
            label = label_match.group(1) if label_match else None
            duration_token = self.get_ppg_duration_token(text)
            subroutine_match = re.search(r"\bsubr\s+(\w+)", text, re.IGNORECASE)
            if re.search(r"\b(?:grad_(?:ramp|shape|off)|groff)\b", lowered):
                kind = "gradient"
            elif re.search(r"\((?:\s*p\d+\b[^)]*|[^)]*:[^)]*)\)\s*:f\d", text, re.IGNORECASE):
                kind = "rf"
            elif re.search(r"\b(?:aq|adc|dwgen)_[A-Za-z0-9_]+", text, re.IGNORECASE):
                kind = "acquisition"
            elif subroutine_match:
                # pulseprogram.precomp expands the called body immediately
                # after this line. The call itself has no duration or physical
                # waveform and would make disabled modules look active.
                continue
            elif is_delay(duration_token):
                kind = "delay"
            elif record_index in frequency_offsets:
                kind = "frequency"
            elif re.search(r"\bgoto\b", lowered):
                kind = "flow"
            else:
                continue
            event = {
                "kind": kind, "line": int(record["line"]), "text": text, "label": label,
                "source": str(record.get("source", "")),
                "module": module_stack[-1] if module_stack else None,
                "record_index": record_index,
                "internal_line": record.get("internal_line"),
            }
            if kind == "rf":
                pulse_token = re.search(r"\(\s*([A-Za-z_]\w*)\b", text)
                duration_token = pulse_token.group(1) if pulse_token else duration_token
                rf_instruction = re.search(r"\(([^()]*)\)\s*:f(\d+)\b", text, re.IGNORECASE)
                if rf_instruction:
                    body, channel = rf_instruction.groups()
                    shape = re.match(r"\s*\w+\s*:\s*(\w+)", body)
                    phase = re.search(r"\b(ph\d+)\b", body, re.IGNORECASE)
                    event.update(
                        rf_shape=shape[1].lower() if shape else None, rf_channel=int(channel),
                        rf_phase=phase[1].lower() if phase else None,
                    )
            elif kind == "acquisition" and not duration_token:
                explicit_delays = re.search(r"\)\s*\(([^)]*)\)", text)
                if explicit_delays:
                    values = re.findall(r"\d+(?:\.\d+)?[mun]", explicit_delays.group(1))
                    if values:
                        event["duration"] = sum(
                            self.resolve_ppg_duration(value) or 0.0 for value in values
                        )
            if kind == "acquisition" and text.lower().startswith(("aq_", "aqjob")):
                parameters = structure.get("parameters", {})
                acquisition_ms = parameters.get("PVM_SpecAcquisitionTime", parameters.get("PVM_AcquisitionTime"))
                if isinstance(acquisition_ms, (int, float)):
                    event["duration"] = float(acquisition_ms) * 1e-3
            if duration_token:
                event["duration_token"] = duration_token
                resolved_duration = self.resolve_ppg_duration(duration_token, delay_aliases)
                if resolved_duration is not None:
                    event["duration"] = resolved_duration
            if kind == "gradient":
                event["gradient_order"] = gradient_order
                gradient_order += 1
            if record_index in frequency_offsets:
                event["frequency_offsets"] = frequency_offsets[record_index]
            events.append(event)
        point_targets = structure.get("ramped_gradient_targets", ())
        point_events = [
            event for event in events if event["kind"] == "gradient"
            and re.search(r"\bgrad_(?:ramp|shape|off)\b", event["text"].partition(";")[0], re.IGNORECASE)
        ]
        # Gcon point rows follow compiled gradient instruction order. Hardware
        # groff does not consume a point. Require a complete match before using it.
        if point_targets and len(point_events) == len(point_targets):
            for event, targets in zip(point_events, point_targets, strict=True):
                if targets is not None:
                    event["compiled_ramped_target"] = targets
        return self.annotate_ppg_gradient_variants(events, active_records, structure.get("parameters", {}))

    def build_ppg_epi_readout(
        self, records: list[dict], start_index: int, job: str,
        delay_aliases: dict[str, str], loop_aliases: dict[str, str],
    ) -> tuple[int, dict | None] | None:
        """Use the compiled EPI block, with a complete XML occurrence as a fallback."""
        start_record = records[start_index]
        source = str(start_record["source"])
        count_token = None
        for end_index in range(start_index + 1, len(records)):
            record = records[end_index]
            code = str(record["text"]).partition(";")[0].strip()
            if record.get("source") != source or re.search(r"\badc_job\d+\b", code, re.IGNORECASE):
                return None
            loop = re.fullmatch(r"loop\s+(\w+)", code, re.IGNORECASE)
            if loop:
                count_token = loop.group(1)
            if re.search(rf"\beoscnpzbwrif_job{job}\b", code, re.IGNORECASE):
                break
        else:
            return None

        structure = self.pulseProgramStructure
        event = {
            "kind": "epi_readout", "source": source, "line": start_record["line"],
            "text": start_record["text"],
            "count": resolve_loop_count(count_token, structure, loop_aliases) if count_token else None,
        }
        compiled = build_compiled_epi_waveform(structure, records[start_index:end_index + 1])
        if compiled is not None:
            return end_index, {**event, **compiled, "start": 0.0}

        timeline = getattr(self, "pulseProgramTimeline", None)
        if timeline is None or len(timeline) != 2:
            return end_index, None
        times, internal_lines = map(np.asarray, timeline)
        if times.ndim != 1 or times.shape != internal_lines.shape:
            return end_index, None
        mapping = getattr(self, "pulseProgramLineMapping", {})
        source_lines = {
            line: item["line"] for line, item in mapping.items()
            if Path(str(item.get("source", ""))).name == Path(source).name and "line" in item
        }
        start_lines = [line for line, source_line in source_lines.items() if source_line == start_record["line"]]
        end_lines = [line for line, source_line in source_lines.items() if source_line == record["line"]]
        starts = np.sort(times[np.isin(internal_lines, start_lines)])
        ends = np.unique(times[np.isin(internal_lines, end_lines)])
        if not starts.size or not ends.size:
            return end_index, None
        delay_token = self.get_ppg_duration_token(str(record["text"]))
        end_delay = self.resolve_ppg_duration(delay_token, delay_aliases) if delay_token else None
        windows = [
            (float(window["start"]), float(window["end"]))
            for window in self.get_acquisition_window_details()
        ]
        previous_end = -np.inf
        for end_time in ends:
            occurrence_index = starts.searchsorted(previous_end, side="right")
            previous_end = end_time
            if occurrence_index == starts.size or starts[occurrence_index] >= end_time:
                continue
            start = float(starts[occurrence_index])
            end = float(end_time) + (end_delay or 0.0)
            readout_windows = [(left, right) for left, right in windows if start <= left < right <= end]
            if readout_windows:
                return end_index, {
                    **event, "start": start, "duration": end - start, "acquisition_windows": readout_windows,
                }
        return end_index, None

    @staticmethod
    def representative_gradient_indices(values: Collection[list[float] | np.ndarray], count: int) -> list[int]:
        candidates = [0, count - 1]
        for array in values:
            samples = array[:count]
            candidates.extend((int(np.argmin(samples)), int(np.argmax(samples))))
        candidates.extend(np.linspace(0, count - 1, _PPG_MAX_GRADIENT_VARIANTS, dtype=int).tolist())
        return sorted(list(dict.fromkeys(candidates))[:_PPG_MAX_GRADIENT_VARIANTS])

    @classmethod
    def annotate_ppg_gradient_variants(cls, events: list[dict], records: list[dict], parameters: dict) -> list[dict]:
        """Select iterations for gradient lists advanced by a compiled loop."""
        for loop in reversed(events):
            count = loop.get("count")
            start = loop.get("start_record_index")
            if loop["kind"] != "loop" or not isinstance(count, int) or count <= 1 or start is None:
                continue
            stop = loop["record_index"]
            advanced = {
                name for record in records[start:stop]
                for name in re.findall(r"\b([A-Za-z_]\w*)\.inc\b", record["text"].partition(";")[0])
            }
            gradients = [
                (event, set(re.findall(r"\b[A-Za-z_]\w*\b", event["text"].partition(";")[0])))
                for event in events if event["kind"] == "gradient" and start <= event["record_index"] < stop
            ]
            names = {
                name for _, symbols in gradients for name in symbols
                if name in advanced or (loop["target"] == "echo" and name.startswith("ACQ_spatial_phase_"))
            }
            lists = {
                name: values for name in sorted(names)
                if isinstance(values := parameters.get(name), list) and len(values) > 1
            }
            if not lists:
                continue
            sample_count = min(count, *(len(values) for values in lists.values()))
            indices = cls.representative_gradient_indices(lists.values(), sample_count)
            for event, symbols in gradients:
                indexed_lists = symbols & lists.keys()
                if indexed_lists:
                    event["phase_encode_indices"] = indices
                    event["indexed_gradient_lists"] = indexed_lists
        return events

    def evaluate_ppg_condition(self, expression: str, parameters: dict) -> bool | None:
        expression = expression.strip()
        unary = re.fullmatch(r"!?\s*([A-Za-z_]\w*)", expression)
        if unary and unary.group(1) in parameters:
            value = parameters[unary.group(1)]
            enabled = str(value).strip("<> ").lower() not in {
                "", "0", "0.0", "off", "no", "false", "none",
            }
            return not enabled if expression.lstrip().startswith("!") else enabled
        match = re.search(r"([A-Za-z_]\w*)\s*(==|!=|>=|<=|>|<)\s*([A-Za-z_]\w*|[-+]?\d+(?:\.\d+)?)", expression)
        if not match:
            return None
        left_name, operator, right_token = match.groups()
        if left_name not in parameters:
            return None
        left = parameters[left_name]
        right = parameters.get(right_token, right_token)
        enum_values = {
            "PVM_DiffPrepMode": {
                "spinecho": 0.0,
                "stimulatedecho": 1.0,
                "double spinecho": 2.0,
                "doublespinecho": 2.0,
            },
            "PVM_WsMode": {"no suppression": 0.0, "nosuppression": 0.0, "chess": 1.0, "vapor": 2.0},
        }
        if left_name in enum_values and isinstance(left, str):
            normalized_enum = re.sub(r"[_-]+", " ", left.strip("<> ").lower())
            left = enum_values[left_name].get(normalized_enum, enum_values[left_name].get(normalized_enum.replace(" ", ""), left))
        try:
            left, right = float(left), float(right)
        except (TypeError, ValueError):
            left, right = str(left).strip("<>"), str(right).strip("<>")
        return {
            "==": left == right, "!=": left != right, ">": left > right,
            "<": left < right, ">=": left >= right, "<=": left <= right,
        }[operator]

    @staticmethod
    def compiled_ppg_branches(statements: list[dict], compiled_lines: Collection[int]) -> dict[str, bool]:
        """Use emitted instruction references to resolve compile-time branches."""
        if not compiled_lines:
            return {}
        code = [str(record["text"]).partition(";")[0].strip() for record in statements]
        meaningful = [index for index, text in enumerate(code) if text]
        emitted = [index for index, record in enumerate(statements) if record.get("internal_line") in compiled_lines]
        blocks = {}
        openings = []
        for index in meaningful:
            if code[index] == "{":
                openings.append(index)
            elif code[index] == "}" and openings:
                blocks[openings.pop()] = index

        def has_instructions(start: int, stop: int) -> bool:
            index = bisect_left(emitted, start + 1)
            return index < len(emitted) and emitted[index] < stop

        choices: dict[str, set[bool]] = {}
        for index, following in zip(meaningful, meaningful[1:]):
            match = re.fullmatch(r"if\s*\((.*)\)", code[index], re.IGNORECASE)
            if not match or following not in blocks:
                continue
            end = blocks[following]
            alternate = bisect_left(meaningful, end + 1)
            else_start = meaningful[alternate + 1] if alternate + 1 < len(meaningful) else -1
            then_compiled = has_instructions(following, end)
            else_compiled = (
                else_start in blocks and code[meaningful[alternate]].lower() == "else"
                and has_instructions(else_start, blocks[else_start])
            )
            if then_compiled != else_compiled:
                key = re.sub(r"\s+", "", match[1])
                choices.setdefault(key, set()).add(then_compiled)
        return {key: next(iter(values)) for key, values in choices.items() if len(values) == 1}

    def select_active_ppg_records(
        self, records: list[dict], parameters: dict, compiled_lines: Collection[int] = (),
    ) -> list[dict]:
        """Select one basic if/else path using resolved method parameters."""
        selected: list[dict] = []
        active_stack = [{"active": True, "condition": True, "parent": True}]
        pending_condition: bool | None = None
        last_closed: dict | None = None
        statements = []
        for record in records:
            code = str(record["text"]).partition(";")[0].strip()
            combined = re.fullmatch(r"(\})\s*(else)\s*(\{)?", code, re.IGNORECASE)
            opener = re.fullmatch(r'(if\s*(?:\(.*\)|"[^"]+")|else|loop\s+\w+)\s*(\{)', code, re.IGNORECASE)
            parts = (combined or opener).groups() if combined or opener else ()
            if parts:
                statements.extend({**record, "text": part} for part in parts if part)
            else:
                statements.append(record)
        compiled_branches = self.compiled_ppg_branches(statements, compiled_lines)
        runtime_parameters = dict(parameters)
        for record in statements:
            text = str(record.get("text", "")).partition(";")[0].strip()
            if_match = re.match(r"if\s*\((.*)\)\s*$", text, re.IGNORECASE)
            if if_match:
                expression = if_match[1]
                evaluated = compiled_branches.get(re.sub(r"\s+", "", expression))
                if evaluated is None:
                    evaluated = self.evaluate_ppg_condition(expression, runtime_parameters)
                # Unknown runtime conditions use the primary branch only;
                # importantly, they never include both if and else bodies.
                pending_condition = True if evaluated is None else evaluated
                continue
            quoted = re.fullmatch(r'if\s+"([^"]+)"', text, re.IGNORECASE)
            if quoted:
                evaluated = self.evaluate_ppg_condition(quoted[1], runtime_parameters)
                pending_condition = True if evaluated is None else evaluated
            if re.match(r"else\b", text, re.IGNORECASE):
                pending_condition = not bool(last_closed and last_closed["condition"])
                continue
            if text == "{":
                condition = True if pending_condition is None else pending_condition
                parent_active = bool(active_stack[-1]["active"])
                active_stack.append({
                    "active": parent_active and condition,
                    "condition": condition,
                    "parent": parent_active,
                })
                pending_condition = None
                last_closed = None
                if active_stack[-1]["active"]:
                    selected.append(record)
                continue
            if text == "}":
                if len(active_stack) > 1:
                    last_closed = active_stack.pop()
                    if last_closed["active"]:
                        selected.append(record)
                continue
            if active_stack[-1]["active"]:
                assignment = re.fullmatch(r'"(\w+)\s*=\s*([+-]?\d+|\w+)"', text)
                if assignment:
                    name, value = assignment.groups()
                    resolved = runtime_parameters.get(value, value)
                    try:
                        runtime_parameters[name] = float(resolved)
                    except (ValueError, TypeError):
                        runtime_parameters.pop(name, None)
                selected.append(record)
        return selected

    @staticmethod
    def format_ppg_duration(duration: float | None) -> str:
        if duration is None:
            return "unresolved"
        if duration == 0 or duration >= 1:
            return f"{duration:.6g} s"
        for factor, unit in ((1e-3, "ms"), (1e-6, "µs"), (1e-9, "ns")):
            if duration >= factor or unit == "ns":
                return f"{duration / factor:.6g} {unit}"

    @staticmethod
    def format_ppg_frequency(lower: float, upper: float) -> str:
        scale, unit = (1e3, "kHz") if max(abs(lower), abs(upper)) >= 1e3 else (1, "Hz")
        first = f"{lower / scale:+.6g}" if lower else "0"
        last = f"{upper / scale:+.6g}" if upper else "0"
        value = first if lower == upper else f"{first} to {last}"
        return f"{value} {unit}".replace("-", "−")

    @staticmethod
    def configure_schematic_lanes(plot: pg.PlotWidget, ticks: list[tuple[float, str]] | None = None) -> None:
        ticks = _SCHEMATIC_TICKS if ticks is None else ticks
        axis = plot.getPlotItem().getAxis("left")
        metrics = QtGui.QFontMetrics(plot.font())
        axis.setWidth(max((metrics.horizontalAdvance(label) for _, label in ticks), default=0) + 14)
        axis.setTicks([sorted(ticks)])

    @staticmethod
    def ppg_channel_layout(shown: Collection[str], combine_gradients: bool) -> tuple[dict[str, float], list[tuple[float, str]]]:
        lanes: dict[str, float] = {"ADC": 0.0} if "ADC" in shown else {}
        gradients = [axis for axis in _PPG_GRADIENTS if axis in shown]
        position = 1.0 if lanes else 0.5
        if combine_gradients and gradients:
            lanes.update(dict.fromkeys(gradients, position))
            position += 1.0
        else:
            for axis in reversed(gradients):
                lanes[axis] = position
                position += 1.0
        for channel in ("RF phase", "RF"):
            if channel in shown:
                lanes[channel] = position
                position += 1.0
        ticks = [(value, label) for label, value in lanes.items()]
        if combine_gradients and len(gradients) > 1:
            ticks = [(value, name) for value, name in ticks if name not in gradients]
            ticks.append((lanes[gradients[0]], "/".join(gradients)))
        if "RF phase" in lanes:
            phase = lanes["RF phase"]
            ticks.remove((phase, "RF phase"))
            ticks.extend([(phase, "RF phase 0°"), (phase + _RF_PHASE_HEIGHT / 2, "180°"),
                          (phase + _RF_PHASE_HEIGHT, "360°")])
        return lanes, ticks

    @staticmethod
    def clear_schematic(plot: pg.PlotWidget) -> None:
        layout = getattr(plot, "_ppg_timing_layout", None)
        if layout is not None:
            view = plot.getViewBox()
            view.sigXRangeChanged.disconnect(layout)
            view.sigResized.disconnect(layout)
            del plot._ppg_timing_layout
            plot.setMinimumHeight(0)
        plot.clear()

    @staticmethod
    def layout_ppg_timing_labels(
        plot: pg.PlotWidget, labels: list[tuple[float, float, pg.TextItem]],
    ) -> None:
        view = plot.getViewBox()
        # Place wider brackets first so long durations stay readable in the overview.
        labels = sorted(labels, key=lambda label: label[1] - label[0], reverse=True)
        bounds = [item.boundingRect() for _, _, item in labels]
        row_pixels = max(rect.height() for rect in bounds) + 4
        fitted_y = view.viewRange()[1]
        previous_geometry = None

        def layout(*_args) -> None:
            nonlocal fitted_y, previous_geometry
            view_left, view_right = view.viewRange()[0]
            view_width = view.width()
            geometry = (view_left, view_right, view_width)
            if geometry == previous_geometry or view_width <= 0:
                return
            previous_geometry = geometry
            pixels_per_unit = view_width / (view_right - view_left)
            rows: list[list[tuple[float, float]]] = []
            for (left, right, item), rect in zip(labels, bounds, strict=True):
                visible_left = max(left, view_left)
                visible_right = min(right, view_right)
                if visible_right <= visible_left:
                    item.hide()
                    continue
                center = (visible_left + visible_right) / 2
                pixel = (center - view_left) * pixels_per_unit
                width = rect.width()
                label_left = min(max(pixel - width / 2, 2.0), max(view_width - width - 2.0, 2.0))
                label_right = label_left + width
                for row, spans in enumerate(rows):
                    index = bisect_left(spans, (label_left, label_right))
                    if index and spans[index - 1][1] + 4 > label_left:
                        continue
                    if index < len(spans) and label_right + 4 > spans[index][0]:
                        continue
                    spans.insert(index, (label_left, label_right))
                    break
                else:
                    row = len(rows)
                    rows.append([(label_left, label_right)])
                item.setPos(view_left + label_left / pixels_per_unit, -0.58 - _PPG_TIMING_ROW_STEP * row)
                item.show()

            # Extend the canvas for dense stacks so the waveform lanes keep their height.
            was_fitted = view.viewRange()[1] == fitted_y
            fitted_y = [min(-1.1, -0.65 - _PPG_TIMING_ROW_STEP * len(rows)), fitted_y[1]]
            if was_fitted:
                view.setYRange(*fitted_y, padding=0)
            if isinstance(view, SchematicViewBox) and view.fit_range is not None:
                view.fit_range = (view.fit_range[0], fitted_y)
            axis_height = plot.getPlotItem().getAxis("bottom").height()
            min_height = (fitted_y[1] - fitted_y[0]) * row_pixels / _PPG_TIMING_ROW_STEP + axis_height + 10
            plot.setMinimumHeight(int(np.ceil(min_height)) if rows else 0)

        plot._ppg_timing_layout = layout
        view.sigXRangeChanged.connect(layout)
        view.sigResized.connect(layout)
        layout()

    def render_full_ppg_schematic(
        self, plot: pg.PlotWidget, *, delay_scale: str = "compressed",
        shown_channels: Collection[str] | None = None, combine_gradients: bool = False,
    ) -> dict[str, float | None]:
        self.clear_schematic(plot)
        events = self.build_full_ppg_schematic_events()
        delay_lengths = {}
        structure = self.pulseProgramStructure
        recorded_candidates = []
        for event in events:
            if event["kind"] != "rf":
                continue
            event["compiled_rf"] = compiled_rf_shape(event, structure)
            if event["compiled_rf"] is not None:
                shape = event["compiled_rf"][1]
                if not event.get("duration"):
                    event["duration"] = shape["duration"]
                if offset := shape.get("offset_hz"):
                    frequency = ("Shape", f"f{event['rf_channel']}", event["rf_shape"], offset, offset)
                    event.setdefault("frequency_offsets", []).append(frequency)
            if event["compiled_rf"] is None or event["compiled_rf"][1]["phase"] is None:
                recorded_candidates.append(event)
        recorded_rf = recorded_rf_envelopes(
            recorded_candidates, self.__dict__.get("pulseProgramTimeline"),
            schematic_rf_channels(getattr(self, "channels", [])) if recorded_candidates else {},
        )
        lanes, ticks = self.ppg_channel_layout(
            _PPG_CHANNELS if shown_channels is None else shown_channels, combine_gradients,
        )
        self.configure_schematic_lanes(plot, ticks)
        highest_lane = max(lanes.values(), default=0.0)
        plot.getPlotItem().getAxis("bottom").setTicks([])
        plot.setLabel("bottom", f"Pulse-program order · {delay_scale} idle delays")
        widths = {
            "rf": 1.0, "gradient": 1.0, "acquisition": 1.0, "epi_readout": 1.0,
            "subroutine": 0.0, "delay": 0.15, "flow": 0.0, "loop": 0.0, "frequency": 0.0,
        }
        x = 0.0
        positioned: list[tuple[dict, float, float]] = []
        physical_durations = []
        for event in events:
            kind = event["kind"]
            token = event.get("duration_token")
            duration = event.get("duration")
            event_width = widths[kind]
            idle_delay = kind == "delay" or (kind == "loop" and bool(token))
            if token and kind != "rf" and not _PPG_TIME.fullmatch(token):
                if kind == "delay" or duration is not None:
                    delay_lengths.setdefault(token.casefold(), duration)
            if isinstance(duration, (int, float)) and duration > 0:
                linear_width = duration / _PPG_SECONDS_PER_UNIT
                if idle_delay and delay_scale == "uniform":
                    event_width = 8.0
                elif idle_delay and delay_scale == "compressed":
                    event_width = max(0.08, 8.0 * linear_width / (8.0 + linear_width))
                else:
                    event_width = linear_width
            elif kind == "delay" and duration == 0 and token and _PPG_TIME.fullmatch(token):
                event_width = 0.0
            physical_durations.append(duration if duration and duration > 0 else event_width * _PPG_SECONDS_PER_UNIT)
            positioned.append((event, x, x + event_width))
            x += event_width
        # Gradient commands run asynchronously across subsequent PPG instructions.
        # Map their physical sample times through the same delay scale as the program.
        display_edges = [0.0, *(right for _, _, right in positioned)]
        physical_edges = np.cumsum([0.0, *physical_durations])
        next_gradient = np.empty(len(positioned), dtype=int)
        following_gradient = len(positioned)
        for index in range(len(positioned) - 1, -1, -1):
            next_gradient[index] = following_gradient
            if positioned[index][0]["kind"] in {"gradient", "epi_readout"}:
                following_gradient = index
        total_width = max(x, 1.0)
        plot.setXRange(0.0, total_width, padding=0.015)
        for value in sorted(set(lanes.values())):
            plot.plot([0.0, total_width], [value, value], pen=pg.mkPen("#777777"))

        loop_row_ends: list[float] = []
        timing_labels: list[tuple[float, float, pg.TextItem]] = []
        label_metrics = QtGui.QFontMetrics(plot.font())
        units_per_pixel = total_width / max(1000, plot.width() - 60)
        rf_label_end = -np.inf

        def label_position(caption: str, center: float) -> tuple[float, float, float]:
            width = (label_metrics.horizontalAdvance(caption) + 12) * units_per_pixel
            left = min(max(center - width / 2, 0.0), total_width - width)
            return left, left + width, (center - left) / width

        gradient_colors = _PPG_GRADIENT_COLORS
        gradient_states: list[list[float | None]] = [[0.0], [0.0], [0.0]]
        gradient_ramp_ends = [0.0] * 3
        placeholder_pen = pg.mkPen(_PPG_PLACEHOLDER_COLOR, width=1.6, style=Qt.PenStyle.DashLine)
        unresolved_height = 0.12
        for index, (event, left, right) in enumerate(positioned):
            center = (left + right) * 0.5
            kind, text = str(event["kind"]), str(event["text"])
            token = event.get("duration_token")
            duration = event.get("duration")
            gradient_timing = kind == "gradient" and duration is not None
            if right > left and (gradient_timing or kind == "delay" or (kind == "loop" and token)):
                length = self.format_ppg_duration(duration)
                timing_name = "gradient" if gradient_timing else "delay"
                color = "#1d4ed8" if gradient_timing else "#4b5563"
                tooltip = f"PPG {timing_name}: {token}\nDuration: {length}\nSource: {event['source']}:{event['line']}\n{text}"
                timing_curve = plot.plot([left, left, right, right], [-0.35, -0.48, -0.48, -0.35], pen=pg.mkPen(color))
                timing_curve.setToolTip(tooltip)
                self.add_schematic_hover_region(plot, left, right, -0.82, -0.25, tooltip)
                named_duration = token and not _PPG_TIME.fullmatch(token)
                caption = f"{token} · {length}" if named_duration else length
                item = pg.TextItem(caption, color=color, anchor=(0.0, 0.0))
                item.setToolTip(tooltip)
                plot.addItem(item, ignoreBounds=True)
                timing_labels.append((left, right, item))
            for role, channel, symbol, lower, upper in event.get("frequency_offsets", ()):
                caption = f"{role} {channel} · {self.format_ppg_frequency(lower, upper)}"
                tooltip = f"{caption}\nCompiled frequency offset · {symbol}\nSource: {event['source']}:{event['line']}"
                color = "#0e7490"
                marker = plot.plot([left, left], [-0.12, -0.29], pen=pg.mkPen(color, width=2))
                marker.setToolTip(tooltip)
                item = pg.TextItem(caption, color=color, anchor=(0, 0))
                item.setToolTip(tooltip)
                plot.addItem(item, ignoreBounds=True)
                timing_labels.append((left, max(right, left + total_width * 1e-6), item))
            if kind not in {"gradient", "epi_readout"}:
                for axis_index, states in enumerate(gradient_states):
                    lane = lanes.get(_PPG_GRADIENTS[axis_index])
                    hold_start = max(left, gradient_ramp_ends[axis_index])
                    if lane is None or hold_start >= right:
                        continue
                    for state in dict.fromkeys(states):
                        if state is not None and abs(state) <= 1e-12:
                            continue
                        height = unresolved_height if state is None else 0.42 * state
                        hold = plot.plot(
                            [hold_start, right], [lane + height] * 2,
                            pen=placeholder_pen if state is None else pg.mkPen(gradient_colors[axis_index], width=1.1),
                        )
                        if state is None:
                            hold.setToolTip("Gradient hold: amplitude unresolved. Dashed trace is a placeholder.")
            if kind == "rf":
                caption = token or f"RF line {event['line']}"
                rf_color = self.render_ppg_rf_pulse(
                    plot, event, left, right, caption, recorded_rf.get(event.get("internal_line")), placeholder_pen,
                    rf_lane=lanes.get("RF"), phase_lane=lanes.get("RF phase"),
                )
                label = f"{caption} · {self.format_ppg_duration(duration)}"
                label_left, label_right, anchor = label_position(label, center)
                if "RF" in lanes and label_left > rf_label_end:
                    item = pg.TextItem(label, color=rf_color, anchor=(anchor, 1.0))
                    item.setPos(center, lanes["RF"] + .78)
                    plot.addItem(item)
                    rf_label_end = label_right
            elif kind == "gradient":
                profiles = self.pulseProgramStructure.get("recorded_gradient_profiles", {})
                profile = profiles.get(event.get("internal_line"))
                if (profile is not None and not event.get("phase_encode_indices")
                        and all(len(set(states)) == 1 for states in gradient_states)
                        and duration is not None and duration >= profile[0][-1] - 1e-12):
                    gradient_states = self.render_recorded_ppg_gradient(
                        plot, event, left, right, profile, gradient_states, gradient_colors, lanes,
                    )
                    continue
                component_match = re.search(r"\{([^}]*)\}", text)
                components = [part.strip() for part in component_match.group(1).split(",")] if component_match else []
                gradient_off = bool(re.search(r"\b(?:grad_off|groff)\b", text, re.IGNORECASE))
                shaped = bool(re.search(r"\bgrad_shape\b", text, re.IGNORECASE))
                compiled_target = event.get("compiled_ramped_target")
                phase_indices = event.get("phase_encode_indices", [0])
                indexed_lists = event.get("indexed_gradient_lists", ())
                if gradient_off:
                    compiled_target = None
                elif compiled_target is not None and not shaped:
                    source_components = components * 3 if len(components) == 1 else components
                    if len(source_components) == 3 and all(
                        self.resolve_ppg_gradient_component(
                            component, axis, phase_index, indexed_lists=indexed_lists,
                        ) is not None
                        for axis, component in enumerate(source_components)
                        for phase_index in phase_indices
                    ):
                        compiled_target = None
                compiled_values = None
                if compiled_target is not None:
                    indices_by_length = {
                        length: self.representative_gradient_indices(
                            [values for values in compiled_target if values.size == length], length,
                        )
                        for length in {values.size for values in compiled_target}
                    }
                    compiled_values = [
                        np.clip(values[indices_by_length[values.size]], -1.0, 1.0).tolist()
                        for values in compiled_target
                    ]
                shape_unavailable = shaped and compiled_target is None
                if gradient_off:
                    active_axes = [
                        index for index, states in enumerate(gradient_states)
                        if any(state is None or abs(state) > 1e-12 for state in states)
                    ]
                elif compiled_values is not None:
                    active_axes = [
                        index for index, targets in enumerate(compiled_values)
                        if any(targets) or any(state is None or abs(state) > 1e-12 for state in gradient_states[index])
                    ]
                elif "magnet_coord" in text or "rps_coord" in text or len(components) != 3:
                    active_axes = range(3)
                else:
                    active_axes = [
                        index for index, component in enumerate(components)
                        if component not in {"0", "0.0"}
                        or any(state is None or abs(state) > 1e-12 for state in gradient_states[index])
                    ]
                gradient_hovers: dict[float, list[tuple[float, str]]] = {}
                for axis_index in active_axes:
                    lane = lanes.get(_PPG_GRADIENTS[axis_index])
                    if lane is None:
                        continue
                    component = (
                        components[axis_index] if len(components) == 3 else
                        components[0] if len(components) == 1 else "G"
                    )
                    if compiled_values is not None:
                        targets = compiled_values[axis_index]
                    elif gradient_off or component in {"0", "0.0"}:
                        targets = [0.0]
                    else:
                        targets = []
                        for phase_index in phase_indices:
                            resolved = self.resolve_ppg_gradient_component(
                                component, axis_index, phase_index,
                                indexed_lists=indexed_lists,
                            )
                            targets.append(
                                float(np.clip(resolved / 100.0, -1.0, 1.0))
                                if resolved is not None and np.isfinite(resolved) else None
                            )
                    ramps = getattr(self, "pulseProgramStructure", {}).get("gradient_ramps", [])
                    ramp = np.asarray(ramps[axis_index], dtype=float) if axis_index < len(ramps) else np.asarray([])
                    ramp_placeholder = ramp.size < 2 or shape_unavailable
                    if ramp_placeholder:
                        ramp = np.asarray([0.0, 1.0])
                    ramp_durations = getattr(self, "pulseProgramStructure", {}).get("gradient_ramp_durations", [])
                    ramp_duration = (
                        float(ramp_durations[axis_index]) if axis_index < len(ramp_durations)
                        else physical_durations[index]
                    )
                    start_time = physical_edges[index]
                    end_time = min(start_time + ramp_duration, physical_edges[next_gradient[index]])
                    sample_times = np.linspace(start_time, start_time + ramp_duration, ramp.size)
                    crossed_edges = physical_edges[index + 1:next_gradient[index]]
                    ramp_times = np.union1d(sample_times[sample_times < end_time], crossed_edges[crossed_edges < end_time])
                    ramp_times = np.append(ramp_times, end_time)
                    ramp = np.interp(ramp_times, sample_times, ramp)
                    shape_x = np.interp(ramp_times, physical_edges, display_edges)
                    ramp_right = float(shape_x[-1])
                    gradient_ramp_ends[axis_index] = ramp_right
                    starts = gradient_states[axis_index]
                    if len(starts) == 1 and len(targets) > 1:
                        starts = starts * len(targets)
                    elif len(targets) == 1 and len(starts) > 1:
                        targets = targets * len(starts)
                    actual_pen = pg.mkPen(gradient_colors[axis_index], width=1.1)
                    gradient_tooltip = self.ppg_gradient_tooltip(
                        event, axis_index, component, targets,
                        ramp_placeholder=ramp_placeholder,
                        start_unresolved=any(state is None for state in starts),
                        compiled_ramped=compiled_target is not None,
                    )
                    transitions = dict.fromkeys(zip(starts, targets, strict=False))
                    for start_state, target in transitions:
                        start_height = unresolved_height if start_state is None else 0.42 * start_state
                        target_height = unresolved_height if target is None else 0.42 * target
                        shape_y = lane + start_height + ramp * (target_height - start_height)
                        held_placeholder = target is None or duration is None
                        shape_placeholder = held_placeholder or start_state is None or ramp_placeholder
                        curve = plot.plot(shape_x, shape_y, pen=placeholder_pen if shape_placeholder else actual_pen)
                        curve.setToolTip(gradient_tooltip)
                    if ramp_right < right:
                        for target in dict.fromkeys(targets):
                            target_height = unresolved_height if target is None else 0.42 * target
                            held_placeholder = target is None or duration is None or shape_unavailable
                            hold = plot.plot(
                                [ramp_right, right], [lane + target_height] * 2,
                                pen=placeholder_pen if held_placeholder else actual_pen,
                            )
                            hold.setToolTip(gradient_tooltip)
                    gradient_hovers.setdefault(lane, []).append((max(right, ramp_right), gradient_tooltip))
                    end_fraction = float(ramp[-1])
                    end_states = targets if end_fraction == 1.0 else [
                        start + end_fraction * (target - start) if start is not None and target is not None else None
                        for start, target in zip(starts, targets, strict=False)
                    ]
                    # Variants that converge share one starting level for the next ramp.
                    gradient_states[axis_index] = end_states[:1] if len(set(end_states)) == 1 else end_states
                for lane, tooltips in gradient_hovers.items():
                    self.add_schematic_hover_region(
                        plot, left, max(end for end, _ in tooltips), lane - .48, lane + .48,
                        "\n\n".join(tooltip for _, tooltip in tooltips),
                    )
            elif kind == "epi_readout":
                gradient_states = self.render_ppg_epi_readout(plot, event, left, right, lanes=lanes)
            elif kind == "acquisition" and "ADC" in lanes:
                setup = bool(re.search(r"\badc_\w+", text, re.IGNORECASE) and not re.search(
                    r"\bdwgen_\w+", text, re.IGNORECASE,
                ))
                if setup or duration is None or duration <= 0:
                    caption = "ADC setup" if setup else "ADC unresolved"
                    color = _PPG_PLACEHOLDER_COLOR
                    tooltip = f"ADC placeholder: {caption}. Sampling window unresolved.\n{text}"
                    curve = plot.plot([left, right], [0.28, 0.28], pen=placeholder_pen)
                    curve.setToolTip(tooltip)
                    self.add_schematic_hover_region(plot, left, right, 0.06, 0.5, tooltip)
                else:
                    name = "Navigator ADC" if re.search(r"\bdwgen_Navigator\b", text, re.IGNORECASE) else "ADC"
                    caption, color = f"{name} · {self.format_ppg_duration(duration)}", "#1d4ed8"
                    region = pg.LinearRegionItem(
                        (left, right), movable=False, brush=pg.mkBrush(70, 120, 210, 45), pen=None,
                    )
                    region.setZValue(-10)
                    plot.addItem(region)
                    curve = plot.plot([left, right], [0.28, 0.28], pen=pg.mkPen(color, width=3))
                    curve.setToolTip(caption)
                if (not setup and duration is not None and duration > 0) or right - left > 8 * units_per_pixel:
                    _, _, anchor = label_position(caption, center)
                    item = pg.TextItem(caption, color=color, anchor=(anchor, 1.0))
                    item.setPos(center, 0.38)
                    plot.addItem(item)

            elif kind == "subroutine":
                region = pg.LinearRegionItem((left, right), movable=False, brush=pg.mkBrush(100, 116, 139, 25), pen=pg.mkPen(100, 116, 139, 90))
                region.setZValue(-20)
                plot.addItem(region)
                item = pg.TextItem(str(event.get("name", "subroutine")), color="#334155", anchor=(0.5, 0.0), angle=-45)
                item.setPos(center, highest_lane + 1.05)
                plot.addItem(item)
            elif kind == "loop":
                start_record = event["start_record_index"]
                body_start = next(
                    ((body, start, end) for body, start, end in positioned
                     if body["kind"] != "loop" and start_record is not None
                     and body["record_index"] <= event["record_index"]
                     and body.get("end_record_index", body["record_index"]) >= start_record
                     and start <= left),
                    None,
                )
                target_x = max(0.0, left - total_width * 0.015)
                if body_start is not None:
                    body, target_x, body_end = body_start
                    if event["expanded"] and body["kind"] == "epi_readout":
                        right = body_end
                        if "loop_bounds" in body:
                            loop_start, loop_end = body["loop_bounds"]
                            scale = (body_end - target_x) / body["duration"]
                            right = target_x + loop_end * scale
                            target_x += loop_start * scale
                row = next(
                    (index for index, occupied_until in enumerate(loop_row_ends)
                     if target_x > occupied_until + total_width * 0.01),
                    len(loop_row_ends),
                )
                if row == len(loop_row_ends):
                    loop_row_ends.append(right)
                else:
                    loop_row_ends[row] = right
                y = highest_lane + 1.1 + _PPG_LOOP_ROW_STEP * row
                bracket = plot.plot([target_x, target_x, right, right], [y - 0.1, y, y, y - 0.1], pen=pg.mkPen("#111827", width=1.3))
                bracket.setToolTip(
                    f"{text}\nCount expression: {event['count_token']}\n"
                    f"Source: {event['source']}:{event['line']}"
                )
                self.add_schematic_hover_region(plot, target_x, right, y - 0.12, y + 0.12, bracket.toolTip())
                loop_name = {
                    "PVM_EpiEchoCounter": "EPI echo pairs",
                    "PVM_EpiNShots": "EPI shots",
                    "NA": "Averages",
                    "NR": "Repetitions",
                    "lds": "Dummy scans",
                    "c_echo2": "Echo images",
                }.get(event["count_token"]) or {
                    "decr": "dummy echoes",
                    "navigator": "navigator",
                    "echo": "Echoes",
                    "slice": "slices",
                    "acc": "averages",
                    "start": str(event.get("count_token", "scan")),
                }.get(event["target"], event["target"])
                count = event["count"]
                item = pg.TextItem(
                    f"{loop_name} ×{count}" if count is not None else f"{loop_name} (runtime)",
                    color="#111827", anchor=(0.5, 1.0),
                )
                item.setPos((target_x + right) * 0.5, y)
                plot.addItem(item)
        top = highest_lane + max(2.5, 1.3 + _PPG_LOOP_ROW_STEP * len(loop_row_ends))
        module_spans: list[tuple[str, float, float]] = []
        for event, left, right in positioned:
            name = _PPG_PREPARATION_LABELS.get(event.get("module"))
            if name is None or right <= left:
                continue
            if module_spans and module_spans[-1][0] == name and module_spans[-1][2] == left:
                module_spans[-1] = (name, module_spans[-1][1], right)
            else:
                module_spans.append((name, left, right))
        module_row_ends: list[float] = []
        for name, left, right in module_spans:
            center = (left + right) / 2
            label_left, label_right, anchor = label_position(name, center)
            row = next(
                (index for index, end in enumerate(module_row_ends) if label_left >= end),
                len(module_row_ends),
            )
            if row == len(module_row_ends):
                module_row_ends.append(label_right)
            else:
                module_row_ends[row] = label_right
            y = top + _PPG_LOOP_ROW_STEP * row
            color = "#0f766e"
            bracket = plot.plot([left, left, right, right], [y - .1, y, y, y - .1], pen=pg.mkPen(color))
            bracket.setToolTip(name)
            item = pg.TextItem(name, color=color, anchor=(anchor, 1.0))
            item.setPos(center, y)
            plot.addItem(item)
        plot.setYRange(-1.1, top + _PPG_LOOP_ROW_STEP * len(module_row_ends), padding=0)
        if timing_labels:
            self.layout_ppg_timing_labels(plot, timing_labels)
        return dict(sorted(delay_lengths.items(), key=lambda item: (
            0, int(item[0][1:]),
        ) if re.fullmatch(r"d\d+", item[0]) else (1, item[0])))

    def render_ppg_rf_pulse(
        self, plot: pg.PlotWidget, event: dict, left: float, right: float, caption: str,
        recorded: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None, placeholder_pen: QtGui.QPen,
        *, rf_lane: float | None, phase_lane: float | None,
    ) -> str:
        if rf_lane is None and phase_lane is None:
            return "#7b2cbf"
        nco = event.get("rf_channel")
        phase = None
        compiled = event["compiled_rf"]
        if compiled is not None:
            nco, shape = compiled
            amplitude = shape["amplitude"]
            if shape["phase"] is not None:
                phase = np.r_[shape["phase"], shape["phase"][-1]]
            fraction = np.linspace(0, 1, amplitude.size + 1)
            amplitude = np.r_[amplitude, amplitude[-1]]
            origin = f"Compiled RF envelope · {event['rf_shape']} · NCO {nco}"
        elif recorded is not None:
            fraction, amplitude, phase = recorded
            origin = "Recorded RF envelope"
        elif "rf_shape" in event and event["rf_shape"] is None and (event.get("duration") or 0) > 0:
            fraction = np.array([0.0, 1.0])
            amplitude, phase = np.ones(2), np.zeros(2)
            origin = "Programmed hard RF pulse"
        else:
            fraction = np.linspace(0, 1, 65)
            amplitude = np.sinc(2 * fraction - 1) ** 2
            origin = "Placeholder envelope. The shape is a representative template."

        placeholder = origin.startswith("Placeholder")
        color = _PPG_PLACEHOLDER_COLOR if placeholder else "#7b2cbf"
        x = left + fraction * (right - left)
        if rf_lane is not None:
            envelope = plot.plot(
                x, rf_lane + 0.68 * amplitude,
                pen=placeholder_pen if placeholder else pg.mkPen(color, width=1.6),
                stepMode=None if placeholder else "right",
            )
            rf_tooltip = self.ppg_rf_tooltip(event, caption, origin)
            envelope.setToolTip(rf_tooltip)
            self.add_schematic_hover_region(plot, left, right, rf_lane - 0.08, rf_lane + 0.82, rf_tooltip)
        if phase_lane is None:
            return color

        phase_x, phase_amplitude = x, amplitude
        use_recorded_phase = recorded is not None and recorded[2] is not None and (
            compiled is None or compiled[1]["phase"] is None
        )
        if use_recorded_phase:
            phase_fraction, phase_amplitude, phase = recorded
            phase_x = left + phase_fraction * (right - left)
        phase_name = event.get("rf_phase")
        phase_description = "Recorded phase registers"
        phase_placeholder = False
        if not use_recorded_phase:
            tables = self.pulseProgramStructure.get("rf_phase_lists", {})
            cycle = tables.get((phase_name, nco))
            if cycle is None:
                matches = [values for (name, _), values in tables.items() if name == phase_name]
                cycle = matches[0] if len(matches) == 1 else None
            offset = float(cycle[0]) if cycle is not None and len(cycle) else None
            if phase is not None:
                if offset is not None:
                    phase = np.mod(phase + offset, 360.0)
                    phase_description = f"Shape phase with first phase-cycle entry\n{phase_name} · {offset:g}°"
                else:
                    phase_description = "Shape phase"
                    if phase_name:
                        phase_placeholder = True
                        phase_description += f"\nPhase-cycle offset unresolved for {phase_name}"
            elif offset is not None:
                phase = np.full(amplitude.size, offset)
                phase_placeholder = True
                phase_description = f"First phase-cycle entry only. Shape phase unavailable.\n{phase_name} · {offset:g}°"

        if phase is not None:
            self.render_schematic_rf_phase(
                plot, phase_x, phase, phase_amplitude, f"RF phase · {caption}\n{phase_description}",
                placeholder_pen if phase_placeholder else pg.mkPen(_RF_PHASE_COLOR, width=1.5),
                lane=phase_lane,
            )
            self.add_schematic_hover_region(
                plot, left, right, phase_lane - 0.08, phase_lane + _RF_PHASE_HEIGHT,
                f"RF phase · {caption}\n{phase_description}",
            )
        return color

    @staticmethod
    def render_schematic_rf_phase(
        plot: pg.PlotWidget, x: np.ndarray, phase: np.ndarray, amplitude: np.ndarray, tooltip: str, pen: QtGui.QPen,
        *, lane: float = _RF_PHASE_LANE,
    ) -> None:
        y = np.where(np.abs(amplitude) > 1e-12, lane + _RF_PHASE_HEIGHT * phase / 360.0, np.nan)
        trace = plot.plot(x, y, pen=pen, stepMode="right", connect="finite")
        trace.setToolTip(tooltip)

    def render_recorded_ppg_gradient(
        self, plot: pg.PlotWidget, event: dict, left: float, right: float,
        profile: tuple[np.ndarray, np.ndarray], previous: list[list[float | None]], colors: tuple[str, ...],
        lanes: dict[str, float],
    ) -> list[list[float]]:
        times, values = profile
        duration = float(event["duration"])
        sample_x = np.concatenate((left + times * ((right - left) / duration), [right]))
        endpoints = []
        gradient_hovers: dict[float, list[str]] = {}
        for index, (axis, samples) in enumerate(zip(_PPG_GRADIENTS, values, strict=True)):
            heights = np.clip(samples / 100.0, -1.0, 1.0)
            endpoints.append([float(heights[-1])])
            lane = lanes.get(axis)
            if lane is None or (not np.any(heights) and all(state == 0.0 for state in previous[index])):
                continue
            y = lane + 0.42 * np.concatenate((heights, heights[-1:]))
            curve = plot.plot(sample_x, y, pen=pg.mkPen(colors[index], width=1.1), stepMode="right")
            tooltip = (
                f"{axis} recorded gradient\nEnd amplitude: {samples[-1]:.6g}%\n"
                f"Programmed duration: {self.format_ppg_duration(duration)}\n"
                f"Source: {event['source']}:{event['line']}\n{event['text']}"
            )
            curve.setToolTip(tooltip)
            gradient_hovers.setdefault(lane, []).append(tooltip)
        for lane, tooltips in gradient_hovers.items():
            self.add_schematic_hover_region(plot, left, right, lane - .48, lane + .48, "\n\n".join(tooltips))
        return endpoints

    @staticmethod
    def clip_schematic_gradient(line: dict, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
        """Clip timestamped gradient steps and hold the sample at each boundary."""
        time = np.asarray(line.get("t", []), dtype=float)
        values = np.asarray(line.get("raw_data", line.get("data", [])), dtype=float)
        if time.size == 0:
            return time, values
        first = time.searchsorted(start, side="right")
        last = time.searchsorted(end, side="left")
        end_index = time.searchsorted(end, side="right") - 1
        sample_time = np.concatenate(([start], time[first:last], [end]))
        start_value = values[first - 1] if first else 0.0
        end_value = values[end_index] if end_index >= 0 else 0.0
        gradient = np.concatenate(([start_value], values[first:last], [end_value]))
        return sample_time, gradient

    def render_ppg_epi_readout(
        self, plot: pg.PlotWidget, event: dict, left: float, right: float, *, lanes: dict[str, float] | None = None,
    ) -> list[list[float]]:
        start, duration = float(event["start"]), float(event["duration"])
        end = start + duration

        def map_time(times):
            return left + (np.asarray(times) - start) * ((right - left) / duration)

        lanes = _SCHEMATIC_LANES if lanes is None else lanes
        colors = dict(zip("xyz", _PPG_GRADIENT_COLORS, strict=True))
        states = {axis: [0.0] for axis in colors}
        shape_lengths = [
            f"{token} · {self.format_ppg_duration(length)}"
            for token, length in event.get("gradient_shape_durations", {}).items()
        ]
        train_length = event.get("gradient_train_duration")
        train_label = (
            f"Gradient train · {self.format_ppg_duration(train_length)}" if train_length is not None else ""
        )
        if "gradient_time" in event:
            gradients = [(axis, event["gradient_time"], values) for axis, values in zip(
                colors, event["gradient_values"], strict=True,
            )]
            origin = "Compiled"
        else:
            gradients = [
                (self.classify_gradient_axis(line), *self.clip_schematic_gradient(line, start, end))
                for channel in self.channels for line in channel if line.get("type") == "grads"
            ]
            origin = "Simulated"
        for axis, sample_time, gradient in gradients:
            channel = f"G{axis}"
            if channel not in lanes or sample_time.size == 0:
                continue
            normalized = np.clip(gradient / 100.0, -1.0, 1.0)
            states[axis] = [float(normalized[-1])]
            lane = lanes[channel]
            curve = plot.plot(
                map_time(sample_time), lane + 0.42 * normalized,
                pen=pg.mkPen(colors[axis], width=1.1), stepMode="right",
            )
            curve.setToolTip("\n".join([
                f"{origin} EPI gradient G{axis}", f"{event['source']}:{event['line']}",
                *([train_label, *shape_lengths] if train_label else []),
            ]))

        center = (left + right) * 0.5
        caption = pg.TextItem(
            f"EPI readout · {self.format_ppg_duration(duration)}", color="#111827", anchor=(0.5, 1.0),
        )
        caption.setPos(center, max(lanes.values(), default=0.0) + .8)
        plot.addItem(caption)
        visible_gradients = [lanes[axis] for axis in _PPG_GRADIENTS if axis in lanes]
        if shape_lengths and visible_gradients:
            shape_label = pg.TextItem(
                "    ".join((train_label, *shape_lengths)), color="#1d4ed8", anchor=(0.5, 0.5),
            )
            shape_label.setPos(center, max(visible_gradients) + .65)
            plot.addItem(shape_label)

        for window_start, window_end in event["acquisition_windows"] if "ADC" in lanes else ():
            acq_left, acq_right = map_time([window_start, window_end])
            region = pg.LinearRegionItem((acq_left, acq_right), movable=False, brush=pg.mkBrush(70, 120, 210, 45), pen=None)
            region.setZValue(-10)
            plot.addItem(region)
            curve = plot.plot([acq_left, acq_right], [0.28, 0.28], pen=pg.mkPen("#1d4ed8", width=3))
            caption = f"ADC {self.format_time(window_end - window_start)}"
            curve.setToolTip(caption)
            item = pg.TextItem(caption, color="#1d4ed8", anchor=(0.5, 1.0))
            item.setPos((acq_left + acq_right) * 0.5, 0.38)
            plot.addItem(item)
        return [states[axis] for axis in colors]

    def add_schematic_hover_region(
        self, plot: pg.PlotWidget, left: float, right: float, bottom: float, top: float, tooltip: str,
    ) -> None:
        """Add a transparent, easy-to-hit tooltip target over a schematic item."""
        width = max(right - left, 0.08)
        hitbox = QtWidgets.QGraphicsRectItem(left, bottom, width, top - bottom)
        hitbox.setPen(QtGui.QPen(Qt.PenStyle.NoPen))
        hitbox.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 1)))
        hitbox.setAcceptHoverEvents(True)
        hitbox.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        hitbox.setToolTip(tooltip)
        hitbox.setZValue(50)
        plot.addItem(hitbox)

    def ppg_gradient_tooltip(
        self, event: dict, axis_index: int, component: str, targets: list[float | None], *,
        ramp_placeholder: bool, start_unresolved: bool, compiled_ramped: bool,
    ) -> str:
        axis = ("Gx", "Gy", "Gz")[axis_index]
        structure = getattr(self, "pulseProgramStructure", {})
        parameters = structure.get("parameters", {})
        ramp_times = structure.get("gradient_ramp_durations", [])
        ramp_time = ramp_times[axis_index] if axis_index < len(ramp_times) else None
        referenced = []
        for name in dict.fromkeys(re.findall(r"[A-Za-z_]\w*", component)):
            if name in parameters:
                value = parameters[name]
                if isinstance(value, list):
                    value = f"array[{len(value)}]"
                referenced.append(f"{name}={value}")
        resolved = ", ".join(
            f"{100.0 * value:.4g}%" if value is not None else "unresolved" for value in dict.fromkeys(targets)
        )
        notes = []
        if len(set(targets) - {None}) > 1:
            notes.append("Overlaid amplitudes from selected loop iterations.")
        if compiled_ramped:
            notes.append("Compiled Gcon ramped point · physical axes, first slice.")
        if any(value is None for value in targets):
            notes.append("Amplitude unresolved. Dashed trace is a placeholder.")
        if start_unresolved:
            notes.append("Starting amplitude unresolved.")
        if ramp_placeholder:
            notes.append("Compiled waveform unavailable. The dashed trace is a placeholder.")
        if event.get("duration") is None:
            notes.append("Duration unresolved. Display width is a placeholder.")
        explanation = "".join(f"{note}\n" for note in notes)
        return (
            f"{axis} PPG expression: {component}\nResolved target(s): {resolved}\n"
            f"{explanation}"
            f"Duration: {self.format_time(event.get('duration'))}\n"
            f"Ramp time: {self.format_time(ramp_time)}\n"
            f"ParaVision/ACQP: {', '.join(referenced) if referenced else 'no named parameter'}\n"
            f"Source: {event.get('source')}:{event.get('line')}\n{event.get('text')}"
        )

    def ppg_rf_tooltip(self, event: dict, symbol: str, origin: str) -> str:
        calibrations = list(getattr(self, "rfPulseCalibrations", []))
        normalized = re.sub(r"[^a-z0-9]", "", symbol.lower())
        matching = next(
            (calibration for calibration in calibrations
             if normalized and normalized in re.sub(r"[^a-z0-9]", "", str(calibration.get("name", "")).lower())),
            None,
        )
        if matching is None:
            expected_name = "excpulse" if normalized == "p0" else "refpulse" if normalized == "p1" else ""
            matching = next(
                (calibration for calibration in calibrations
                 if expected_name in re.sub(r"[^a-z0-9]", "", str(calibration.get("name", "")).lower())),
                None,
            ) if expected_name else None
        pv = "not matched"
        if matching:
            pv = (
                f"{matching.get('name')}, duration={self.format_time(matching.get('duration'))}, "
                f"flip={matching.get('flip_angle')}°"
            )
        return (
            f"PPG RF pulse: {symbol}\nDuration: {self.format_time(event.get('duration'))}\n"
            f"{origin}\n"
            f"ParaVision pulse: {pv}\nSource: {event.get('source')}:{event.get('line')}\n{event.get('text')}"
        )

    def resolve_ppg_gradient_component(
        self, expression: str, axis_index: int = 0, phase_encode_index: int = 0, *,
        indexed_lists: Collection[str] = (),
    ) -> float | None:
        """Resolve simple PPG gradient arithmetic from method/acqp parameters."""
        parameters = getattr(self, "pulseProgramStructure", {}).get("parameters", {})
        cleaned = expression.replace("[]", "").strip()
        try:
            tree = ast.parse(cleaned, mode="eval")
        except SyntaxError:
            return None

        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Expression): return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return float(node.value)
            if isinstance(node, ast.Name):
                value = parameters.get(node.id)
                if isinstance(value, (int, float)): return float(value)
                if isinstance(value, list):
                    if node.id in indexed_lists or node.id.startswith("ACQ_spatial_phase_"):
                        value_index = phase_encode_index
                    elif node.id.startswith("PVM_EncGenValues"):
                        value_index = 0
                    else:
                        value_index = axis_index
                    if value_index < len(value) and isinstance(value[value_index], (int, float)):
                        return float(value[value_index])
                raise ValueError
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                values = parameters.get(node.value.id)
                index = evaluate(node.slice)
                if isinstance(values, list) and index.is_integer() and 0 <= index < len(values):
                    return float(values[int(index)])
                raise ValueError
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add): return left + right
                if isinstance(node.op, ast.Sub): return left - right
                if isinstance(node.op, ast.Mult): return left * right
                return left / right
            raise ValueError
        try:
            return evaluate(tree)
        except (ValueError, ZeroDivisionError):
            return None

    def get_ppg_duration_token(self, text: str) -> str | None:
        """Return the duration field preceding a PPG action."""
        body = re.sub(r"^\s*[A-Za-z_]\w*\s*,\s*", "", text).strip()
        match = re.match(rf"([A-Za-z_]\w*(?:\[\d+\])?|{_PPG_TIME.pattern})(?=\s|;|$)", body, re.IGNORECASE)
        return match.group(1) if match else None

    def resolve_ppg_duration(self, token: str, aliases: dict[str, str] | None = None) -> float | None:
        """Resolve a PPG delay token to seconds using the output D array."""
        structure = getattr(self, "pulseProgramStructure", {})
        parameters = structure.get("parameters", {})
        compiled_delays = structure.get("compiled_delays", {})
        aliases = {name.casefold(): value for name, value in (aliases or {}).items()}
        resolving = set()

        def parse(expression: str) -> ast.Expression:
            converted = re.sub(
                rf"(?<![\w.])({_PPG_NUMBER})\s*([nums])\b",
                lambda match: f"({match[1]}*{_PPG_TIME_FACTORS[match[2].casefold()]})",
                expression.strip(), flags=re.IGNORECASE,
            )
            return ast.parse(converted, mode="eval")

        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Expression): return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return float(node.value)
            if isinstance(node, ast.Name):
                name = node.id.casefold()
                if name in compiled_delays:
                    return float(compiled_delays[name])
                if name in aliases:
                    if name in resolving:
                        raise ValueError
                    resolving.add(name)
                    try:
                        return evaluate(parse(aliases[name]))
                    finally:
                        resolving.remove(name)
                value = parameters.get(node.id)
                if value is None:
                    value = next(
                        (candidate for key, candidate in parameters.items() if key.casefold() == name),
                        None,
                    )
                if value is None and node.id.startswith("ACQ_spatial_phase_"):
                    return 0.0
                if isinstance(value, (int, float)): return float(value)
                raise ValueError
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                values = parameters.get(node.value.id)
                index = evaluate(node.slice)
                if isinstance(values, list) and index.is_integer() and 0 <= index < len(values):
                    return float(values[int(index)])
                raise ValueError
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add): return left + right
                if isinstance(node.op, ast.Sub): return left - right
                if isinstance(node.op, ast.Mult): return left * right
                return left / right
            raise ValueError
        try:
            duration = evaluate(parse(token))
            return max(0.0, duration) if np.isfinite(duration) else None
        except (SyntaxError, ValueError, ZeroDivisionError):
            return None

    def get_ppg_gradient_shape_template(self, event: dict) -> list[np.ndarray] | None:
        """Return the real output waveform generated by this PPG source line."""
        timeline = getattr(self, "pulseProgramTimeline", None)
        mapping = getattr(self, "pulseProgramLineMapping", {})
        if timeline is None or len(timeline) != 2:
            return None
        times = np.asarray(timeline[0], dtype=float)
        internal_lines = np.asarray(timeline[1], dtype=int)
        if times.size < 2 or internal_lines.size != times.size:
            return None
        event_source = Path(str(event.get("source", ""))).name
        event_line = int(event.get("line", -1))
        candidates = [
            index for index, internal_line in enumerate(internal_lines[:-1])
            if Path(str(mapping.get(int(internal_line), {}).get("source", ""))).name == event_source
            and int(mapping.get(int(internal_line), {}).get("line", -9999)) == event_line
            and times[index + 1] > times[index]
        ]
        gradient_channel = next(
            (channel for channel in getattr(self, "channels", []) if channel and channel[0].get("type") == "grads"),
            [],
        )
        if not gradient_channel:
            return None
        best_shapes: list[np.ndarray] | None = None
        best_peak = 0.0
        for index in candidates:
            start, end = float(times[index]), float(times[index + 1])
            sample_time = np.linspace(start, end, 65)
            shapes = []
            for axis in range(3):
                if axis >= len(gradient_channel):
                    shapes.append(np.zeros(sample_time.size))
                    continue
                line = gradient_channel[axis]
                line_time = np.asarray(line.get("t", []), dtype=float)
                line_data = np.asarray(line.get("raw_data", line.get("data", [])), dtype=float)
                shapes.append(np.interp(sample_time, line_time, line_data) if line_time.size else np.zeros(sample_time.size))
            peak = max(float(np.max(np.abs(shape))) for shape in shapes)
            if peak > best_peak:
                best_peak, best_shapes = peak, shapes
        if best_shapes is None or best_peak <= 1e-12:
            return None
        return [shape / best_peak for shape in best_shapes]

    def render_sequence_schematic(self, plot: pg.PlotWidget, cycle: dict) -> None:
        self.clear_schematic(plot)
        start, end = float(cycle["start"]), float(cycle["end"])
        width = max(end - start, 1e-12)
        pulses = [
            pulse for pulse in self.detect_rf_pulse_descriptors()
            if float(pulse["end"]) >= start and float(pulse["start"]) < end
        ]
        acquisition_windows = [
            (max(window_start, start), min(window_end, end))
            for window_start, window_end in self.detect_acquisition_windows()
            if window_end >= start and window_start < end
        ]
        active_intervals = [(float(pulse["start"]), float(pulse["end"])) for pulse in pulses]
        active_intervals.extend(acquisition_windows)
        gradient_lines = [
            line for channel in getattr(self, "channels", []) for line in channel if line.get("type") == "grads"
        ]
        gradient_traces = []
        for line in gradient_lines:
            time, data = self.clip_schematic_gradient(line, start, end)
            if time.size == 0:
                continue
            # A sample at the end timestamp belongs to the next cycle.
            data[-1] = data[-2]
            gradient_traces.append((self.classify_gradient_axis(line), time, data))
            active_mask = np.abs(data[:-1]) > 1e-12
            active_starts = np.flatnonzero(active_mask & np.concatenate(([True], ~active_mask[:-1])))
            active_ends = np.flatnonzero(active_mask & np.concatenate((~active_mask[1:], [True])))
            active_intervals.extend(
                (float(time[first]), float(time[last + 1]))
                for first, last in zip(active_starts, active_ends, strict=True)
            )
        active_intervals.sort()
        merged_intervals: list[list[float]] = []
        margin = width * 0.002
        for interval_start, interval_end in active_intervals:
            interval_start, interval_end = max(start, interval_start - margin), min(end, interval_end + margin)
            if merged_intervals and interval_start <= merged_intervals[-1][1] + margin:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], interval_end)
            else:
                merged_intervals.append([interval_start, interval_end])
        breakpoints = sorted({start, end, *(value for interval in merged_intervals for value in interval)})
        active_total = sum(interval[1] - interval[0] for interval in merged_intervals)
        idle_cap = max(active_total * 0.08, width * 0.003, 1e-9)
        mapped_breakpoints = [0.0]
        for left, right in zip(breakpoints[:-1], breakpoints[1:], strict=True):
            midpoint = (left + right) * 0.5
            is_active = any(interval[0] <= midpoint <= interval[1] for interval in merged_intervals)
            mapped_breakpoints.append(mapped_breakpoints[-1] + ((right - left) if is_active else min(right - left, idle_cap)))

        def map_time(values):
            return np.interp(values, np.asarray(breakpoints), np.asarray(mapped_breakpoints))

        mapped_start, mapped_end = float(map_time(start)), float(map_time(end))
        self.configure_schematic_lanes(plot)
        bottom_axis = plot.getPlotItem().getAxis("bottom")
        bottom_axis.setTicks([[
            (mapped_start, self.format_time(start)),
            (mapped_end, self.format_time(end)),
        ]])
        plot.setLabel("bottom", "Sequence time (long idle delays compressed)")
        plot.setYRange(-0.8, 6.55, padding=0)
        plot.setXRange(mapped_start, mapped_end, padding=0.01)

        for position in _SCHEMATIC_LANES.values():
            plot.plot([mapped_start, mapped_end], [position, position], pen=pg.mkPen("#777777", width=1))

        for nco, signals in schematic_rf_channels(self.channels).items():
            if "am" not in signals:
                continue
            time, data = signals["am"]
            mask = (time >= start) & (time < end)
            if not np.any(mask):
                continue
            time, amplitude = time[mask], data[mask]
            peak = max(float(np.max(np.abs(amplitude))), 1e-12)
            x = map_time(time)
            plot.plot(x, _RF_LANE + 0.72 * amplitude / peak, pen=pg.mkPen("#7b2cbf", width=1.5), stepMode="right")
            phase = recorded_rf_phase(signals, time)
            if phase is not None:
                self.render_schematic_rf_phase(
                    plot, x, phase, amplitude, f"RF phase · NCO {nco}\nRecorded p0 + p1 + p2, wrapped to 0–360°",
                    pg.mkPen(_RF_PHASE_COLOR, width=1.5),
                )

        colors = {"x": "#15803d", "y": "#dc2626", "z": "#2563eb"}
        for gradient_axis, time, data in gradient_traces:
            if gradient_axis is None:
                continue
            peak = max(float(np.max(np.abs(data))), 1e-12)
            lane = _SCHEMATIC_LANES[f"G{gradient_axis}"]
            plot.plot(
                map_time(time), lane + 0.38 * data / peak,
                pen=pg.mkPen(colors[gradient_axis], width=1.4), stepMode="right",
            )

        for visible_start, visible_end in acquisition_windows:
            mapped_window_start, mapped_window_end = map_time([visible_start, visible_end])
            region = pg.LinearRegionItem(
                values=(mapped_window_start, mapped_window_end), orientation="vertical", movable=False,
                brush=pg.mkBrush(70, 120, 210, 45), pen=pg.mkPen(70, 120, 210, 100),
            )
            region.setZValue(-10)
            plot.addItem(region)
            plot.plot([mapped_window_start, mapped_window_end], [0.25, 0.25], pen=pg.mkPen("#1d4ed8", width=3))
            text = pg.TextItem("acq", color="#1d4ed8", anchor=(0.5, 1.0))
            text.setPos((mapped_window_start + mapped_window_end) * 0.5, 0.35)
            plot.addItem(text)

        # Label calibrated RF pulses at their matched focus. Preparation pulses
        # remain visible and named rather than being mistaken for excitations.
        labelled_calibrations: set[str] = set()
        for calibration in getattr(self, "rfPulseCalibrations", []):
            for focus in self.match_rf_calibration_foci(calibration, pulses):
                if not start <= focus < end:
                    continue
                label = str(calibration.get("name", "RF"))
                if label in labelled_calibrations:
                    continue
                labelled_calibrations.add(label)
                angle = float(calibration.get("flip_angle", 0.0))
                text = pg.TextItem(f"{label}\n{angle:g}°", color="#5b21b6", anchor=(0.5, 1.0))
                text.setPos(float(map_time(focus)), 5.82)
                plot.addItem(text)

        # Draw contiguous source-module spans above the waveform lanes.
        labelled_modules: set[str] = set()
        for module_index, module in enumerate(cycle["modules"]):
            events = module["events"]
            module_start = float(events[0]["time"])
            module_end = min(float(events[-1]["time"]) + float(events[-1]["delay"]), end)
            mapped_module_start, mapped_module_end = map_time([module_start, module_end])
            plot.plot([mapped_module_start, mapped_module_start, mapped_module_end, mapped_module_end], [5.95, 6.08, 6.08, 5.95], pen=pg.mkPen("#374151"))
            source = str(module["source"])
            if source not in labelled_modules and mapped_module_end - mapped_module_start > (mapped_end - mapped_start) * 0.025:
                labelled_modules.add(source)
                text = pg.TextItem(source, color="#111827", anchor=(0.5, 0.0))
                text.setPos((mapped_module_start + mapped_module_end) * 0.5, 6.08 + 0.16 * (module_index % 2))
                plot.addItem(text)

        # Mark only visually significant delays to prevent label collisions.
        delays = [
            event for module in cycle["modules"] for event in module["events"]
            if float(event["delay"]) >= width * 0.015
        ]
        for event in sorted(delays, key=lambda item: float(item["delay"]), reverse=True)[:12]:
            delay_start = float(event["time"])
            delay_end = min(delay_start + float(event["delay"]), end)
            mapped_delay_start, mapped_delay_end = map_time([delay_start, delay_end])
            plot.plot([mapped_delay_start, mapped_delay_start, mapped_delay_end, mapped_delay_end], [-0.4, -0.52, -0.52, -0.4], pen=pg.mkPen("#4b5563"))
            text = pg.TextItem(f"Δ {self.format_time(float(event['delay']))}", color="#374151", anchor=(0.5, 0.0))
            text.setPos((mapped_delay_start + mapped_delay_end) * 0.5, -0.68)
            plot.addItem(text)

    def format_measurement_entry(
        self,
        start_time: float,
        end_time: float,
        delta_time: float,
        label: str = "",
    ) -> str:
        start_text = self.format_time(start_time)
        end_text = self.format_time(end_time)
        delta_text = self.format_time(delta_time)
        prefix = f"{label}: " if label else ""
        return f"{prefix}{delta_text}  [{start_text} -> {end_text}]"

    def refresh_measurements_list(self) -> None:
        if not hasattr(self, "measurementsListWidget"):
            return

        selected_row = self.measurementsListWidget.currentRow()
        self.measurementsListWidget.clear()
        for measurement in getattr(self, "measurements", []):
            start_time = float(measurement["start"])
            end_time = float(measurement["end"])
            delta_time = float(measurement["delta"])
            label = str(measurement.get("label", ""))
            item = QListWidgetItem(self.format_measurement_entry(start_time, end_time, delta_time, label))
            item.setData(Qt.ItemDataRole.UserRole, measurement)
            self.measurementsListWidget.addItem(item)

        has_measurements = bool(getattr(self, "measurements", []))
        if has_measurements:
            self.measurementsListWidget.setCurrentRow(min(max(selected_row, 0), len(self.measurements) - 1))
        if hasattr(self, "removeMeasurementButton"):
            self.removeMeasurementButton.setEnabled(has_measurements)
        if hasattr(self, "clearMeasurementsButton"):
            self.clearMeasurementsButton.setEnabled(has_measurements)
        if hasattr(self, "exportMeasurementsButton"):
            self.exportMeasurementsButton.setEnabled(has_measurements)
        self.on_measurement_selection_changed()

    def add_persistent_measurement(self, start_time: float, end_time: float) -> None:
        normalized_start = min(float(start_time), float(end_time))
        normalized_end = max(float(start_time), float(end_time))
        measurement_count = len(getattr(self, "measurements", [])) + 1
        measurement_entry = {
            "label": f"Measurement {measurement_count}",
            "start": normalized_start,
            "end": normalized_end,
            "delta": normalized_end - normalized_start,
        }
        if not hasattr(self, "measurements"):
            self.measurements = []
        self.measurements.append(measurement_entry)
        self.refresh_measurements_list()
        self.measurementsListWidget.setCurrentRow(len(self.measurements) - 1)

    def on_measurement_selection_changed(self, *args: object) -> None:
        if not hasattr(self, "measurementsListWidget") or not hasattr(self, "measurementLabelEdit"):
            return

        current_row = self.measurementsListWidget.currentRow()
        has_selection = 0 <= current_row < len(getattr(self, "measurements", []))
        self.measurementLabelEdit.setEnabled(has_selection)
        if hasattr(self, "saveMeasurementLabelButton"):
            self.saveMeasurementLabelButton.setEnabled(has_selection)
        if not has_selection:
            self.measurementLabelEdit.clear()
            return
        label = str(self.measurements[current_row].get("label", ""))
        self.measurementLabelEdit.setText(label)

    def rename_selected_measurement(self) -> None:
        if not hasattr(self, "measurementLabelEdit") or not hasattr(self, "measurementsListWidget"):
            return

        current_row = self.measurementsListWidget.currentRow()
        if current_row < 0 or current_row >= len(getattr(self, "measurements", [])):
            return

        label = self.measurementLabelEdit.text().strip()
        if not label:
            label = f"Measurement {current_row + 1}"
        self.measurements[current_row]["label"] = label
        self.refresh_measurements_list()
        self.measurementsListWidget.setCurrentRow(current_row)

    def remove_selected_measurement(self) -> None:
        if not hasattr(self, "measurementsListWidget"):
            return

        current_row = self.measurementsListWidget.currentRow()
        if current_row < 0 or current_row >= len(getattr(self, "measurements", [])):
            return
        self.measurements.pop(current_row)
        self.refresh_measurements_list()

    def clear_measurements(self) -> None:
        self.measurements = []
        self.refresh_measurements_list()

    def show_saved_measurement(self, start_time: float, end_time: float) -> None:
        if not hasattr(self, "plots"):
            return

        self.measurement_start_x = float(start_time)
        self.measurement_source_plot = None
        for plot in self.plots:
            plot.ensure_measurement_overlay(start_time)
            plot.update_measurement_overlay(end_time)

    def jump_to_measurement_item(self, item: QListWidgetItem) -> None:
        measurement = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(measurement, dict):
            return

        start_time = float(measurement.get("start", 0.0))
        end_time = float(measurement.get("end", start_time))
        mid_time = (start_time + end_time) * 0.5
        self.tPos = mid_time
        measurement_width = max(end_time - start_time, 0.0)
        if measurement_width > 0:
            self.windowWidth = max(self.get_min_zoom_width(), max(self.windowWidth, measurement_width * 1.5))
        self.updateView()
        self.show_saved_measurement(start_time, end_time)
        self.update_status(cursor_time=mid_time, measurement=end_time - start_time)

    def update_sidebar_toggle_button(self) -> None:
        if not hasattr(self, "sidebarToggleButton"):
            return
        collapsed = bool(getattr(self, "sidebarCollapsed", False))
        self.sidebarToggleButton.setText(">" if collapsed else "<")
        self.sidebarToggleButton.setToolTip("Show sidebar" if collapsed else "Hide sidebar")

    def set_sidebar_available(self, available: bool) -> None:
        if not hasattr(self, "sidePanelDock"):
            return
        self.sidePanelDock.setVisible(available)
        if not available:
            return
        self.set_side_panel_collapsed(bool(getattr(self, "sidebarCollapsed", False)), save_setting=False)

    def set_side_panel_collapsed(self, collapsed: bool, *, save_setting: bool = True) -> None:
        if not hasattr(self, "mainSplitter") or not hasattr(self, "sidePanelDock"):
            return

        self.sidebarCollapsed = collapsed
        rail_width = max(self.sidebarRail.sizeHint().width(), self.sidebarToggleButton.width())

        if collapsed:
            current_width = max(self.mainSplitter.sizes()[0] - rail_width, 0)
            if current_width > 0:
                self.sidebarWidth = current_width
            self.sidePanel.hide()
            self.mainSplitter.setSizes([rail_width, max(self.mainSplitter.width() - rail_width, 1)])
        else:
            self.sidePanel.show()
            target_width = max(int(getattr(self, "sidebarWidth", 260)), 180)
            self.mainSplitter.setSizes([target_width + rail_width, max(self.mainSplitter.width() - target_width, 1)])

        self.update_sidebar_toggle_button()
        if save_setting and hasattr(self, "settings"):
            self.settings.setValue("sidebarCollapsed", self.sidebarCollapsed)
            self.settings.setValue("sidebarWidth", int(getattr(self, "sidebarWidth", 260)))

    def toggle_side_panel(self) -> None:
        self.set_side_panel_collapsed(not bool(getattr(self, "sidebarCollapsed", False)))

    def on_main_splitter_moved(self, pos: int, index: int) -> None:
        if not hasattr(self, "mainSplitter") or bool(getattr(self, "sidebarCollapsed", False)):
            return
        rail_width = max(self.sidebarRail.width(), self.sidebarToggleButton.width())
        sidebar_width = max(self.mainSplitter.sizes()[0] - rail_width, 180)
        self.sidebarWidth = sidebar_width
        if hasattr(self, "settings"):
            self.settings.setValue("sidebarWidth", sidebar_width)

    def get_min_zoom_width(self) -> float:
        full_width = max(float(getattr(self, "tMax", 0.0)) - float(getattr(self, "tMin", 0.0)), 0.0)
        if full_width <= 0:
            return 1e-15
        return max(full_width * 1e-12, 1e-15)

    def make_dark_palette(self) -> QPalette:
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(190, 190, 190))
        return palette

    def get_theme_palette(self, theme_mode: str) -> tuple[QPalette, bool]:
        if theme_mode == "dark":
            return self.make_dark_palette(), True
        if theme_mode == "light":
            return QPalette(self.standardPalette), False

        palette = QPalette(self.systemPalette)
        base_color = palette.color(QPalette.ColorRole.Base)
        return palette, base_color.value() < 128

    def apply_theme_settings(self) -> None:
        qt_app = QApplication.instance()
        if qt_app is None:
            return

        palette, dark_mode = self.get_theme_palette(self.themeMode)
        if dark_mode:
            palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(190, 190, 190))
        else:
            palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(110, 110, 110))
        qt_app.setPalette(palette)
        self.setPalette(palette)
        self.darkMode = dark_mode

        if hasattr(self, "channelFilter"):
            filter_palette = self.channelFilter.palette()
            filter_palette.setColor(QPalette.ColorRole.PlaceholderText, palette.color(QPalette.ColorRole.PlaceholderText))
            self.channelFilter.setPalette(filter_palette)

        if self.darkMode:
            pg.setConfigOption("background", "black")
            pg.setConfigOption("foreground", "white")
        else:
            pg.setConfigOption("background", "white")
            pg.setConfigOption("foreground", "black")
        self.apply_widget_theme_styles()

    def apply_widget_theme_styles(self) -> None:
        if not hasattr(self, "sidePanel"):
            return

        assets_dir = Path(__file__).resolve().parent / "assets"

        if self.darkMode:
            window_bg = "#13161b"
            text_color = "#e7ecf3"
            muted_text = "#99a5b6"
            panel_bg = "#1b2028"
            panel_alt_bg = "#232934"
            input_bg = "#151922"
            border = "#303846"
            border_strong = "#465165"
            tab_bg = "#222834"
            tab_selected_bg = "#171b23"
            button_bg = "#262d39"
            button_hover = "#2e3745"
            button_pressed = "#1f2631"
            button_text = "#dce5f2"
            button_border = "#3b4657"
            checkbox_bg = "#171c24"
            accent = "#6f94d6"
            accent_soft = "#263a57"
            slider_groove = "#252c37"
            slider_subpage = "#5f83c5"
            slider_handle = "#dde5f1"
            slider_handle_border = "#8097ba"
            combo_arrow = (assets_dir / "combo_arrow_light.svg").as_posix()
            splitter_handle = "#2a313d"
            splitter_handle_hover = "#3b4555"
        else:
            window_bg = "#edf1f5"
            text_color = "#1f2936"
            muted_text = "#2e3c4d"
            panel_bg = "#f7f9fb"
            panel_alt_bg = "#eef2f6"
            input_bg = "#ffffff"
            border = "#ccd5df"
            border_strong = "#b2bfcd"
            tab_bg = "#e9eef4"
            tab_selected_bg = "#ffffff"
            button_bg = "#f2f5f8"
            button_hover = "#e8edf3"
            button_pressed = "#dde5ee"
            button_text = "#223041"
            button_border = "#c5d0db"
            checkbox_bg = "#ffffff"
            accent = "#466b9f"
            accent_soft = "#dde7f4"
            slider_groove = "#d7dee6"
            slider_subpage = "#6d8fbe"
            slider_handle = "#ffffff"
            slider_handle_border = "#9fb0c3"
            combo_arrow = (assets_dir / "combo_arrow_dark.svg").as_posix()
            splitter_handle = "#d6dde6"
            splitter_handle_hover = "#bcc8d4"

        style = f"""
            QMainWindow {{
                background: {window_bg};
            }}
            QWidget {{
                color: {text_color};
            }}
            QGroupBox {{
                color: {text_color};
                background: {panel_bg};
                border: 1px solid {border};
                border-radius: 8px;
                margin-top: 1.05em;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {muted_text};
                background: {panel_bg};
                letter-spacing: 0.04em;
            }}
            QLabel {{
                color: {text_color};
            }}
            QTabWidget {{
                background: transparent;
            }}
            QTabWidget::pane {{
                border: 1px solid {border};
                border-radius: 10px;
                background: {panel_bg};
                top: -1px;
            }}
            QWidget#sidePanelDock {{
                background: {panel_bg};
            }}
            QWidget#sidebarRail {{
                background: {panel_alt_bg};
                border-right: 1px solid {border};
            }}
            QTabBar::tab {{
                background: {tab_bg};
                color: {text_color};
                border: 1px solid {border};
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                padding: 7px 13px;
                margin-right: 4px;
            }}
            QTabBar::tab:selected {{
                background: {tab_selected_bg};
                border-color: {border_strong};
                font-weight: 600;
            }}
            QPushButton {{
                color: {button_text};
                background: {button_bg};
                border: 1px solid {button_border};
                border-radius: 7px;
                padding: 7px 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: {button_hover};
            }}
            QPushButton:pressed {{
                background: {button_pressed};
            }}
            QPushButton#primaryButton {{
                color: white;
                background: {accent};
                border-color: {accent};
                font-weight: 600;
            }}
            QPushButton#primaryButton:hover {{
                background: {slider_subpage};
            }}
            QPushButton#primaryButton:pressed {{
                background: {button_pressed};
            }}
            QPushButton#modeToggleButton:checked {{
                background: {accent_soft};
                border-color: {accent};
                color: {text_color};
                font-weight: 600;
            }}
            QPushButton#sidebarToggleButton {{
                min-width: 22px;
                max-width: 22px;
                padding: 0;
                border-radius: 6px;
                font-weight: 600;
            }}
            QLineEdit, QDoubleSpinBox, QComboBox {{
                color: {text_color};
                background: {input_bg};
                border: 1px solid {border};
                border-radius: 7px;
                padding: 6px 9px;
                selection-background-color: #3a7bd5;
                selection-color: #ffffff;
            }}
            QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border: 1px solid {accent};
                background: {input_bg};
            }}
            QAbstractSpinBox {{
                color: {text_color};
                background: {input_bg};
                border: 1px solid {border};
                border-radius: 7px;
            }}
            QComboBox {{
                padding-right: 30px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left: 1px solid {border};
                background: {panel_alt_bg};
                border-top-right-radius: 7px;
                border-bottom-right-radius: 7px;
            }}
            QComboBox::down-arrow {{
                image: url("{combo_arrow}");
                width: 10px;
                height: 6px;
            }}
            QComboBox QAbstractItemView {{
                background: {input_bg};
                color: {text_color};
                border: 1px solid {border_strong};
                selection-background-color: {accent};
                selection-color: white;
            }}
            QSlider::groove:horizontal {{
                height: 10px;
                background: {slider_groove};
                border-radius: 5px;
            }}
            QSlider::sub-page:horizontal {{
                background: {slider_subpage};
                border-radius: 5px;
            }}
            QSlider::add-page:horizontal {{
                background: {slider_groove};
                border-radius: 5px;
            }}
            QSlider::handle:horizontal {{
                width: 18px;
                background: {slider_handle};
                border: 1px solid {slider_handle_border};
                border-radius: 9px;
                margin: -6px 0;
            }}
            QSlider::handle:horizontal:hover {{
                border-color: {accent};
            }}
            QScrollArea, QListWidget, QListView {{
                background: {panel_bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            QSplitter::handle {{
                background: {splitter_handle};
            }}
            QSplitter::handle:hover {{
                background: {splitter_handle_hover};
            }}
            QSplitter::handle:horizontal {{
                width: 6px;
            }}
            QCheckBox {{
                color: {text_color};
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 5px;
                border: 1px solid {border_strong};
                background: {checkbox_bg};
            }}
            QCheckBox::indicator:checked {{
                background: {accent};
                border-color: {accent};
            }}
        """

        self.setStyleSheet(style)
        if hasattr(self, "channelListWidget"):
            self.channelListWidget.setStyleSheet(f"background: {panel_bg}; color: {text_color};")

    def update_existing_plot_themes(self) -> None:
        for plot in getattr(self, "plots", []):
            plot.apply_theme(dark_mode=self.darkMode)

    def on_theme_mode_changed(self) -> None:
        self.themeMode = str(self.themeModeComboBox.currentData())
        self.settings.setValue("themeMode", self.themeMode)
        self.apply_theme_settings()
        self.update_existing_plot_themes()
        self.update()

    def activate_measure(self) -> None:
        self.setInteractionMode("measure")

    def registerShortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Open, self, activated=self.open_folder)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Plus), self, activated=self.zoomIn)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Minus), self, activated=self.zoomOut)
        QtGui.QShortcut(QtGui.QKeySequence("A"), self, activated=self.autoscale_visible_y)
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, activated=self.resetView)
        QtGui.QShortcut(QtGui.QKeySequence("M"), self, activated=self.measureButton.toggle)
        QtGui.QShortcut(QtGui.QKeySequence("Z"), self, activated=self.zoomModeButton.toggle)
        QtGui.QShortcut(QtGui.QKeySequence("E"), self, activated=self.snapMeasureAction.toggle)
        QtGui.QShortcut(QtGui.QKeySequence("T"), self, activated=self.zeroTrajectoryAtCursorAction.trigger)
        QtGui.QShortcut(QtGui.QKeySequence("F"), self, activated=self.refocusTrajectoryAtCursorAction.trigger)
        QtGui.QShortcut(QtGui.QKeySequence("Shift+F"), self, activated=self.learnRefocusPulseAction.trigger)
        QtGui.QShortcut(QtGui.QKeySequence("J"), self, activated=self.jumpToPpgLineAction.trigger)
        QtGui.QShortcut(QtGui.QKeySequence("["), self, activated=self.jump_to_previous_rf_pulse)
        QtGui.QShortcut(QtGui.QKeySequence("]"), self, activated=self.jump_to_next_rf_pulse)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Left), self, activated=self.jumpXNeg)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Right), self, activated=self.jumpXPos)

    def showUserGuide(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("SimView User Guide")
        dialog.resize(760, 680)
        layout = QtWidgets.QVBoxLayout(dialog)
        guide = QtWidgets.QTextBrowser(dialog)
        guide.setOpenExternalLinks(True)
        guide.setHtml(
            """
            <h1>SimView User Guide</h1>
            <p>SimView displays RF, gradient, acquisition, and calculated pulse-sequence signals on a shared time axis.</p>

            <h2>1. Load and inspect a sequence</h2>
            <ol>
              <li>Choose <b>File → Open Folder</b> and select a Bruker simulation or NMRScopeB output folder.</li>
              <li>Use the <b>Channels</b> tab to show the signals you need. The filter searches channel names; <b>Show All</b> and <b>Hide All</b> affect the filtered list.</li>
              <li>Move the mouse over any plot. The red cursor, timestamp, curve values, status bar, and b-matrix readout update together.</li>
            </ol>

            <h2>2. Navigate the timeline</h2>
            <ul>
              <li><b>Zoom + / Zoom −</b> change the visible time span. The bottom slider moves the window.</li>
              <li><b>Previous RF / Next RF</b> jump between detected RF pulse starts.</li>
              <li><b>Reset View</b> restores the complete time range; <b>A</b> rescales visible plots vertically.</li>
              <li><b>Zoom Mode</b> lets you drag a time region. Mouse-wheel gestures are listed under Help → Keyboard &amp; Mouse Shortcuts.</li>
            </ul>

            <h2>3. Measure time intervals</h2>
            <p>Enable <b>Measure</b>, click once to start and again to finish. Completed measurements appear in the
            <b>Measurements</b> tab and can be labelled, revisited, removed, or exported. Enable
            <b>Stick Measurements To Events</b> to snap endpoints to nearby sequence events.</p>

            <h2>4. Gradient and trajectory channels</h2>
            <ul>
              <li><b>Gradients</b>: recorded gradient waveforms in magnet x, y, and z coordinates.</li>
              <li><b>Gradient Trajectory</b>: the unmodified integral of each recorded gradient.</li>
              <li><b>Effective Gradient</b>: the gradient with its sign reversed after every selected 180° refocusing pulse.</li>
              <li><b>Effective Trajectory</b>: the integral of the effective gradient. Echo detection and the b-matrix use this trajectory.</li>
              <li><b>Gradient Trajectory Residual</b>: the magnitude of the effective trajectory.</li>
            </ul>

            <h2>5. Define the coherence path</h2>
            <p>Place the cursor at the desired event and use:</p>
            <ul>
              <li><b>T / Zero Trajectory At Cursor</b> to define the trajectory origin.</li>
              <li><b>F / 180 Flip</b> to add a refocusing pulse at the cursor.</li>
              <li><b>Shift+F</b> while hovering over an RF pulse to identify pulses of that duration as 180° pulses.</li>
              <li><b>Reset Traj</b> to clear the selected origin and refocusing flips.</li>
            </ul>
            <p>Detected flips, trajectory resets, echoes, and acquisition windows are drawn as annotations. Detailed
            flip management and echo results are available in the <b>Settings</b> tab.</p>

            <h2>6. b-matrix at the cursor</h2>
            <p>The panel below the channel list shows the cumulative 3×3 b-matrix at the cursor in <b>s/mm²</b>.
            Rows and columns are in <b>magnet coordinates (x, y, z)</b>; off-diagonal entries describe interactions
            between gradient axes. The calculation follows the effective trajectory and therefore updates when the
            trajectory origin or 180° flips change.</p>
            <p>A physical gradient calibration is required. Configure <b>Grad Calibration</b>, nucleus,
            and display units in the <b>Settings</b> tab. Without calibration, the panel displays dashes.</p>

            <h2>7. Export and display settings</h2>
            <ul>
              <li><b>File → Export Visible Plots</b> exports the currently shown plots.</li>
              <li><b>File → Export Measurements</b> writes completed measurements to a spreadsheet.</li>
              <li>The <b>Settings</b> tab controls theme, gradient calibration and units, derived-signal parameters,
              gradient layout, RF refocusing assumptions, and echo tolerance.</li>
            </ul>

            <h2>Reading the display</h2>
            <p>The status bar reports interaction mode, snap state, visible span, cursor time, active measurement,
            number of 180° flips, effective trajectory components and magnitude, and the pulse-program location.
            Coloured shaded regions mark acquisition jobs; dashed labelled lines mark RF foci, flips, resets, and echoes.</p>
            """
        )
        guide.moveCursor(QtGui.QTextCursor.MoveOperation.Start)
        layout.addWidget(guide)
        close_button = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        close_button.rejected.connect(dialog.reject)
        layout.addWidget(close_button)
        dialog.exec()

    def showShortcutsHelp(self) -> None:
        help_text = (
            "<b>Keyboard shortcuts</b><br><br>"
            "<b>Ctrl+O</b> Open folder<br>"
            "<b>+</b> Zoom in<br>"
            "<b>-</b> Zoom out<br>"
            "<b>A</b> Autoscale Y in current X window<br>"
            "<b>Left / Right</b> Pan backward / forward<br>"
            "<b>R</b> Reset full view<br>"
            "<b>M</b> Toggle measure mode<br>"
            "<b>Z</b> Toggle zoom mode<br>"
            "<b>E</b> Toggle measure snap to events<br>"
            "<b>T</b> Zero trajectory at cursor<br>"
            "<b>F</b> Add 180 refocus flip at cursor<br>"
            "<b>Shift+F</b> Treat hovered RF pulse as 180°<br>"
            "<b>[ / ]</b> Jump previous / next RF pulse<br>"
            "<b>J</b> Jump to a pulse-program line<br>"
            "<b>F1</b> Open the user guide<br><br>"
            "<b>Mouse controls</b><br><br>"
            "<b>Move mouse</b> Inspect synced cursor across plots<br>"
            "<b>Measure mode</b> Click once to start, click again to finish<br>"
            "<b>Zoom mode</b> Left click-drag to zoom into a region<br>"
            "<b>Shift + left drag</b> Temporary zoom without switching modes<br>"
            "<b>Mouse wheel</b> Horizontal zoom around cursor<br>"
            "<b>Ctrl + wheel</b> Vertical zoom around cursor<br>"
            "<b>Shift + wheel</b> Horizontal pan<br>"
            "<b>Double click</b> Reset horizontal view"
        )

        QtWidgets.QMessageBox.information(self, "SimView Shortcuts", help_text)

    def toggleMeasureMode(self, checked: object) -> None:
        is_checked = bool(checked)
        block_signals = True
        unblock_signals = False
        if is_checked:
            self.zoomModeButton.blockSignals(block_signals)
            self.zoomModeButton.setChecked(False)
            self.zoomModeButton.blockSignals(unblock_signals)
            self.setInteractionMode("measure")
        elif self.interactionMode == "measure":
            self.setInteractionMode("inspect")

    def toggleZoomMode(self, checked: object) -> None:
        is_checked = bool(checked)
        block_signals = True
        unblock_signals = False
        if is_checked:
            self.measureButton.blockSignals(block_signals)
            self.measureButton.setChecked(False)
            self.measureButton.blockSignals(unblock_signals)
            self.setInteractionMode("zoom")
        elif self.interactionMode == "zoom":
            self.setInteractionMode("inspect")

    def toggleMeasureSnapToEvents(self, checked: object) -> None:
        is_checked = bool(checked)
        self.measureSnapToEvents = is_checked
        self.settings.setValue("measureSnapToEvents", is_checked)
        self.update_status()

    def zero_trajectory_at_cursor(self) -> None:
        if self.currentCursorTime is None:
            dialog.showErrorMessage("Move the cursor over a plot before zeroing trajectory.")
            return
        self.trajectoryZeroReferenceTime = float(self.currentCursorTime)
        self.settings.setValue("trajectoryZeroReferenceTime", self.trajectoryZeroReferenceTime)
        self.apply_trajectory_zero_in_place()
        self.update_status()

    def refocus_trajectory_at_cursor(self) -> None:
        if self.currentCursorTime is None:
            dialog.showErrorMessage("Move the cursor over a plot before adding a 180 refocus flip.")
            return

        refocus_time = float(self.currentCursorTime)
        if not hasattr(self, "trajectoryRefocusTimes"):
            self.trajectoryRefocusTimes = []
        if all(abs(refocus_time - existing_time) > 1e-15 for existing_time in self.trajectoryRefocusTimes):
            self.trajectoryRefocusTimes.append(refocus_time)
            self.trajectoryRefocusTimes.sort()
            if not hasattr(self, "trajectoryFlipSources"):
                self.trajectoryFlipSources = {}
            self.trajectoryFlipSources[refocus_time] = "Manual"
        self.apply_trajectory_zero_in_place()
        self.update_status()

    def reset_trajectory_zero(self) -> None:
        self.trajectoryZeroReferenceTime = None
        self.settings.remove("trajectoryZeroReferenceTime")
        self.apply_trajectory_zero_in_place()
        self.update_status()

    def reset_trajectory_refocuses(self) -> None:
        self.trajectoryRefocusTimes = []
        self.trajectoryFlipSources = {}
        self.apply_trajectory_zero_in_place()
        self.update_status()

    def reset_trajectory_transforms(self) -> None:
        self.trajectoryZeroReferenceTime = None
        self.trajectoryRefocusTimes = []
        self.trajectoryFlipSources = {}
        self.settings.remove("trajectoryZeroReferenceTime")
        self.apply_trajectory_zero_in_place()
        self.update_status()

    def detect_rf_pulse_windows(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.channels:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        pulse_start_times: list[float] = []
        pulse_focus_times: list[float] = []
        for channel in self.channels:
            for line in channel:
                if str(line.get("type", "")).upper() != "NCO":
                    continue
                key = str(line.get("key", "")).lower()
                is_amplitude_key = key == "am" or key.endswith("_am")
                if not is_amplitude_key:
                    continue

                time_values = np.asarray(line.get("t", []), dtype=float)
                data_values = np.asarray(line.get("data", []), dtype=float)
                if time_values.size == 0 or data_values.size == 0:
                    continue
                norm_time, norm_data = self.normalize_time_series(time_values, data_values)
                if norm_time.size == 0 or norm_data.size == 0:
                    continue

                threshold = 1e-12
                active = norm_data > threshold
                rise_indices = np.flatnonzero(active & np.concatenate(([True], ~active[:-1])))
                for rise_index in rise_indices:
                    start_time = float(norm_time[rise_index])
                    pulse_start_times.append(start_time)

                    inactive_after = np.flatnonzero(~active[rise_index + 1 :])
                    if inactive_after.size > 0:
                        end_index = rise_index + 1 + int(inactive_after[0])
                    else:
                        end_index = min(rise_index + 1, norm_time.size - 1)
                    end_time = float(norm_time[end_index])
                    pulse_focus_times.append((start_time + end_time) * 0.5)

        if not pulse_start_times:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        return (
            np.asarray(sorted(set(pulse_start_times)), dtype=float),
            np.asarray(sorted(set(pulse_focus_times)), dtype=float),
        )

    def detect_rf_pulse_starts(self) -> np.ndarray:
        pulse_starts, _pulse_focuses = self.detect_rf_pulse_windows()
        return pulse_starts

    def detect_rf_pulse_descriptors(self) -> list[dict[str, float | str]]:
        descriptors: list[dict[str, float | str]] = []
        for channel in getattr(self, "channels", []):
            for line in channel:
                if str(line.get("type", "")).upper() != "NCO" or str(line.get("key", "")).lower() != "am":
                    continue
                time, amplitude = self.normalize_time_series(
                    np.asarray(line.get("t", []), dtype=float), np.asarray(line.get("data", []), dtype=float),
                )
                active = amplitude > 1e-12
                for start_index in np.flatnonzero(active & np.concatenate(([True], ~active[:-1]))):
                    inactive = np.flatnonzero(~active[start_index + 1 :])
                    end_index = start_index + 1 + int(inactive[0]) if inactive.size else time.size - 1
                    if end_index <= start_index:
                        continue
                    start, end = float(time[start_index]), float(time[end_index])
                    descriptors.append({
                        "start": start, "end": end, "focus": (start + end) * 0.5, "duration": end - start,
                        "area": float(np.trapezoid(amplitude[start_index : end_index + 1], time[start_index : end_index + 1])),
                        "nco": str(line.get("ind", "")),
                    })
        return descriptors

    def learn_hovered_rf_pulse_as_refocus(self) -> None:
        if self.currentCursorTime is None:
            dialog.showErrorMessage("Hover over an RF pulse before classifying it as 180°.")
            return
        pulses = self.detect_rf_pulse_descriptors()
        selected = next((pulse for pulse in pulses if pulse["start"] <= self.currentCursorTime <= pulse["end"]), None)
        if selected is None:
            dialog.showErrorMessage("The cursor is not over a detected RF amplitude pulse.")
            return
        matching = [
            pulse for pulse in pulses
            if pulse["nco"] == selected["nco"]
            and np.isclose(float(pulse["duration"]), float(selected["duration"]), rtol=1e-3, atol=1e-9)
            and np.isclose(abs(float(pulse["area"])), abs(float(selected["area"])), rtol=1e-2, atol=1e-12)
        ]
        if "trajectoryFlipSources" not in self.__dict__:
            self.trajectoryFlipSources = {}
        added = 0
        for pulse in matching:
            focus_time = float(pulse["focus"])
            if all(abs(focus_time - existing) > 1e-15 for existing in self.trajectoryRefocusTimes):
                self.trajectoryRefocusTimes.append(focus_time)
                self.trajectoryFlipSources[focus_time] = "RF 180 match"
                added += 1
        self.trajectoryRefocusTimes.sort()
        self.apply_trajectory_zero_in_place()
        self.update_status()
        self.statusBar().showMessage(f"Added {added} matching 180° RF flip(s).", 4000)

    def apply_calibrated_refocus_flips(self) -> None:
        all_calibrations = list(getattr(self, "rfPulseCalibrations", []))
        calibrations = [
            calibration for calibration in all_calibrations
            if abs(float(calibration.get("flip_angle", 0.0)) - 180.0) <= 5.0
        ]
        if not calibrations:
            return
        if "trajectoryFlipSources" not in self.__dict__:
            self.trajectoryFlipSources = {}
        diffusion_calibration = next(
            (item for item in calibrations if re.search(r"(?:dw|diff)", str(item.get("name", "")), re.IGNORECASE)),
            None,
        )
        rare_calibration = next(
            (
                item for item in calibrations
                if re.search(r"(?:rare|refpulse|refpulse)", str(item.get("name", "")), re.IGNORECASE)
                and item is not diffusion_calibration
            ),
            None,
        )
        excitation_calibrations = [
            item for item in all_calibrations
            if self.is_trajectory_excitation_calibration(item)
        ]
        added = 0
        refocus_index_in_block = 0
        for pulse in sorted(self.detect_rf_pulse_descriptors(), key=lambda item: float(item["focus"])):
            is_excitation = any(
                np.isclose(float(pulse["duration"]), float(calibration["duration"]), rtol=0.03, atol=2e-6)
                for calibration in excitation_calibrations
            )
            if is_excitation:
                refocus_index_in_block = 0
                continue
            matching = next(
                (
                    calibration for calibration in calibrations
                    if np.isclose(float(pulse["duration"]), float(calibration["duration"]), rtol=0.03, atol=2e-6)
                ),
                None,
            )
            if matching is None:
                continue
            # PVM_DwRfcPulse is the diffusion-module refocus immediately
            # following excitation.  The remaining matched pulses in that
            # excitation block form the RARE train and use RefPulse.
            if refocus_index_in_block == 0 and diffusion_calibration is not None:
                matching = diffusion_calibration
            elif refocus_index_in_block > 0 and rare_calibration is not None:
                matching = rare_calibration
            focus_time = float(pulse["focus"])
            if all(abs(focus_time - existing) > 1e-15 for existing in self.trajectoryRefocusTimes):
                self.trajectoryRefocusTimes.append(focus_time)
                self.trajectoryFlipSources[focus_time] = f"Method: {matching['name']} (180°)"
                added += 1
            refocus_index_in_block += 1
        if added:
            self.trajectoryRefocusTimes.sort()
            self.apply_trajectory_zero_in_place()

    def is_rf_amplitude_channel(self, channel: list[dict]) -> bool:
        for line in channel:
            if str(line.get("type", "")).upper() != "NCO":
                continue
            key = str(line.get("key", "")).lower()
            if key == "am" or key.endswith("_am"):
                return True
        return False

    def update_rf_pulse_navigation_state(self) -> None:
        pulse_times, pulse_focus_times = self.detect_rf_pulse_windows()
        self.rfPulseStartTimes = pulse_times
        self.rfPulseFocusTimes = pulse_focus_times
        has_pulses = pulse_times.size > 0

        if hasattr(self, "prevRfPulseButton"):
            self.prevRfPulseButton.setEnabled(has_pulses)
        if hasattr(self, "nextRfPulseButton"):
            self.nextRfPulseButton.setEnabled(has_pulses)
        if hasattr(self, "prevRfPulseAction"):
            self.prevRfPulseAction.setEnabled(has_pulses)
        if hasattr(self, "nextRfPulseAction"):
            self.nextRfPulseAction.setEnabled(has_pulses)

    def add_rf_pulse_focus_markers(self) -> None:
        focus_times = np.asarray(getattr(self, "rfPulseFocusTimes", []), dtype=float)
        if focus_times.size == 0:
            return

        for channel, plot in zip(getattr(self, "channels", []), getattr(self, "plots", []), strict=False):
            if not self.is_rf_amplitude_channel(channel):
                continue
            for focus_time in focus_times:
                plot.add_annotation_marker(float(focus_time), "RF focus", color="m")

    def refresh_trajectory_flip_markers(self) -> None:
        flip_times = sorted(
            {
                float(time_value)
                for time_value in getattr(self, "trajectoryRefocusTimes", [])
                if np.isfinite(float(time_value))
            },
        )
        acquisition_details = self.get_acquisition_window_details()
        acquisition_windows = [(float(item["start"]), float(item["end"])) for item in acquisition_details]
        echo_candidates = self.find_trajectory_echo_candidates()
        spin_echoes = self.find_spin_echo_candidates()
        zero_time = getattr(self, "trajectoryZeroReferenceTime", None)
        for channel, plot in zip(getattr(self, "channels", []), getattr(self, "plots", []), strict=False):
            plot.clear_annotation_markers(group="trajectory_flip")
            plot.clear_annotation_markers(group="trajectory_echo")
            plot.clear_annotation_markers(group="spin_echo")
            plot.clear_annotation_markers(group="trajectory_zero")
            plot.clear_annotation_markers(group="trajectory_excitation_reset")
            plot.clear_overlay_regions(group="acquisition_window")
            if not channel or channel[0].get("chanLabel") not in {
                "Gradient Trajectory", "Effective Gradient", "Effective Trajectory", "Gradient Trajectory Residual", "Coherence Order",
            }:
                continue
            for item in acquisition_details:
                start_time, end_time = float(item["start"]), float(item["end"])
                plot.add_overlay_region(
                    start_time, end_time,
                    color=self.get_acquisition_job_color(str(item.get("job_type", "ADC"))),
                    group="acquisition_window",
                )
            for flip_time in flip_times:
                plot.add_annotation_marker(
                    flip_time,
                    "180° flip",
                    color="#006b6b",
                    group="trajectory_flip",
                )
            if zero_time is not None:
                plot.add_annotation_marker(
                    float(zero_time),
                    "Trajectory zero",
                    color="#7a3e00",
                    group="trajectory_zero",
                )
            source_time = np.asarray(channel[0].get("t", []), dtype=float)
            for excitation_time in self.get_trajectory_excitation_times(source_time):
                if zero_time is not None and abs(float(excitation_time) - float(zero_time)) <= 1e-12:
                    continue
                plot.add_annotation_marker(
                    float(excitation_time),
                    "TR trajectory reset",
                    color="#7a3e00",
                    group="trajectory_excitation_reset",
                )
            for candidate in echo_candidates:
                echo_time = float(candidate["time"])
                in_acquisition = any(start <= echo_time <= end for start, end in acquisition_windows)
                plot.add_annotation_marker(
                    echo_time,
                    "Echo in ADC" if in_acquisition else "Echo",
                    color="#006b3c" if in_acquisition else "#8a6500",
                    group="trajectory_echo",
                )
            for candidate in spin_echoes:
                echo_time = float(candidate["time"])
                in_acquisition = any(start <= echo_time <= end for start, end in acquisition_windows)
                plot.add_annotation_marker(
                    echo_time,
                    "SE in ADC" if in_acquisition else "SE (90°–180°)",
                    color="#7b2cbf" if in_acquisition else "#9c36b5",
                    group="spin_echo",
                )

        self.refresh_trajectory_flip_table()
        self.refresh_coherence_results()

    def refresh_coherence_results(self) -> None:
        legend = self.__dict__.get("acquisitionJobLegendLabel")
        table = self.__dict__.get("echoResultsListWidget")
        details = self.get_acquisition_window_details()
        if legend is not None:
            job_types = sorted({str(item.get("job_type", "ADC")) for item in details})
            legend.setText("Acquisition colors: " + ", ".join(job_types) if job_types else "No acquisition jobs detected.")
        if table is None:
            return
        table.clear()
        for candidate in self.find_spin_echo_candidates():
            time_value = float(candidate["time"])
            job = next(
                (str(item.get("job_type", "ADC")) for item in details if float(item["start"]) <= time_value <= float(item["end"])),
                "outside ADC",
            )
            item = QtWidgets.QListWidgetItem(
                f"SE {self.format_time(time_value)} | {job} | "
                f"90° {self.format_time(float(candidate['excitation_time']))} → "
                f"180° {self.format_time(float(candidate['refocus_time']))}",
            )
            item.setData(Qt.ItemDataRole.UserRole, time_value)
            table.addItem(item)
        for candidate in self.find_trajectory_echo_candidates():
            time_value = float(candidate["time"])
            job = next(
                (str(item.get("job_type", "ADC")) for item in details if float(item["start"]) <= time_value <= float(item["end"])),
                "outside ADC",
            )
            item = QtWidgets.QListWidgetItem(
                f"{self.format_time(time_value)} | {job} | "
                f"K=({candidate['kx']:.3g}, {candidate['ky']:.3g}, {candidate['kz']:.3g}) | "
                f"|K|={candidate['residual']:.3g}",
            )
            item.setData(Qt.ItemDataRole.UserRole, time_value)
            table.addItem(item)

    def find_spin_echo_candidates(self) -> list[dict[str, float]]:
        """Predict Hahn spin echoes produced by detected 90°–180° pairs."""
        channel_times = [
            np.asarray(line.get("t", []), dtype=float)
            for channel in getattr(self, "channels", [])
            for line in channel[:1]
            if np.asarray(line.get("t", []), dtype=float).size
        ]
        if not channel_times:
            return []
        start_time = min(float(time[0]) for time in channel_times)
        end_time = max(float(time[-1]) for time in channel_times)
        excitations = self.get_trajectory_excitation_times(np.asarray([start_time, end_time], dtype=float))
        if excitations.size == 0:
            return []
        refocuses = sorted({
            float(value) for value in getattr(self, "trajectoryRefocusTimes", [])
            if np.isfinite(float(value)) and start_time <= float(value) <= end_time
        })
        candidates: list[dict[str, float]] = []
        paired_excitation_indices: set[int] = set()
        for refocus_time in refocuses:
            excitation_index = int(np.searchsorted(excitations, refocus_time, side="left") - 1)
            if excitation_index < 0 or excitation_index in paired_excitation_indices:
                continue
            paired_excitation_indices.add(excitation_index)
            excitation_time = float(excitations[excitation_index])
            echo_time = 2.0 * refocus_time - excitation_time
            next_excitation = (
                float(excitations[excitation_index + 1])
                if excitation_index + 1 < excitations.size else end_time
            )
            if echo_time < refocus_time or echo_time > next_excitation + 1e-12:
                continue
            candidates.append({
                "time": echo_time,
                "excitation_time": excitation_time,
                "refocus_time": refocus_time,
            })
        return candidates

    def jump_to_echo_result(self, item: QtWidgets.QListWidgetItem) -> None:
        self.jump_to_rf_pulse_time(float(item.data(Qt.ItemDataRole.UserRole)))

    def refresh_trajectory_flip_table(self) -> None:
        table = self.__dict__.get("trajectoryFlipListWidget")
        if table is None:
            return
        table.clear()
        anchor = getattr(self, "trajectoryZeroReferenceTime", None)
        for index, flip_time in enumerate(sorted(getattr(self, "trajectoryRefocusTimes", []))):
            source = getattr(self, "trajectoryFlipSources", {}).get(flip_time, "Manual")
            before = "-1" if (anchor is None or flip_time >= anchor) ^ (index % 2 == 1) else "+1"
            after = "+1" if before == "-1" else "-1"
            item = QtWidgets.QListWidgetItem(
                f"{self.format_time(float(flip_time))}  |  {source}  |  p: {before} → {after}",
            )
            item.setData(Qt.ItemDataRole.UserRole, float(flip_time))
            table.addItem(item)

    def remove_selected_trajectory_flip(self) -> None:
        table = self.__dict__.get("trajectoryFlipListWidget")
        item = table.currentItem() if table is not None else None
        if item is None:
            return
        flip_time = float(item.data(Qt.ItemDataRole.UserRole))
        self.trajectoryRefocusTimes = [time for time in self.trajectoryRefocusTimes if abs(time - flip_time) > 1e-15]
        getattr(self, "trajectoryFlipSources", {}).pop(flip_time, None)
        self.apply_trajectory_zero_in_place()
        self.update_status()

    def move_selected_trajectory_flip(self) -> None:
        table = self.__dict__.get("trajectoryFlipListWidget")
        item = table.currentItem() if table is not None else None
        if item is None:
            return
        old_time = float(item.data(Qt.ItemDataRole.UserRole))
        new_time, accepted = QtWidgets.QInputDialog.getDouble(
            self, "Move 180° Flip", "Time (s)", old_time, decimals=12,
        )
        if not accepted or abs(new_time - old_time) <= 1e-15:
            return
        source = getattr(self, "trajectoryFlipSources", {}).pop(old_time, "Manual")
        self.trajectoryRefocusTimes = [time for time in self.trajectoryRefocusTimes if abs(time - old_time) > 1e-15]
        self.trajectoryRefocusTimes.append(float(new_time))
        self.trajectoryRefocusTimes.sort()
        self.trajectoryFlipSources[float(new_time)] = source
        self.apply_trajectory_zero_in_place()
        self.update_status()

    def detect_acquisition_windows(self) -> list[tuple[float, float]]:
        sample_windows: set[tuple[float, float]] = set()
        rgp_windows: set[tuple[float, float]] = set()
        for channel in getattr(self, "channels", []):
            for line in channel:
                key = str(line.get("key", "")).lower()
                if key not in {"acq", "rgp"}:
                    continue
                time_values, gate_values = self.normalize_time_series(
                    np.asarray(line.get("t", []), dtype=float),
                    np.asarray(line.get("data", []), dtype=float),
                )
                for index in range(max(time_values.size - 1, 0)):
                    if abs(float(gate_values[index])) > 1e-12 and time_values[index + 1] > time_values[index]:
                        target = sample_windows if key == "acq" else rgp_windows
                        target.add((float(time_values[index]), float(time_values[index + 1])))
        return sorted(sample_windows or rgp_windows)

    def get_acquisition_window_details(self) -> list[dict[str, float | str]]:
        metadata = self.__dict__.get("acquisitionWindowDetails", [])
        if metadata:
            return [dict(item) for item in metadata]
        return [
            {"start": start, "end": end, "job_type": "ADC"}
            for start, end in self.detect_acquisition_windows()
        ]

    def get_acquisition_job_color(self, job_type: str) -> tuple[int, int, int, int]:
        palette = ((65, 125, 220, 65), (80, 165, 115, 65), (180, 115, 55, 65), (145, 85, 175, 65))
        index = sum(ord(character) for character in job_type) % len(palette)
        return palette[index]

    def detect_trajectory_echoes(self) -> list[float]:
        return [candidate["time"] for candidate in self.find_trajectory_echo_candidates()]

    def find_trajectory_echo_candidates(self) -> list[dict[str, float]]:
        trajectory_channel = next(
            (
                channel
                for channel in getattr(self, "channels", [])
                if channel and channel[0].get("chanLabel") == "Effective Trajectory"
            ),
            None,
        )
        if trajectory_channel is None:
            trajectory_channel = next(
                (channel for channel in getattr(self, "channels", [])
                 if channel and channel[0].get("chanLabel") == "Gradient Trajectory"),
                None,
            )
        if not trajectory_channel or not getattr(self, "trajectoryRefocusTimes", []):
            return []

        knot_parts = [np.asarray(line.get("t", []), dtype=float) for line in trajectory_channel]
        knot_parts.append(np.asarray(getattr(self, "trajectoryRefocusTimes", []), dtype=float))
        anchor_time = getattr(self, "trajectoryZeroReferenceTime", None)
        if anchor_time is not None:
            knot_parts.append(np.asarray([anchor_time], dtype=float))
        common_time = np.unique(np.concatenate([part for part in knot_parts if part.size])) if any(
            part.size for part in knot_parts
        ) else np.asarray([], dtype=float)
        if common_time.size < 2:
            return []
        components: list[np.ndarray] = []
        for line in trajectory_channel:
            line_time = np.asarray(line.get("t", []), dtype=float)
            line_data = np.asarray(line.get("data", line.get("raw_data", [])), dtype=float)
            if line_time.size and line_data.size:
                components.append(np.interp(common_time, line_time, line_data))
        if not components:
            return []

        trajectory = np.vstack(components).T
        residual = np.linalg.norm(trajectory, axis=1)
        peak = float(np.max(residual))
        tolerance = max(peak * float(self.__dict__.get("trajectoryEchoRelativeTolerance", 1e-2)), 1e-12)
        if anchor_time is None:
            anchor_time = float(common_time[0])
        first_flip = min(float(value) for value in self.trajectoryRefocusTimes)
        candidate_indices: list[tuple[float, np.ndarray]] = []
        for index, interval_duration in enumerate(np.diff(common_time)):
            if interval_duration <= 0:
                continue
            start = trajectory[index]
            delta = trajectory[index + 1] - start
            denominator = float(np.dot(delta, delta))
            fraction = 0.0 if denominator <= 1e-30 else float(np.clip(-np.dot(start, delta) / denominator, 0.0, 1.0))
            moment = start + fraction * delta
            time_value = float(common_time[index] + fraction * interval_duration)
            magnitude = float(np.linalg.norm(moment))
            # An interpolated minimum belongs to this interval only when it is
            # genuinely interior.  Endpoint minima are considered once below as
            # knot-local minima; otherwise every below-threshold sample becomes
            # a duplicate "echo".
            if fraction <= 1e-12 or fraction >= 1.0 - 1e-12:
                continue
            if time_value <= max(float(anchor_time), first_flip) or magnitude > tolerance:
                continue
            candidate_indices.append((time_value, moment))

        # Include minima which occur exactly at a trajectory knot.  The strict
        # comparison on either side suppresses flat, sub-threshold plateaus.
        minimum_time = max(float(anchor_time), first_flip)
        for index in range(1, common_time.size - 1):
            if common_time[index] <= minimum_time or residual[index] > tolerance:
                continue
            if residual[index] <= residual[index - 1] and residual[index] <= residual[index + 1] and (
                residual[index] < residual[index - 1] or residual[index] < residual[index + 1]
            ):
                candidate_indices.append((float(common_time[index]), trajectory[index]))

        candidates: list[dict[str, float]] = []
        for time_value, moment in sorted(candidate_indices, key=lambda item: item[0]):
            if candidates and abs(time_value - candidates[-1]["time"]) <= 1e-12:
                continue
            candidates.append({
                "time": time_value,
                "kx": float(moment[0]) if moment.size > 0 else 0.0,
                "ky": float(moment[1]) if moment.size > 1 else 0.0,
                "kz": float(moment[2]) if moment.size > 2 else 0.0,
                "residual": float(np.linalg.norm(moment)),
            })
        return candidates

    def get_trajectory_moment_summary(self, time_value: float | None) -> str:
        if time_value is None:
            return "K: -"
        trajectory = next(
            (channel for channel in getattr(self, "channels", []) if channel and channel[0].get("chanLabel") == "Effective Trajectory"),
            [],
        )
        if not trajectory:
            trajectory = next(
                (channel for channel in getattr(self, "channels", [])
                 if channel and channel[0].get("chanLabel") == "Gradient Trajectory"),
                [],
            )
        values: list[float] = []
        labels: list[str] = []
        for line in trajectory:
            time = np.asarray(line.get("t", []), dtype=float)
            data = np.asarray(line.get("data", []), dtype=float)
            if time.size and data.size:
                value = float(np.interp(time_value, time, data))
                values.append(value)
                labels.append(f"{line.get('key', '?')}={value:.4g}")
        return f"K: {', '.join(labels)}, |K|={np.linalg.norm(values):.4g}" if values else "K: -"

    def get_b_matrix_summary(self, time_value: float | None) -> str:
        matrix = self.compute_b_matrix_at_time(time_value)
        if matrix is None:
            return "b: - (gradient calibration required)"
        rows = [",".join(f"{value:.3g}" for value in row) for row in matrix]
        return "b [s/mm²]: [" + "; ".join(rows) + "]"

    def refresh_b_matrix_display(self, time_value: float | None) -> None:
        labels = self.__dict__.get("bMatrixValueLabels")
        if not labels:
            return
        matrix = self.compute_b_matrix_at_time(time_value)
        cursor_label = self.__dict__.get("bMatrixCursorLabel")
        units_label = self.__dict__.get("bMatrixUnitsLabel")
        if matrix is None:
            for row in labels:
                for label in row:
                    label.setText("—")
            if cursor_label is not None:
                cursor_label.setText(
                    "Move the cursor over a plot" if time_value is None else "Gradient calibration required"
                )
            if units_label is not None:
                units_label.setText("s/mm²")
            return
        for row_index, row in enumerate(labels):
            for column_index, label in enumerate(row):
                label.setText(f"{matrix[row_index, column_index]:.4g}")
        if cursor_label is not None:
            cursor_label.setText(f"t = {self.format_time(time_value)}")
        if units_label is not None:
            units_label.setText("s/mm²")

    def jump_to_rf_pulse_time(self, target_time: float) -> None:
        if not self.plots:
            return
        target = float(target_time)
        self.tPos = target
        self.updateView()
        for plot in self.plots:
            plot.cursor_line.setPos(target)
        self.update_status(cursor_time=target)

    def jump_to_next_rf_pulse(self) -> None:
        pulse_times = np.asarray(getattr(self, "rfPulseStartTimes", []), dtype=float)
        if pulse_times.size == 0:
            dialog.showErrorMessage("No RF pulse starts were detected in the loaded channels.")
            return

        cursor_time = self.currentCursorTime
        if cursor_time is None:
            target_index = 0
        else:
            target_index = int(np.searchsorted(pulse_times, float(cursor_time) + 1e-15, side="right"))
            target_index = min(target_index, pulse_times.size - 1)
        self.jump_to_rf_pulse_time(float(pulse_times[target_index]))

    def jump_to_previous_rf_pulse(self) -> None:
        pulse_times = np.asarray(getattr(self, "rfPulseStartTimes", []), dtype=float)
        if pulse_times.size == 0:
            dialog.showErrorMessage("No RF pulse starts were detected in the loaded channels.")
            return

        cursor_time = self.currentCursorTime
        if cursor_time is None:
            target_index = pulse_times.size - 1
        else:
            target_index = int(np.searchsorted(pulse_times, float(cursor_time) - 1e-15, side="left") - 1)
            target_index = max(target_index, 0)
        self.jump_to_rf_pulse_time(float(pulse_times[target_index]))

    def setInteractionMode(self, mode: str) -> None:
        self.interactionMode = mode
        self.measurement_start_x = None
        self.measurement_source_plot = None
        for plot in self.plots:
            plot.set_interaction_mode(mode)
        self.update_status()

    def open_folder(self) -> None:
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Simulation output folder",
            self.dataPath,
            QtWidgets.QFileDialog.Option.ShowDirsOnly | QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if not folder_path:
            return

        self.dataPath = folder_path
        if os.path.exists(self.dataPath + "/" + "pulse_seq.json"):
            self.settings.setValue("lastFolder", self.dataPath)
            self.loadData()
            return

        if not os.path.exists(self.dataPath + "/" + "_GCube.xml"):
            dialog.showErrorMessage("No GCube file found in folder!")
            return

        if not os.path.exists(self.dataPath + "/" + "_FCube1.xml"):
            dialog.showErrorMessage("No FCube file found in folder!")
            return

        self.settings.setValue("lastFolder", self.dataPath)
        self.loadData()

    def calculate_max_gradient_strength_mt_per_m(
        self,
        calibration_hz_per_mm: float,
        gamma_mhz_per_t: float,
    ) -> float | None:
        if calibration_hz_per_mm <= 0 or gamma_mhz_per_t <= 0:
            return None
        return calibration_hz_per_mm / gamma_mhz_per_t

    def update_scanner_settings_display(self) -> None:
        max_gradient_mt_per_m = self.calculate_max_gradient_strength_mt_per_m(
            float(self.gradientCalibrationSpinBox.value()),
            float(self.nucleusGammaSpinBox.value()),
        )
        if max_gradient_mt_per_m is None:
            self.maxGradientStrengthValue.setText("-")
        else:
            self.maxGradientStrengthValue.setText(f"{max_gradient_mt_per_m:.3f} mT/m")
        if hasattr(self, "refresh_channel_checkbox_labels"):
            self.refresh_channel_checkbox_labels()

    def reload_current_data(self) -> None:
        if self.dataPath is not None:
            self.loadData()
        elif self.inlineData is not None:
            self.loadData(self.inlineData)

    def apply_scanner_settings(self) -> None:
        self.themeMode = self.themeModeComboBox.currentData()
        self.gradientCalibrationHzPerMm = float(self.gradientCalibrationSpinBox.value())
        self.nucleusGammaMHzPerT = float(self.nucleusGammaSpinBox.value())
        self.brukerPwReferenceWatts = float(self.brukerPwReferenceSpinBox.value())
        self.gradientDisplayUnits = str(self.gradientDisplayUnitsComboBox.currentData() or "hz_per_mm")
        self.splitGradientChannels = bool(self.splitGradientChannelsCheckBox.isChecked())
        self.derivedSignalStartupPadding = float(self.derivedSignalStartupPaddingSpinBox.value())
        self.trajectoryEchoRelativeTolerance = float(self.trajectoryEchoToleranceSpinBox.value()) / 100.0
        self.trajectoryRefocusFlipAngleDegrees = float(self.trajectoryRefocusFlipAngleSpinBox.value())
        self.settings.setValue("themeMode", self.themeMode)
        self.settings.setValue("gradientCalibrationHzPerMm", self.gradientCalibrationHzPerMm)
        self.settings.setValue("nucleusGammaMHzPerT", self.nucleusGammaMHzPerT)
        self.settings.setValue("brukerPwReferenceWatts", self.brukerPwReferenceWatts)
        self.settings.setValue("gradientDisplayUnits", self.gradientDisplayUnits)
        self.settings.setValue("splitGradientChannels", self.splitGradientChannels)
        self.settings.setValue("derivedSignalStartupPadding", self.derivedSignalStartupPadding)
        self.settings.setValue("trajectoryEchoRelativeTolerance", self.trajectoryEchoRelativeTolerance)
        self.settings.setValue("trajectoryRefocusFlipAngleDegrees", self.trajectoryRefocusFlipAngleDegrees)
        self.apply_theme_settings()
        self.update_existing_plot_themes()
        self.update_scanner_settings_display()
        if self.channels:
            self.refresh_trajectory_flip_markers()
        if self.channels:
            self.selectedChannels = [
                str(getattr(check_box, "channel_key", check_box.text()))
                for check_box in self.checkBoxes
                if check_box.isChecked()
            ]
            # A saved channel selection otherwise keeps a newly relevant
            # branching plot hidden after this settings-triggered reload.
            if not np.isclose(self.trajectoryRefocusFlipAngleDegrees, 180.0, atol=1e-9):
                pathway_channel_key = "Imperfect RF Pathway Weights"
                if pathway_channel_key not in self.selectedChannels:
                    self.selectedChannels.append(pathway_channel_key)
            self.reload_current_data()
        self.update_status()

    def zoomIn(self) -> None:
        self.windowWidth = max(self.get_min_zoom_width(), self.windowWidth * 0.8)
        self.updateView()

    def zoomOut(self) -> None:
        self.windowWidth = min(self.tMax - self.tMin, self.windowWidth / 0.8)
        self.updateView()

    def autoscale_visible_y(self) -> None:
        if not self.plots or not self.channels:
            return

        for index, plot in enumerate(self.plots):
            if index >= len(self.channels):
                continue
            if index < len(self.plotContainers) and not self.plotContainers[index].isVisible():
                continue

            x_min, x_max = plot.viewRange()[0]
            finite_segments: list[np.ndarray] = []
            for line in self.channels[index]:
                t_vals = np.asarray(line.get("t", []), dtype=float)
                y_vals = np.asarray(line.get("data", []), dtype=float)
                if t_vals.size == 0 or y_vals.size == 0:
                    continue
                n = min(t_vals.size, y_vals.size)
                t_vals = t_vals[:n]
                y_vals = y_vals[:n]
                if n == 0:
                    continue

                left = int(np.searchsorted(t_vals, x_min, side="left"))
                right = int(np.searchsorted(t_vals, x_max, side="right"))
                if right <= left:
                    continue
                visible = y_vals[left:right]
                finite = visible[np.isfinite(visible)]
                if finite.size > 0:
                    finite_segments.append(finite)

            if not finite_segments:
                continue

            y_min = min(float(np.min(values)) for values in finite_segments)
            y_max = max(float(np.max(values)) for values in finite_segments)
            if np.isclose(y_min, y_max):
                padding = max(abs(y_min) * 0.05, 1.0)
            else:
                padding = max((y_max - y_min) * 0.05, 1e-12)
            plot.setYRange(y_min - padding, y_max + padding, padding=0)

    def zoom_to_cursor(self, cursor_time: float, zoom_factor: float) -> None:
        if self.tMax <= self.tMin:
            return

        old_width = self.windowWidth
        min_width = self.get_min_zoom_width()
        self.windowWidth = min(max(old_width * zoom_factor, min_width), self.tMax - self.tMin)

        if old_width <= 0:
            self.updateView()
            return

        relative_position = (cursor_time - (self.tPos - old_width * 0.5)) / old_width
        relative_position = min(max(relative_position, 0.0), 1.0)
        self.tPos = cursor_time - (relative_position - 0.5) * self.windowWidth
        self.updateView()

    def pan_horizontally(self, delta_time: float) -> None:
        if delta_time == 0:
            return
        self.tPos += delta_time
        self.updateView()

    def changeXRange(self) -> None:
        self.tPos = self.tSlider.value() * self.sliderScaler
        self.updateView()

    def jumpXPos(self) -> None:
        self.tPos = np.minimum(self.tMax, self.tPos + self.windowWidth * 0.5)
        self.tSlider.setValue(int(self.tPos / self.sliderScaler))
        self.updateView()

    def jumpXNeg(self) -> None:
        self.tPos = np.maximum(self.tMin, self.tPos - self.windowWidth * 0.5)
        self.tSlider.setValue(int(self.tPos / self.sliderScaler))
        self.updateView()

    def updateView(self) -> None:
        self.windowWidth = min(max(self.windowWidth, self.get_min_zoom_width()), self.tMax - self.tMin)
        half_width = self.windowWidth * 0.5
        self.tPos = min(max(self.tPos, self.tMin + half_width), self.tMax - half_width)
        rangePos = self.tPos + half_width
        rangeNeg = self.tPos - half_width

        for plot in self.plots:
            plot.setXRange(rangeNeg, rangePos, padding=0)

        block_signals = True
        unblock_signals = False
        self.tSlider.blockSignals(block_signals)
        if self.sliderScaler > 0:
            self.tSlider.setValue(int(self.tPos / self.sliderScaler))
        self.tSlider.blockSignals(unblock_signals)
        self.settings.setValue("tPos", self.tPos)
        self.settings.setValue("windowWidth", self.windowWidth)
        self.update_status()

    def resetView(self) -> None:
        self.windowWidth = self.tMax - self.tMin
        self.tPos = (self.tMax + self.tMin) / 2
        self.updateView()

    def format_time(self, dt_seconds: float | None) -> str:
        if dt_seconds is None:
            return "-"
        if dt_seconds >= 1:
            return f"{dt_seconds:.3f} s"
        if dt_seconds >= 1e-3:
            return f"{dt_seconds * 1e3:.3f} ms"
        if dt_seconds >= 1e-6:
            return f"{dt_seconds * 1e6:.3f} us"
        return f"{dt_seconds * 1e9:.3f} ns"

    def get_pulse_program_location(self, cursor_time: float | None) -> str:
        if cursor_time is None:
            return "-"

        timeline = getattr(self, "pulseProgramTimeline", None)
        if timeline is None:
            return "-"

        if not isinstance(timeline, tuple) or len(timeline) != 2:
            return "-"

        times, line_numbers = timeline
        if times is None or line_numbers is None:
            return "-"
        if len(times) == 0 or len(line_numbers) == 0:
            return "-"

        index = int(np.searchsorted(times, cursor_time, side="right") - 1)
        if index < 0:
            return "-"

        line_number = int(line_numbers[min(index, len(line_numbers) - 1)])
        mapping = getattr(self, "pulseProgramLineMapping", {})
        mapped = mapping.get(line_number, {})

        source_name = mapped.get("source")
        source_line = mapped.get("line")
        if source_name is not None and source_line is not None:
            return f"{source_name}:{source_line} (ln {line_number})"
        if source_name is not None:
            return f"{source_name} (ln {line_number})"
        return f"ln {line_number}"

    def get_pulse_program_line_number(self, cursor_time: float | None) -> int | None:
        if cursor_time is None:
            return None

        timeline = getattr(self, "pulseProgramTimeline", None)
        if timeline is None or not isinstance(timeline, tuple) or len(timeline) != 2:
            return None

        times, line_numbers = timeline
        if times is None or line_numbers is None:
            return None
        if len(times) == 0 or len(line_numbers) == 0:
            return None

        index = int(np.searchsorted(times, cursor_time, side="right") - 1)
        if index < 0:
            return None
        return int(line_numbers[min(index, len(line_numbers) - 1)])

    def get_pulse_program_jump_targets(self) -> list[tuple[str, float, int]]:
        timeline = getattr(self, "pulseProgramTimeline", None)
        if timeline is None or not isinstance(timeline, tuple) or len(timeline) != 2:
            return []
        times, line_numbers = timeline
        if times is None or line_numbers is None:
            return []
        if len(times) == 0 or len(line_numbers) == 0:
            return []

        mapping = getattr(self, "pulseProgramLineMapping", {})
        seen_lines: set[int] = set()
        targets: list[tuple[str, float, int]] = []
        for time_value, line_number_value in zip(times, line_numbers, strict=False):
            line_number = int(line_number_value)
            if line_number in seen_lines:
                continue
            seen_lines.add(line_number)
            mapped = mapping.get(line_number, {})
            source_name = mapped.get("source")
            source_line = mapped.get("line")
            if source_name is not None and source_line is not None:
                location_text = f"{source_name}:{source_line} (ln {line_number})"
            elif source_name is not None:
                location_text = f"{source_name} (ln {line_number})"
            else:
                location_text = f"ln {line_number}"

            target_time = float(time_value)
            label = f"{location_text} @ {self.format_time(target_time)}"
            targets.append((label, target_time, line_number))
        return targets

    def jump_to_pulse_program_time(self, target_time: float) -> None:
        if not self.plots:
            return
        self.tPos = float(target_time)
        self.updateView()
        for plot in self.plots:
            plot.cursor_line.setPos(self.tPos)
        self.update_status(cursor_time=self.tPos)

    def jump_to_ppg_line(self) -> None:
        targets = self.get_pulse_program_jump_targets()
        if not targets:
            dialog.showErrorMessage("No pulse-program line mapping is available for this dataset.")
            return

        current_line_number = self.get_pulse_program_line_number(self.currentCursorTime)
        current_index = 0
        if current_line_number is not None:
            for idx, (_label, _time_value, line_number) in enumerate(targets):
                if line_number == current_line_number:
                    current_index = idx
                    break

        labels = [item[0] for item in targets]
        selected_label, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Jump To PPG Line",
            "Pulse program location:",
            labels,
            current_index,
            False,
        )
        if not accepted:
            return

        selected_index = labels.index(selected_label)
        _, target_time, _line_number = targets[selected_index]
        self.jump_to_pulse_program_time(target_time)

    def update_status(
        self,
        cursor_time: float | None | object = _UNSET,
        measurement: float | None | object = _UNSET,
    ) -> None:
        if cursor_time is not _UNSET:
            self.currentCursorTime = cursor_time
        if measurement is not _UNSET:
            self.currentMeasurement = measurement

        mode_text = self.interactionMode.capitalize()
        span_text = self.format_time(getattr(self, "windowWidth", None))
        cursor_text = self.format_time(self.currentCursorTime)
        measurement_text = self.format_time(self.currentMeasurement)
        snap_text = "On" if self.measureSnapToEvents else "Off"
        refocus_count = len(getattr(self, "trajectoryRefocusTimes", []))
        pulse_program_text = self.get_pulse_program_location(self.currentCursorTime)
        moment_text = self.get_trajectory_moment_summary(self.currentCursorTime)
        self.refresh_b_matrix_display(self.currentCursorTime)
        self.statusBar().showMessage(
            " | ".join(
                (
                    f"Mode: {mode_text}",
                    f"Snap: {snap_text}",
                    f"View width: {span_text}",
                    f"Cursor: {cursor_text}",
                    f"Measurement: {measurement_text}",
                    f"180 flips: {refocus_count}",
                    moment_text,
                    f"PPG: {pulse_program_text}",
                ),
            ),
        )
