import os

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import QCoreApplication, QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from utils import multiplot
from utils.simUtilsBrkr import parseBrkrChannels
from utils.simUtilsNMRScopeB import readNMRScopeBChannels
from widgets.mulitPlotCursor import CursorPlot


class BrukerLoadWorker(QObject):
    finished = pyqtSignal(int, object, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, int, str)

    def __init__(self, path: str, load_id: int) -> None:
        super().__init__()
        self.path = path
        self.load_id = load_id

    def run(self) -> None:
        try:
            def emit_progress(value: int, label: str) -> None:
                self.progress.emit(self.load_id, value, label)

            channels, info = parseBrkrChannels(self.path, progress_callback=emit_progress)
            self.finished.emit(self.load_id, channels, info)
        except Exception as exc:
            self.failed.emit(self.load_id, str(exc))


class DataLoadingMixin:
    def get_channel_checkbox_key(self, channel: list[dict]) -> str:
        return str(channel[0].get("chanLabel", ""))

    def format_channel_checkbox_label(self, channel: list[dict]) -> str:
        return self.get_channel_checkbox_key(channel)

    def refresh_channel_checkbox_labels(self) -> None:
        for index, check_box in enumerate(self.checkBoxes):
            if index >= len(self.channels):
                break
            check_box.setText(self.format_channel_checkbox_label(self.channels[index]))

    def compute_gradient_sample_ticks(self, channel: list[dict]) -> np.ndarray:
        grad_lines = [line for line in channel if line.get("type") == "grads"]
        if not grad_lines:
            return np.asarray([], dtype=float)

        base_t = np.asarray(grad_lines[0].get("t", []), dtype=float)
        if base_t.size == 0:
            return np.asarray([], dtype=float)

        aligned = True
        for line in grad_lines[1:]:
            line_t = np.asarray(line.get("t", []), dtype=float)
            if line_t.shape != base_t.shape or not np.array_equal(line_t, base_t):
                aligned = False
                break

        if aligned:
            ticks = base_t
        else:
            collected: list[np.ndarray] = []
            for line in grad_lines:
                t = np.asarray(line.get("t", []), dtype=float)
                if t.size == 0:
                    continue
                collected.append(t)
            if not collected:
                return np.asarray([], dtype=float)
            ticks = np.unique(np.concatenate(collected))

        if ticks.size > 2_000_000_000:
            step = int(np.ceil(ticks.size / 2_000_000_000))
            ticks = ticks[::step]
        return ticks

    def get_initial_y_range(self, channel: list[dict]) -> tuple[float, float] | None:
        if channel[0]["plotType"] == "phase":
            if channel[0].get("units", "deg") != "rad":
                return 0.0, 360.0
            return -float(np.pi), float(np.pi)

        finite_segments: list[np.ndarray] = []
        for line in channel:
            values = np.asarray(line.get("data", []), dtype=float)
            if values.size == 0:
                continue
            finite_values = values[np.isfinite(values)]
            if finite_values.size > 0:
                finite_segments.append(finite_values)

        if not finite_segments:
            return None

        y_min = min(float(np.min(values)) for values in finite_segments)
        y_max = max(float(np.max(values)) for values in finite_segments)
        if np.isclose(y_min, y_max):
            padding = max(abs(y_min) * 0.05, 1.0)
            return y_min - padding, y_max + padding

        padding = (y_max - y_min) * 0.05
        return y_min - padding, y_max + padding

    def is_zero_channel(self, channel: list[dict]) -> bool:
        for line in channel:
            if line.get("annotations"):
                return False
            values = np.asarray(line.get("data", []), dtype=float)
            if values.size == 0:
                continue

            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                continue

            if not np.allclose(finite_values, 0.0, atol=1e-15, rtol=0.0):
                return False

        return True

    def resetApp(self) -> None:
        if self.plotContainers:
            for container in self.plotContainers:
                self.imageLayout.removeWidget(container)
                container.setParent(None)
                container.deleteLater()

        self.plotContainers = []
        self.plots = []
        self.checkBoxes = []
        self.channels = []
        self.measurements = []
        if hasattr(self, "refresh_measurements_list"):
            self.refresh_measurements_list()
        self.pulseProgramSource = None
        self.pulseProgramTimeline = None
        self.pulseProgramLineMapping = {}
        self.rfPulseStartTimes = None
        if hasattr(self, "prevRfPulseButton"):
            self.prevRfPulseButton.setEnabled(False)
        if hasattr(self, "nextRfPulseButton"):
            self.nextRfPulseButton.setEnabled(False)
        if hasattr(self, "prevRfPulseAction"):
            self.prevRfPulseAction.setEnabled(False)
        if hasattr(self, "nextRfPulseAction"):
            self.nextRfPulseAction.setEnabled(False)

    def loadData(self, data: str | None = None) -> None:
        self._loadRequestId = getattr(self, "_loadRequestId", 0) + 1
        load_id = self._loadRequestId
        self.resetApp()
        if data is not None:
            self.inlineData = data

        progress = QtWidgets.QProgressDialog("Parsing simulation data ...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Please Wait")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.show()
        progress.setValue(5)
        QCoreApplication.processEvents()

        if data is None:
            if os.path.exists(self.dataPath + "/" + "pulse_seq.json"):
                self.channels = readNMRScopeBChannels(self.dataPath, progress, self)
            else:
                if os.path.exists(self.dataPath + "/mrScanSim"):
                    self.dataPath = self.dataPath + "/mrScanSim"
                self._start_bruker_load(path=self.dataPath, progress=progress, load_id=load_id)
                return
        else:
            self.channels = readNMRScopeBChannels(data, progress, self)

        self._complete_channel_load(progress=progress)

    def _complete_channel_load(self, progress: QtWidgets.QProgressDialog) -> None:
        self.update_gradient_channels()
        self.channels.extend(self.build_gradient_derived_channels())
        self.channels.extend(self.build_nco_power_derived_channels())
        self.channels = [channel for channel in self.channels if not self.is_zero_channel(channel)]

        if not self.channels:
            self.set_sidebar_available(False)
            progress.close()
            return

        self.tMax = 0
        for channel in self.channels:
            self.tMax = np.maximum(self.tMax, channel[0]["t"][-1].item())

        self.tMin = 0
        self.sliderScaler = max((self.tMax - self.tMin) / self.tSlider.maximum(), 1e-12)

        self.registerCheckBoxes()
        self.set_sidebar_available(True)

        self.tPos = float(self.settings.value("tPos", self.tMax / 2))
        self.windowWidth = float(self.settings.value("windowWidth", self.tMax / 2))

        self.initPlots()
        progress.setValue(90)
        progress.close()

        for plot in self.plots:
            plot.showCursor()
            plot.set_interaction_mode(self.interactionMode)
        self.update_rf_pulse_navigation_state()
        self.update_status(cursor_time=None, measurement=self.currentMeasurement)

    def _start_bruker_load(self, path: str, progress: QtWidgets.QProgressDialog, load_id: int) -> None:
        progress.setLabelText("Starting Bruker parsing...")
        progress.setRange(0, 100)
        progress.setValue(10)
        progress.setCancelButton(None)
        self._loadProgress = progress

        self._brukerLoadThread = QThread(self)
        self._brukerLoadWorker = BrukerLoadWorker(path, load_id)
        self._brukerLoadWorker.moveToThread(self._brukerLoadThread)

        self._brukerLoadThread.started.connect(self._brukerLoadWorker.run)
        self._brukerLoadWorker.progress.connect(self._on_bruker_load_progress)
        self._brukerLoadWorker.finished.connect(self._on_bruker_load_finished)
        self._brukerLoadWorker.failed.connect(self._on_bruker_load_failed)
        self._brukerLoadWorker.finished.connect(self._brukerLoadThread.quit)
        self._brukerLoadWorker.failed.connect(self._brukerLoadThread.quit)
        self._brukerLoadThread.finished.connect(self._brukerLoadWorker.deleteLater)
        self._brukerLoadThread.finished.connect(self._brukerLoadThread.deleteLater)
        self._brukerLoadThread.start()

    def _on_bruker_load_progress(self, load_id: int, value: int, label: str) -> None:
        if load_id != getattr(self, "_loadRequestId", -1):
            return

        progress = getattr(self, "_loadProgress", None)
        if progress is None:
            return

        progress.setValue(value)
        progress.setLabelText(label)

    def _on_bruker_load_finished(self, load_id: int, channels: list[list[dict]], info: dict[str, object]) -> None:
        if load_id != getattr(self, "_loadRequestId", -1):
            return

        progress = getattr(self, "_loadProgress", None)
        if progress is None:
            return

        self.channels = channels
        source = str(info.get("pulProg", ""))
        self.pulseProgramSource = source or None
        self.pulseProgramTimeline = (
            info.get("lineEventTimes"),
            info.get("lineEventNumbers"),
        )
        self.pulseProgramLineMapping = info.get("lineMapping", {})
        if source:
            self.setWindowTitle(f"{self.dataPath} originPPG: {source}")

        self._complete_channel_load(progress=progress)
        self._loadProgress = None

    def _on_bruker_load_failed(self, load_id: int, error: str) -> None:
        if load_id != getattr(self, "_loadRequestId", -1):
            return

        progress = getattr(self, "_loadProgress", None)
        if progress is not None:
            progress.close()
            self._loadProgress = None

        QtWidgets.QMessageBox.critical(
            self,
            "Bruker Load Failed",
            f"Could not parse Bruker files.\n\n{error}",
        )

    def registerCheckBoxes(self) -> None:
        if hasattr(self, "channelListLayout"):
            while self.channelListLayout.count():
                item = self.channelListLayout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        for channel in self.channels:
            checkBox = QtWidgets.QCheckBox()
            channel_key = self.get_channel_checkbox_key(channel)
            checkBox.channel_key = channel_key
            checkBox.setText(self.format_channel_checkbox_label(channel))
            default_show = channel[0].get("show", True)
            if self.selectedChannels:
                default_show = channel_key in self.selectedChannels
            checkBox.setChecked(default_show)
            checkBox.stateChanged.connect(self.checkBoxChanged)
            self.channelListLayout.addWidget(checkBox)
            self.checkBoxes.append(checkBox)

    def update_channel_checkbox_state(self, checkBox: QtWidgets.QCheckBox) -> None:
        if not hasattr(checkBox, "contID"):
            return

        channel_name = str(getattr(checkBox, "channel_key", checkBox.text()))
        is_checked = checkBox.isChecked()
        if is_checked:
            if channel_name not in self.selectedChannels:
                self.selectedChannels.append(channel_name)
        elif channel_name in self.selectedChannels:
            self.selectedChannels.remove(channel_name)

        container = self.plotContainers[checkBox.contID]
        container.setVisible(is_checked)
        if is_checked and checkBox.contID < len(self.plots):
            self.plots[checkBox.contID].schedule_curve_refresh()

    def checkBoxChanged(self) -> None:
        check_box = self.sender()
        if isinstance(check_box, QtWidgets.QCheckBox):
            self.update_channel_checkbox_state(check_box)
        self.settings.setValue("selectedChannels", self.selectedChannels)
        self.update_status()

    def filterChannels(self) -> None:
        filter_text = self.channelFilter.text().strip().lower()
        for checkBox in self.checkBoxes:
            matches = filter_text in checkBox.text().lower()
            checkBox.setVisible(matches)

    def setAllChannels(self, *, visible: bool) -> None:
        changed = False
        block_signals = True
        unblock_signals = False
        self.sidePanel.setUpdatesEnabled(False)
        for checkBox in self.checkBoxes:
            if not checkBox.isVisible():
                continue
            if checkBox.isChecked() == visible:
                continue
            checkBox.blockSignals(block_signals)
            checkBox.setChecked(visible)
            checkBox.blockSignals(unblock_signals)
            self.update_channel_checkbox_state(checkBox)
            changed = True
        self.sidePanel.setUpdatesEnabled(True)
        if changed:
            self.settings.setValue("selectedChannels", self.selectedChannels)
            self.update_status()

    def showAllChannels(self) -> None:
        self.setAllChannels(visible=True)

    def hideAllChannels(self) -> None:
        self.setAllChannels(visible=False)

    def initPlots(self) -> None:
        self.imageLayout.removeItem(self.navigation)

        pens, penDict = multiplot.makePens()

        for chanInd, channel in enumerate(self.channels):
            currentPlot = CursorPlot(darkMode=self.darkMode)
            phaseContainer = QWidget()
            phaseContainer_layout = QVBoxLayout(phaseContainer)
            phaseContainer_layout.setContentsMargins(0, 0, 0, 0)
            phaseContainer_layout.addWidget(currentPlot)
            vb = currentPlot.getViewBox()
            vb.setMouseEnabled(x=False, y=True)
            vb.enableAutoRange(axis=vb.YAxis, enable=False)
            vb.setAutoVisible(y=False)
            self.plots.append(currentPlot)
            self.plotContainers.append(phaseContainer)
            self.imageLayout.addWidget(phaseContainer, stretch=1)

            plotItem = currentPlot.getPlotItem()
            plotItem.showAxis("left", show=True)
            plotItem.showAxis("right", show=True)
            axis = currentPlot.getPlotItem().getAxis("right")
            axis.setWidth(60)
            axis.enableAutoSIPrefix(enable=False)
            axis = currentPlot.getPlotItem().getAxis("left")
            axis.setWidth(60)
            axis.enableAutoSIPrefix(enable=False)

            self.checkBoxes[chanInd].contID = len(self.plotContainers) - 1

            if chanInd >= len(self.channels) - 1:
                currentPlot.setLabel("bottom", "Time (s)")

            initial_y_range = self.get_initial_y_range(channel)
            if initial_y_range is not None:
                currentPlot.setYRange(*initial_y_range, padding=0)

            if len(channel) > 1:
                currentPlot.addLegend(offset=(10, 10))

            axis_label, axis_units = self.get_channel_axis_label(channel)
            currentPlot.setLabel("right", axis_label, units=axis_units)

            # Always sync the initial plot visibility with the checkbox state.
            # This keeps channels with show=False hidden when no saved selection exists.
            phaseContainer.setVisible(self.checkBoxes[chanInd].isChecked())

            for line in channel:
                plot_data = line if line.get("drawStyle") == "line" else multiplot.convertToStep(line, "data")
                currentPen = line.get("pen", pens[chanInd % len(pens)])

                if type(currentPen) is str:
                    currentPen = penDict[currentPen]

                currentPlot.add_managed_curve(plot_data["t"], plot_data["data"], name=line["label"], pen=currentPen)

                if len(line["annotations"]) > 0:
                    multiplot.addAnnotations(line, currentPlot)

            if channel and channel[0].get("type") == "grads":
                tick_times = self.compute_gradient_sample_ticks(channel)
                if tick_times.size > 0:
                    currentPlot.add_change_ticks(tick_times, color=(220, 20, 20, 110))

        self.updateView()
        self.filterChannels()
        self.imageLayout.addLayout(self.navigation)
