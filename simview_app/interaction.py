import os

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QDir, Qt

from utils import dialog

from .constants import _UNSET


class InteractionMixin:
    def activate_measure(self) -> None:
        self.setInteractionMode("measure")

    def registerShortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Open, self, activated=self.open_folder)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Plus), self, activated=self.zoomIn)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Minus), self, activated=self.zoomOut)
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, activated=self.resetView)
        QtGui.QShortcut(QtGui.QKeySequence("M"), self, activated=self.measureButton.toggle)
        QtGui.QShortcut(QtGui.QKeySequence("Z"), self, activated=self.zoomModeButton.toggle)
        QtGui.QShortcut(QtGui.QKeySequence("E"), self, activated=self.snapMeasureAction.toggle)
        QtGui.QShortcut(QtGui.QKeySequence("T"), self, activated=self.zeroTrajectoryAtCursorAction.trigger)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Left), self, activated=self.jumpXNeg)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Right), self, activated=self.jumpXPos)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_F1), self, activated=self.showShortcutsHelp)

    def showShortcutsHelp(self) -> None:
        help_text = (
            "<b>Keyboard shortcuts</b><br><br>"
            "<b>Ctrl+O</b> Open folder<br>"
            "<b>+</b> Zoom in<br>"
            "<b>-</b> Zoom out<br>"
            "<b>Left / Right</b> Pan backward / forward<br>"
            "<b>R</b> Reset full view<br>"
            "<b>M</b> Toggle measure mode<br>"
            "<b>Z</b> Toggle zoom mode<br>"
            "<b>E</b> Toggle measure snap to events<br>"
            "<b>T</b> Zero trajectory at cursor<br>"
            "<b>F1</b> Show this help<br><br>"
            "<b>Mouse controls</b><br><br>"
            "<b>Move mouse</b> Inspect synced cursor across plots<br>"
            "<b>Measure mode</b> Click once to start, click again to finish<br>"
            "<b>Zoom mode</b> Click-drag to zoom into a region<br>"
            "<b>Shift + drag</b> Temporary zoom without switching modes"
        )

        QtWidgets.QMessageBox.information(self, "SimView Shortcuts", help_text)

    def toggleMeasureMode(self, checked: bool) -> None:
        if checked:
            self.zoomModeButton.blockSignals(True)  # noqa: FBT003
            self.zoomModeButton.setChecked(False)
            self.zoomModeButton.blockSignals(False)  # noqa: FBT003
            self.setInteractionMode("measure")
        elif self.interactionMode == "measure":
            self.setInteractionMode("inspect")

    def toggleZoomMode(self, checked: bool) -> None:
        if checked:
            self.measureButton.blockSignals(True)  # noqa: FBT003
            self.measureButton.setChecked(False)
            self.measureButton.blockSignals(False)  # noqa: FBT003
            self.setInteractionMode("zoom")
        elif self.interactionMode == "zoom":
            self.setInteractionMode("inspect")

    def toggleMeasureSnapToEvents(self, checked: bool) -> None:
        self.measureSnapToEvents = checked
        self.settings.setValue("measureSnapToEvents", checked)
        self.update_status()

    def zero_trajectory_at_cursor(self) -> None:
        if self.currentCursorTime is None:
            dialog.showErrorMessage("Move the cursor over a plot before zeroing trajectory.")
            return
        self.trajectoryZeroReferenceTime = float(self.currentCursorTime)
        self.settings.setValue("trajectoryZeroReferenceTime", self.trajectoryZeroReferenceTime)
        self.apply_trajectory_zero_in_place()
        self.update_status()

    def reset_trajectory_zero(self) -> None:
        self.trajectoryZeroReferenceTime = None
        self.settings.remove("trajectoryZeroReferenceTime")
        self.apply_trajectory_zero_in_place()
        self.update_status()

    def setInteractionMode(self, mode: str) -> None:
        self.interactionMode = mode
        for plot in self.plots:
            plot.set_interaction_mode(mode)
        self.update_status()

    def open_folder(self) -> None:
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Simulation output folder", self.dataPath)
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
            self.maxGradientStrengthValue.setText(f"{max_gradient_mt_per_m:.6f} mT/m")

    def rebuild_loaded_channels(self) -> None:
        if not self.channels:
            return
        self.channels = [
            channel
            for channel in self.channels
            if channel[0].get("type") not in {"grads_derived", "nco_derived"}
        ]
        self.update_gradient_channels()
        self.channels.extend(self.build_gradient_derived_channels())
        self.channels.extend(self.build_nco_power_derived_channels())

    def reload_current_data(self) -> None:
        if self.dataPath is not None:
            self.loadData()
        elif self.inlineData is not None:
            self.loadData(self.inlineData)

    def apply_scanner_settings(self) -> None:
        self.gradientCalibrationHzPerMm = float(self.gradientCalibrationSpinBox.value())
        self.nucleusGammaMHzPerT = float(self.nucleusGammaSpinBox.value())
        self.displayGradientsInMtPerM = self.displayGradientsInMtPerMCheckBox.isChecked()
        self.useOpenGLAcceleration = self.useOpenGLAccelerationCheckBox.isChecked()
        self.derivedSignalStartupPadding = float(self.derivedSignalStartupPaddingSpinBox.value())
        self.settings.setValue("gradientCalibrationHzPerMm", self.gradientCalibrationHzPerMm)
        self.settings.setValue("nucleusGammaMHzPerT", self.nucleusGammaMHzPerT)
        self.settings.setValue("displayGradientsInMtPerM", self.displayGradientsInMtPerM)
        self.settings.setValue("useOpenGLAcceleration", self.useOpenGLAcceleration)
        self.settings.setValue("derivedSignalStartupPadding", self.derivedSignalStartupPadding)
        pg.setConfigOption("useOpenGL", self.useOpenGLAcceleration)
        self.update_scanner_settings_display()
        if self.channels:
            self.selectedChannels = [check_box.text() for check_box in self.checkBoxes if check_box.isChecked()]
            self.reload_current_data()
        self.update_status()

    def zoomIn(self) -> None:
        self.windowWidth = max((self.tMax - self.tMin) / self.tSlider.maximum(), self.windowWidth * 0.8)
        self.updateView()

    def zoomOut(self) -> None:
        self.windowWidth = min(self.tMax - self.tMin, self.windowWidth / 0.8)
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
        self.windowWidth = min(max(self.windowWidth, self.sliderScaler), self.tMax - self.tMin)
        half_width = self.windowWidth * 0.5
        self.tPos = min(max(self.tPos, self.tMin + half_width), self.tMax - half_width)
        rangePos = self.tPos + half_width
        rangeNeg = self.tPos - half_width

        for plot in self.plots:
            plot.setXRange(rangeNeg, rangePos)

        self.tSlider.blockSignals(True)  # noqa: FBT003
        if self.sliderScaler > 0:
            self.tSlider.setValue(int(self.tPos / self.sliderScaler))
        self.tSlider.blockSignals(False)  # noqa: FBT003
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
        self.statusBar().showMessage(
            f"Mode: {mode_text} | Snap: {snap_text} | View width: {span_text} | Cursor: {cursor_text} | Measurement: {measurement_text}"
        )
