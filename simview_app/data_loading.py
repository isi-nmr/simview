import os

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from utils import multiplot
from utils.simUtilsBrkr import readBrkrChannels
from utils.simUtilsNMRScopeB import readNMRScopeBChannels
from widgets.mulitPlotCursor import CursorPlot


class DataLoadingMixin:
    def is_zero_channel(self, channel: list[dict]) -> bool:
        for line in channel:
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

    def loadData(self, data: str | None = None) -> None:
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
                self.channels = readBrkrChannels(self.dataPath, progress, self)
        else:
            self.channels = readNMRScopeBChannels(data, progress, self)

        self.update_gradient_channels()
        self.channels.extend(self.build_gradient_derived_channels())
        self.channels.extend(self.build_nco_power_derived_channels())
        self.channels = [channel for channel in self.channels if not self.is_zero_channel(channel)]

        if not self.channels:
            self.sidePanel.hide()
            progress.close()
            return

        self.tMax = 0
        for channel in self.channels:
            self.tMax = np.maximum(self.tMax, channel[0]["t"][-1].item())

        self.tMin = 0
        self.sliderScaler = max((self.tMax - self.tMin) / self.tSlider.maximum(), 1e-12)

        self.registerCheckBoxes()
        self.sidePanel.show()

        self.tPos = float(self.settings.value("tPos", self.tMax / 2))
        self.windowWidth = float(self.settings.value("windowWidth", self.tMax / 2))

        self.initPlots()
        progress.setValue(90)
        progress.close()

        for plot in self.plots:
            plot.showCursor()
            plot.set_interaction_mode(self.interactionMode)
        self.update_status(cursor_time=None, measurement=self.currentMeasurement)

    def registerCheckBoxes(self) -> None:
        if hasattr(self, "channelListLayout"):
            while self.channelListLayout.count():
                item = self.channelListLayout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        for channel in self.channels:
            checkBox = QtWidgets.QCheckBox()
            checkBox.setText(channel[0]["chanLabel"])
            default_show = channel[0].get("show", True)
            if self.selectedChannels:
                default_show = channel[0]["chanLabel"] in self.selectedChannels
            checkBox.setChecked(default_show)
            checkBox.stateChanged.connect(self.checkBoxChanged)
            self.channelListLayout.addWidget(checkBox)
            self.checkBoxes.append(checkBox)

    def update_channel_checkbox_state(self, checkBox: QtWidgets.QCheckBox) -> None:
        if not hasattr(checkBox, "contID"):
            return

        channel_name = checkBox.text()
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
        block_signals = True
        unblock_signals = False

        for chanInd, channel in enumerate(self.channels):
            currentPlot = CursorPlot(darkMode=self.darkMode)
            phaseContainer = QWidget()
            phaseContainer_layout = QVBoxLayout(phaseContainer)
            phaseContainer_layout.setContentsMargins(0, 0, 0, 0)
            phaseContainer_layout.addWidget(currentPlot)
            vb = currentPlot.getViewBox()
            vb.setMouseEnabled(x=False, y=True)
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

            if channel[0]["plotType"] == "phase":
                if channel[0].get("units", "deg") != "rad":
                    currentPlot.setYRange(0, 360)
                else:
                    currentPlot.setYRange(-np.pi, np.pi)

            if len(channel) > 1:
                currentPlot.addLegend(offset=(10, 10))

            axis_label, axis_units = self.get_channel_axis_label(channel)
            currentPlot.setLabel("right", axis_label, units=axis_units)

            if self.selectedChannels != []:
                if channel[0]["chanLabel"] in self.selectedChannels:
                    phaseContainer.show()
                    self.checkBoxes[chanInd].blockSignals(block_signals)
                    self.checkBoxes[chanInd].setChecked(True)
                    self.checkBoxes[chanInd].blockSignals(unblock_signals)
                else:
                    phaseContainer.hide()
                    self.checkBoxes[chanInd].blockSignals(block_signals)
                    self.checkBoxes[chanInd].setChecked(False)
                    self.checkBoxes[chanInd].blockSignals(unblock_signals)

            for line in channel:
                plot_data = line if line.get("drawStyle") == "line" else multiplot.convertToStep(line, "data")
                currentPen = line.get("pen", pens[chanInd % len(pens)])

                if type(currentPen) is str:
                    currentPen = penDict[currentPen]

                currentPlot.add_managed_curve(plot_data["t"], plot_data["data"], name=line["label"], pen=currentPen)

                if len(line["annotations"]) > 0:
                    multiplot.addAnnotations(line, currentPlot)

        self.updateView()
        self.filterChannels()
        self.imageLayout.addLayout(self.navigation)
