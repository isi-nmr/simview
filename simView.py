import os
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtGui, QtWidgets, uic
from PyQt6.QtCore import QCoreApplication, QDir, QSettings, Qt
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QApplication, QMainWindow, QSizePolicy, QVBoxLayout, QWidget

from utils import dialog, multiplot
from utils.simUtilsBrkr import readBrkrChannels
from utils.simUtilsNMRScopeB import readNMRScopeBChannels
from widgets.mulitPlotCursor import CursorPlot

_UNSET = object()


class GUIapp(QMainWindow):
    windowWidth = 1e-2
    sliderScaler = 1

    highlightRect = None

    darkMode = False

    def __init__(self, simPath: str | None = None, data: dict | None = None) -> None:
        super().__init__()

        self.plots: list[CursorPlot] = []
        self.channels = []
        self.checkBoxes: list[QtWidgets.QCheckBox] = []
        self.plotContainers = []
        self.selectedChannels = []
        self.interactionMode = "inspect"
        self.currentMeasurement = None
        self.currentCursorTime = None

        path = Path(__file__).resolve().parent / "visusimForm.ui"
        uic.loadUi(path, self)

        self.dataPath = simPath
        self.setMouseTracking(True)

        self.tPos = self.windowWidth * 0.5

        self.tSlider = QtWidgets.QSlider(Qt.Orientation.Horizontal)

        self.jumpPButton = QtWidgets.QPushButton()
        self.jumpNButton = QtWidgets.QPushButton()

        self.jumpPButton.setText("->")
        self.jumpNButton.setText("<-")

        self.jumpPButton.setFixedHeight(50)  # pixels
        self.jumpNButton.setFixedHeight(50)

        self.jumpNButton.clicked.connect(self.jumpXNeg)
        self.jumpPButton.clicked.connect(self.jumpXPos)

        self.tSlider.valueChanged.connect(self.changeXRange)
        self.tSlider.setMinimum(0)
        self.tSlider.setMaximum(10000)
        # Correct stylesheet for horizontal slider
        self.tSlider.setStyleSheet("""
            QSlider::handle:horizontal {
                width: 30px;       /* handle width */
                height: 30px;      /* handle height */
                background: #3498db;
                border: 1px solid #2980b9;
                border-radius: 5px;
                margin: -5px 0;    /* center handle on groove */
            }
            QSlider::groove:horizontal {
                height: 10px;      /* groove thickness */
                background: #bdc3c7;
                border-radius: 5px;
            }
        """)

        # Zoom buttons
        self.zoomInButton = QtWidgets.QPushButton("Zoom +")
        self.zoomOutButton = QtWidgets.QPushButton("Zoom -")
        self.resetViewButton = QtWidgets.QPushButton("Reset View")
        self.zoomModeButton = QtWidgets.QPushButton("Zoom Mode")
        self.zoomModeButton.setCheckable(True)
        self.zoomInButton.setFixedHeight(50)
        self.zoomOutButton.setFixedHeight(50)
        self.resetViewButton.setFixedHeight(50)
        self.zoomModeButton.setFixedHeight(50)
        self.zoomInButton.clicked.connect(self.zoomIn)
        self.zoomOutButton.clicked.connect(self.zoomOut)
        self.resetViewButton.clicked.connect(self.resetView)
        self.zoomModeButton.toggled.connect(self.toggleZoomMode)

        self.measureButton = QtWidgets.QPushButton("Measure")
        self.measureButton.setCheckable(True)
        self.measureButton.toggled.connect(self.toggleMeasureMode)

        # Layout
        buttonLayout = QtWidgets.QHBoxLayout()
        buttonLayout.addWidget(self.jumpNButton)
        buttonLayout.addWidget(self.zoomOutButton)
        buttonLayout.addWidget(self.zoomInButton)
        buttonLayout.addWidget(self.resetViewButton)
        buttonLayout.addWidget(self.jumpPButton)
        buttonLayout.addWidget(self.zoomModeButton)
        buttonLayout.addWidget(self.measureButton)
        self.measureButton.setFixedHeight(50)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(buttonLayout)
        layout.addWidget(self.tSlider)
        self.navigation = layout

        # Menu bar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("File")
        helpMenu = menubar.addMenu("Help")

        # Add "Open Folder" action
        openFolderAction = QtGui.QAction("Open Folder", self)
        openFolderAction.triggered.connect(self.open_folder)
        fileMenu.addAction(openFolderAction)

        shortcutsHelpAction = QtGui.QAction("Shortcuts", self)
        shortcutsHelpAction.setShortcut(QtGui.QKeySequence(Qt.Key.Key_F1))
        shortcutsHelpAction.triggered.connect(self.showShortcutsHelp)
        helpMenu.addAction(shortcutsHelpAction)

        # Create settings object
        self.settings = QSettings("MR_ISIBrno", "BrukerSimView")

        self.selectedChannels = self.settings.value("selectedChannels", [])
        if not isinstance(self.selectedChannels, list):
            self.selectedChannels = [self.selectedChannels] if self.selectedChannels else []

        self.sidePanel = QtWidgets.QWidget()
        self.sidePanel.setMinimumWidth(180)
        self.sidePanel.setMaximumWidth(240)
        self.sidePanel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.leftMenu = QVBoxLayout(self.sidePanel)

        self.channelFilter = QtWidgets.QLineEdit()
        self.channelFilter.setPlaceholderText("Filter channels...")
        self.channelFilter.textChanged.connect(self.filterChannels)

        self.showAllButton = QtWidgets.QPushButton("Show All")
        self.hideAllButton = QtWidgets.QPushButton("Hide All")
        self.showAllButton.clicked.connect(self.showAllChannels)
        self.hideAllButton.clicked.connect(self.hideAllChannels)

        self.channelButtonLayout = QtWidgets.QHBoxLayout()
        self.channelButtonLayout.addWidget(self.showAllButton)
        self.channelButtonLayout.addWidget(self.hideAllButton)

        self.channelListWidget = QtWidgets.QWidget()
        self.channelListLayout = QVBoxLayout(self.channelListWidget)
        self.channelListLayout.setContentsMargins(0, 0, 0, 0)
        self.channelListLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.channelScrollArea = QtWidgets.QScrollArea()
        self.channelScrollArea.setWidgetResizable(True)
        self.channelScrollArea.setWidget(self.channelListWidget)

        self.leftMenu.addWidget(self.channelFilter)
        self.leftMenu.addLayout(self.channelButtonLayout)
        self.leftMenu.addWidget(self.channelScrollArea)
        self.sidePanel.hide()

        self.horizontalLayout_2.insertWidget(0, self.sidePanel)
        self.horizontalLayout_2.setStretch(0, 0)
        self.horizontalLayout_2.setStretch(1, 1)

        palette = app.palette()
        base_color = palette.color(QPalette.ColorRole.Base)
        self.darkMode = base_color.value() < 128  # value() gives brightness (0-255)

        if self.darkMode:
            pg.setConfigOption("background", "black")
            pg.setConfigOption("foreground", "white")
        else:
            pg.setConfigOption("background", "white")
            pg.setConfigOption("foreground", "black")

        # Read a value (with a default)
        if self.dataPath is None:
            self.dataPath = self.settings.value("lastFolder", QDir.homePath())
        else:
            self.loadData()

        if data is not None:
            self.loadData(data)

        self.registerShortcuts()
        self.update_status()

    def activate_measure(self) -> None:
        self.setInteractionMode("measure")

    def registerShortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Open, self, activated=self.open_folder)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Plus), self, activated=self.zoomIn)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Minus), self, activated=self.zoomOut)
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, activated=self.resetView)
        QtGui.QShortcut(QtGui.QKeySequence("M"), self, activated=self.measureButton.toggle)
        QtGui.QShortcut(QtGui.QKeySequence("Z"), self, activated=self.zoomModeButton.toggle)
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
            # NMRScopeBPath
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
        self.statusBar().showMessage(
            f"Mode: {mode_text} | View width: {span_text} | Cursor: {cursor_text} | Measurement: {measurement_text}"
        )

    def resetApp(self) -> None:
        if self.plotContainers:
            for container in self.plotContainers:
                # Remove widget from layout
                self.imageLayout.removeWidget(container)

                # Delete the widget properly
                container.setParent(None)
                container.deleteLater()

        self.plotContainers = []

        self.plots = []
        self.checkBoxes = []
        self.channels = []

    def loadData(self, data: str | None = None) -> None:
        self.resetApp()

        progress = QtWidgets.QProgressDialog("Parsing simulation data ...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Please Wait")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)  # show immediately
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
            default_show = channel[0].get("show", True) and np.sum(np.abs(channel[0]["data"])) > 0
            if self.selectedChannels:
                default_show = channel[0]["chanLabel"] in self.selectedChannels
            checkBox.setChecked(default_show)
            checkBox.stateChanged.connect(self.checkBoxChanged)
            self.channelListLayout.addWidget(checkBox)
            self.checkBoxes.append(checkBox)

    def checkBoxChanged(self) -> None:
        for checkBox in self.checkBoxes:
            if not checkBox.isChecked():
                if checkBox.text() in self.selectedChannels:
                    self.selectedChannels.remove(checkBox.text())
                self.plotContainers[checkBox.contID].hide()
            else:
                if checkBox.text() not in self.selectedChannels:
                    self.selectedChannels.append(checkBox.text())
                self.plotContainers[checkBox.contID].show()

        self.settings.setValue("selectedChannels", self.selectedChannels)
        self.filterChannels()
        self.update_status()

    def filterChannels(self) -> None:
        filter_text = self.channelFilter.text().strip().lower()
        for checkBox in self.checkBoxes:
            matches = filter_text in checkBox.text().lower()
            checkBox.setVisible(matches)

    def setAllChannels(self, visible: bool) -> None:
        for checkBox in self.checkBoxes:
            if not checkBox.isVisible():
                continue
            checkBox.setChecked(visible)

    def showAllChannels(self) -> None:
        self.setAllChannels(True)

    def hideAllChannels(self) -> None:
        self.setAllChannels(False)

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
            self.plots.append(currentPlot)
            self.plotContainers.append(phaseContainer)
            self.imageLayout.addWidget(phaseContainer, stretch=1)

            plotItem = currentPlot.getPlotItem()
            plotItem.showAxis("left", show=True)
            plotItem.showAxis("right", show=True)
            axis = currentPlot.getPlotItem().getAxis("right")
            axis.setWidth(60)  # fixed width in pixels
            axis = currentPlot.getPlotItem().getAxis("left")
            axis.setWidth(60)  # fixed width in pixels

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

            currentPlot.setLabel("right", channel[0]["chanLabel"])

            if np.sum(np.abs(channel[0]["data"])) == 0:
                phaseContainer.hide()
                self.checkBoxes[chanInd].blockSignals(True)  # noqa: FBT003
                self.checkBoxes[chanInd].setChecked(False)
                self.checkBoxes[chanInd].blockSignals(False)  # noqa: FBT003

            if self.selectedChannels != []:
                if channel[0]["chanLabel"] in self.selectedChannels:
                    phaseContainer.show()
                    self.checkBoxes[chanInd].blockSignals(True)  # noqa: FBT003
                    self.checkBoxes[chanInd].setChecked(True)
                    self.checkBoxes[chanInd].blockSignals(False)  # noqa: FBT003
                else:
                    phaseContainer.hide()
                    self.checkBoxes[chanInd].blockSignals(True)  # noqa: FBT003
                    self.checkBoxes[chanInd].setChecked(False)
                    self.checkBoxes[chanInd].blockSignals(False)  # noqa: FBT003

            for _, line in enumerate(channel):
                stepData = multiplot.convertToStep(line, "data")
                currentPen = line.get("pen", pens[chanInd % len(pens)])

                if type(currentPen) is str:
                    currentPen = penDict[currentPen]

                plot_curve = currentPlot.plot(
                    stepData["t"],
                    stepData["data"],
                    name=line["label"],
                    pen=currentPen,
                )
                plot_curve.setClipToView(True)
                plot_curve.setSkipFiniteCheck(True)
                currentPlot.register_curve(line["label"], stepData["t"], stepData["data"])

                if len(line["annotations"]) > 0:
                    multiplot.addAnnotations(line, currentPlot)

        self.updateView()
        self.filterChannels()

        self.imageLayout.addLayout(self.navigation)


def show_graphs_from_dict(data: dict) -> None:
    app = QApplication(sys.argv)  # ✅ Must be first
    gui = GUIapp(data=data)  # ✅ Now it's safe
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    inputArgs = sys.argv

    path = sys.argv[1] if len(sys.argv) > 1 else None

    app = QApplication(sys.argv)  # ✅ Must be first
    gui = GUIapp(path)  # ✅ Now it's safe
    gui.show()
    sys.exit(app.exec())
