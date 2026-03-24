from pathlib import Path
import sys

import pyqtgraph as pg
from PyQt6 import QtGui, QtWidgets, uic
from PyQt6.QtCore import QDir, QSettings, Qt
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QApplication, QMainWindow, QSizePolicy, QVBoxLayout

from simview_app.calculations import CalculationMixin
from simview_app.constants import PROTON_GAMMA_MHZ_PER_T
from simview_app.data_loading import DataLoadingMixin
from simview_app.exporting import ExportMixin
from simview_app.interaction import InteractionMixin
from widgets.mulitPlotCursor import CursorPlot


class GUIapp(
    InteractionMixin,
    CalculationMixin,
    ExportMixin,
    DataLoadingMixin,
    QMainWindow,
):
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
        self.measureSnapToEvents = False
        self.gradientCalibrationHzPerMm = 0.0
        self.nucleusGammaMHzPerT = PROTON_GAMMA_MHZ_PER_T
        self.displayGradientsInMtPerM = False
        self.useOpenGLAcceleration = False
        self.trajectoryZeroReferenceTime: float | None = None
        self.derivedSignalStartupPadding = 1.0
        self.inlineData = data

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
        self.jumpPButton.setFixedHeight(50)
        self.jumpNButton.setFixedHeight(50)
        self.jumpNButton.clicked.connect(self.jumpXNeg)
        self.jumpPButton.clicked.connect(self.jumpXPos)
        self.tSlider.valueChanged.connect(self.changeXRange)
        self.tSlider.setMinimum(0)
        self.tSlider.setMaximum(10000)
        self.tSlider.setStyleSheet("""
            QSlider::handle:horizontal {
                width: 30px;
                height: 30px;
                background: #3498db;
                border: 1px solid #2980b9;
                border-radius: 5px;
                margin: -5px 0;
            }
            QSlider::groove:horizontal {
                height: 10px;
                background: #bdc3c7;
                border-radius: 5px;
            }
        """)

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

        menubar = self.menuBar()
        fileMenu = menubar.addMenu("File")
        viewMenu = menubar.addMenu("View")
        helpMenu = menubar.addMenu("Help")

        openFolderAction = QtGui.QAction("Open Folder", self)
        openFolderAction.triggered.connect(self.open_folder)
        fileMenu.addAction(openFolderAction)

        exportPlotsAction = QtGui.QAction("Export Visible Plots", self)
        exportPlotsAction.triggered.connect(self.export_visible_plots)
        fileMenu.addAction(exportPlotsAction)

        self.snapMeasureAction = QtGui.QAction("Stick Measurements To Events", self)
        self.snapMeasureAction.setCheckable(True)
        self.snapMeasureAction.setChecked(self.measureSnapToEvents)
        self.snapMeasureAction.toggled.connect(self.toggleMeasureSnapToEvents)
        viewMenu.addAction(self.snapMeasureAction)

        self.zeroTrajectoryAtCursorAction = QtGui.QAction("Zero Trajectory At Cursor", self)
        self.zeroTrajectoryAtCursorAction.triggered.connect(self.zero_trajectory_at_cursor)
        viewMenu.addAction(self.zeroTrajectoryAtCursorAction)

        self.resetTrajectoryZeroAction = QtGui.QAction("Reset Trajectory Zero", self)
        self.resetTrajectoryZeroAction.triggered.connect(self.reset_trajectory_zero)
        viewMenu.addAction(self.resetTrajectoryZeroAction)

        shortcutsHelpAction = QtGui.QAction("Shortcuts", self)
        shortcutsHelpAction.setShortcut(QtGui.QKeySequence(Qt.Key.Key_F1))
        shortcutsHelpAction.triggered.connect(self.showShortcutsHelp)
        helpMenu.addAction(shortcutsHelpAction)

        self.settings = QSettings("MR_ISIBrno", "BrukerSimView")
        self.measureSnapToEvents = bool(self.settings.value("measureSnapToEvents", False, type=bool))
        self.snapMeasureAction.setChecked(self.measureSnapToEvents)
        self.gradientCalibrationHzPerMm = float(self.settings.value("gradientCalibrationHzPerMm", 0.0, type=float))
        self.nucleusGammaMHzPerT = float(self.settings.value("nucleusGammaMHzPerT", PROTON_GAMMA_MHZ_PER_T, type=float))
        self.displayGradientsInMtPerM = bool(self.settings.value("displayGradientsInMtPerM", False, type=bool))
        self.useOpenGLAcceleration = bool(self.settings.value("useOpenGLAcceleration", False, type=bool))
        self.derivedSignalStartupPadding = float(self.settings.value("derivedSignalStartupPadding", 1.0, type=float))
        stored_trajectory_zero = self.settings.value("trajectoryZeroReferenceTime", None)
        self.trajectoryZeroReferenceTime = float(stored_trajectory_zero) if stored_trajectory_zero not in {None, ""} else None

        self.selectedChannels = self.settings.value("selectedChannels", [])
        if not isinstance(self.selectedChannels, list):
            self.selectedChannels = [self.selectedChannels] if self.selectedChannels else []

        self.sidePanel = QtWidgets.QWidget()
        self.sidePanel.setMinimumWidth(180)
        self.sidePanel.setMaximumWidth(320)
        self.sidePanel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.sidePanelLayout = QVBoxLayout(self.sidePanel)
        self.sideTabs = QtWidgets.QTabWidget()

        self.channelsTab = QtWidgets.QWidget()
        self.channelsLayout = QVBoxLayout(self.channelsTab)

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

        self.channelsLayout.addWidget(self.channelFilter)
        self.channelsLayout.addLayout(self.channelButtonLayout)
        self.channelsLayout.addWidget(self.channelScrollArea)

        self.settingsTab = QtWidgets.QWidget()
        self.settingsLayout = QVBoxLayout(self.settingsTab)
        self.scannerSettingsHint = QtWidgets.QLabel(
            "Gradient channels are loaded in percent. Enter the scanner calibration at 100% to scale them to Hz/mm."
        )
        self.scannerSettingsHint.setWordWrap(True)
        self.gradientCalibrationSpinBox = QtWidgets.QDoubleSpinBox()
        self.gradientCalibrationSpinBox.setDecimals(3)
        self.gradientCalibrationSpinBox.setRange(0.0, 1_000_000.0)
        self.gradientCalibrationSpinBox.setSingleStep(1.0)
        self.gradientCalibrationSpinBox.setSuffix(" Hz/mm @ 100%")
        self.gradientCalibrationSpinBox.setValue(self.gradientCalibrationHzPerMm)
        self.nucleusGammaSpinBox = QtWidgets.QDoubleSpinBox()
        self.nucleusGammaSpinBox.setDecimals(6)
        self.nucleusGammaSpinBox.setRange(0.001, 1_000.0)
        self.nucleusGammaSpinBox.setSingleStep(0.1)
        self.nucleusGammaSpinBox.setSuffix(" MHz/T")
        self.nucleusGammaSpinBox.setValue(self.nucleusGammaMHzPerT)
        self.maxGradientStrengthValue = QtWidgets.QLineEdit()
        self.maxGradientStrengthValue.setReadOnly(True)
        self.derivedSignalStartupPaddingSpinBox = QtWidgets.QDoubleSpinBox()
        self.derivedSignalStartupPaddingSpinBox.setDecimals(3)
        self.derivedSignalStartupPaddingSpinBox.setRange(0.0, 10_000.0)
        self.derivedSignalStartupPaddingSpinBox.setSingleStep(0.1)
        self.derivedSignalStartupPaddingSpinBox.setSuffix(" s")
        self.derivedSignalStartupPaddingSpinBox.setValue(self.derivedSignalStartupPadding)
        self.displayGradientsInMtPerMCheckBox = QtWidgets.QCheckBox("Display physical gradients in mT/m")
        self.displayGradientsInMtPerMCheckBox.setChecked(self.displayGradientsInMtPerM)
        self.useOpenGLAccelerationCheckBox = QtWidgets.QCheckBox("Use OpenGL acceleration (experimental)")
        self.useOpenGLAccelerationCheckBox.setChecked(self.useOpenGLAcceleration)
        self.applyScannerSettingsButton = QtWidgets.QPushButton("Apply Scanner Settings")
        self.applyScannerSettingsButton.clicked.connect(self.apply_scanner_settings)
        self.settingsFormLayout = QtWidgets.QFormLayout()
        self.settingsFormLayout.addRow("Grad Calibration", self.gradientCalibrationSpinBox)
        self.settingsFormLayout.addRow("Nucleus Gamma", self.nucleusGammaSpinBox)
        self.settingsFormLayout.addRow("Max Grad @ 100%", self.maxGradientStrengthValue)
        self.settingsFormLayout.addRow("Startup Padding", self.derivedSignalStartupPaddingSpinBox)
        self.settingsLayout.addWidget(self.scannerSettingsHint)
        self.settingsLayout.addLayout(self.settingsFormLayout)
        self.settingsLayout.addWidget(self.displayGradientsInMtPerMCheckBox)
        self.settingsLayout.addWidget(self.useOpenGLAccelerationCheckBox)
        self.settingsLayout.addWidget(self.applyScannerSettingsButton)
        self.settingsLayout.addStretch(1)
        self.gradientCalibrationSpinBox.valueChanged.connect(self.update_scanner_settings_display)
        self.nucleusGammaSpinBox.valueChanged.connect(self.update_scanner_settings_display)
        self.update_scanner_settings_display()

        self.sideTabs.addTab(self.channelsTab, "Channels")
        self.sideTabs.addTab(self.settingsTab, "Settings")
        self.sidePanelLayout.addWidget(self.sideTabs)
        self.sidePanel.hide()

        self.horizontalLayout_2.insertWidget(0, self.sidePanel)
        self.horizontalLayout_2.setStretch(0, 0)
        self.horizontalLayout_2.setStretch(1, 1)

        qt_app = QApplication.instance()
        assert qt_app is not None
        palette = qt_app.palette()
        base_color = palette.color(QPalette.ColorRole.Base)
        self.darkMode = base_color.value() < 128

        if self.darkMode:
            pg.setConfigOption("background", "black")
            pg.setConfigOption("foreground", "white")
        else:
            pg.setConfigOption("background", "white")
            pg.setConfigOption("foreground", "black")
        pg.setConfigOption("useOpenGL", self.useOpenGLAcceleration)

        if self.dataPath is None:
            self.dataPath = self.settings.value("lastFolder", QDir.homePath())
        else:
            self.loadData()

        if data is not None:
            self.loadData(data)

        self.registerShortcuts()
        self.update_status()


def show_graphs_from_dict(data: dict) -> None:
    app = QApplication(sys.argv)
    gui = GUIapp(data=data)
    gui.show()
    app.exec()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    app = QApplication(sys.argv)
    gui = GUIapp(path)
    gui.show()
    app.exec()
