import sys
import os
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6 import QtGui, QtWidgets, uic
from PyQt6.QtCore import QDir, QSettings, Qt
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QSizePolicy, QVBoxLayout

from simview_app.calculations import CalculationMixin
from simview_app.constants import PROTON_GAMMA_MHZ_PER_T
from simview_app.data_loading import DataLoadingMixin
from simview_app.exporting import ExportMixin
from simview_app.interaction import InteractionMixin

if TYPE_CHECKING:
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
        self.measurements: list[dict[str, float | str]] = []
        self.currentCursorTime = None
        self.measureSnapToEvents = False
        self.measurement_start_x: float | None = None
        self.measurement_source_plot: CursorPlot | None = None
        self.pulseProgramSource = None
        self.pulseProgramTimeline = None
        self.pulseProgramLineMapping = {}
        self.rfPulseStartTimes = None
        self.gradientCalibrationHzPerMm = 0.0
        self.nucleusGammaMHzPerT = PROTON_GAMMA_MHZ_PER_T
        self.gradientDisplayUnits = "hz_per_mm"
        self.brukerPwReferenceWatts = 1.0
        self.themeMode = "system"
        self.trajectoryZeroReferenceTime: float | None = None
        self.derivedSignalStartupPadding = 0.010
        self.inlineData = data

        path = Path(__file__).resolve().parent / "visusimForm.ui"
        uic.loadUi(path, self)

        self.dataPath = simPath
        self.setMouseTracking(True)

        self.tPos = self.windowWidth * 0.5

        self.tSlider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.jumpPButton = QtWidgets.QPushButton()
        self.jumpNButton = QtWidgets.QPushButton()
        self.prevRfPulseButton = QtWidgets.QPushButton("Prev RF")
        self.nextRfPulseButton = QtWidgets.QPushButton("Next RF")
        self.jumpPButton.setText("->")
        self.jumpNButton.setText("<-")
        self.jumpPButton.setFixedHeight(50)
        self.jumpNButton.setFixedHeight(50)
        self.prevRfPulseButton.setFixedHeight(50)
        self.nextRfPulseButton.setFixedHeight(50)
        self.prevRfPulseButton.setEnabled(False)
        self.nextRfPulseButton.setEnabled(False)
        self.jumpNButton.clicked.connect(self.jumpXNeg)
        self.jumpPButton.clicked.connect(self.jumpXPos)
        self.prevRfPulseButton.clicked.connect(self.jump_to_previous_rf_pulse)
        self.nextRfPulseButton.clicked.connect(self.jump_to_next_rf_pulse)
        self.tSlider.valueChanged.connect(self.changeXRange)
        self.tSlider.setMinimum(0)
        self.tSlider.setMaximum(10000)

        self.zoomInButton = QtWidgets.QPushButton("Zoom +")
        self.zoomOutButton = QtWidgets.QPushButton("Zoom -")
        self.resetViewButton = QtWidgets.QPushButton("Reset View")
        self.zoomModeButton = QtWidgets.QPushButton("Zoom Mode")
        self.zoomModeButton.setCheckable(True)
        self.zoomModeButton.setObjectName("modeToggleButton")
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
        self.measureButton.setObjectName("modeToggleButton")
        self.measureButton.toggled.connect(self.toggleMeasureMode)

        buttonLayout = QtWidgets.QHBoxLayout()
        buttonLayout.addWidget(self.jumpNButton)
        buttonLayout.addWidget(self.prevRfPulseButton)
        buttonLayout.addWidget(self.zoomOutButton)
        buttonLayout.addWidget(self.zoomInButton)
        buttonLayout.addWidget(self.resetViewButton)
        buttonLayout.addWidget(self.nextRfPulseButton)
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

        exportMeasurementsAction = QtGui.QAction("Export Measurements", self)
        exportMeasurementsAction.triggered.connect(self.export_measurements_to_excel)
        fileMenu.addAction(exportMeasurementsAction)

        self.snapMeasureAction = QtGui.QAction("Stick Measurements To Events", self)
        self.snapMeasureAction.setCheckable(True)
        self.snapMeasureAction.setChecked(self.measureSnapToEvents)
        self.snapMeasureAction.toggled.connect(self.toggleMeasureSnapToEvents)
        viewMenu.addAction(self.snapMeasureAction)

        self.zeroTrajectoryAtCursorAction = QtGui.QAction("Zero Trajectory At Cursor", self)
        self.zeroTrajectoryAtCursorAction.triggered.connect(self.zero_trajectory_at_cursor)
        viewMenu.addAction(self.zeroTrajectoryAtCursorAction)

        self.prevRfPulseAction = QtGui.QAction("Jump To Previous RF Pulse", self)
        self.prevRfPulseAction.triggered.connect(self.jump_to_previous_rf_pulse)
        viewMenu.addAction(self.prevRfPulseAction)

        self.nextRfPulseAction = QtGui.QAction("Jump To Next RF Pulse", self)
        self.nextRfPulseAction.triggered.connect(self.jump_to_next_rf_pulse)
        viewMenu.addAction(self.nextRfPulseAction)

        self.resetTrajectoryZeroAction = QtGui.QAction("Reset Trajectory Zero", self)
        self.resetTrajectoryZeroAction.triggered.connect(self.reset_trajectory_zero)
        viewMenu.addAction(self.resetTrajectoryZeroAction)

        self.jumpToPpgLineAction = QtGui.QAction("Jump To PPG Line...", self)
        self.jumpToPpgLineAction.triggered.connect(self.jump_to_ppg_line)
        viewMenu.addAction(self.jumpToPpgLineAction)

        shortcutsHelpAction = QtGui.QAction("Shortcuts", self)
        shortcutsHelpAction.setShortcut(QtGui.QKeySequence(Qt.Key.Key_F1))
        shortcutsHelpAction.triggered.connect(self.showShortcutsHelp)
        helpMenu.addAction(shortcutsHelpAction)

        self.settings = QSettings("MR_ISIBrno", "BrukerSimView")
        default_false = False
        self.measureSnapToEvents = bool(self.settings.value("measureSnapToEvents", default_false, type=bool))
        self.snapMeasureAction.setChecked(self.measureSnapToEvents)
        self.gradientCalibrationHzPerMm = float(self.settings.value("gradientCalibrationHzPerMm", 0.0, type=float))
        self.nucleusGammaMHzPerT = float(self.settings.value("nucleusGammaMHzPerT", PROTON_GAMMA_MHZ_PER_T, type=float))
        env_pw_reference = os.getenv("SIMVIEW_BRUKER_PW_REF_W", "1.0")
        try:
            default_pw_reference = float(env_pw_reference)
        except ValueError:
            default_pw_reference = 1.0
        if default_pw_reference <= 0:
            default_pw_reference = 1.0
        self.brukerPwReferenceWatts = float(
            self.settings.value("brukerPwReferenceWatts", default_pw_reference, type=float),
        )
        if self.brukerPwReferenceWatts <= 0:
            self.brukerPwReferenceWatts = default_pw_reference
        stored_gradient_display_units = self.settings.value("gradientDisplayUnits", None)
        if stored_gradient_display_units in {None, ""}:
            legacy_display_mt_per_m = bool(
                self.settings.value("displayGradientsInMtPerM", default_false, type=bool),
            )
            self.gradientDisplayUnits = "mt_per_m" if legacy_display_mt_per_m else "hz_per_mm"
        else:
            self.gradientDisplayUnits = str(stored_gradient_display_units).lower()
        self.themeMode = str(self.settings.value("themeMode", "system")).lower()
        self.derivedSignalStartupPadding = float(self.settings.value("derivedSignalStartupPadding", 1e-2, type=float))
        stored_trajectory_zero = self.settings.value("trajectoryZeroReferenceTime", None)
        self.trajectoryZeroReferenceTime = (
            float(stored_trajectory_zero) if stored_trajectory_zero not in {None, ""} else None
        )

        self.selectedChannels = self.settings.value("selectedChannels", [])
        if not isinstance(self.selectedChannels, list):
            self.selectedChannels = [self.selectedChannels] if self.selectedChannels else []
        self.sidebarCollapsed = bool(self.settings.value("sidebarCollapsed", False, type=bool))
        self.sidebarWidth = int(self.settings.value("sidebarWidth", 260, type=int))

        self.sidePanel = QtWidgets.QWidget()
        self.sidePanel.setMinimumWidth(180)
        self.sidePanel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.sidePanelLayout = QVBoxLayout(self.sidePanel)
        self.sidePanelLayout.setContentsMargins(10, 10, 10, 10)
        self.sidePanelLayout.setSpacing(10)
        self.sideTabs = QtWidgets.QTabWidget()
        self.sideTabs.setDocumentMode(True)

        self.sidebarToggleButton = QtWidgets.QPushButton()
        self.sidebarToggleButton.setObjectName("sidebarToggleButton")
        self.sidebarToggleButton.setFixedWidth(22)
        self.sidebarToggleButton.setFixedHeight(56)
        self.sidebarToggleButton.clicked.connect(self.toggle_side_panel)

        self.sidebarRail = QtWidgets.QWidget()
        self.sidebarRail.setObjectName("sidebarRail")
        self.sidebarRailLayout = QVBoxLayout(self.sidebarRail)
        self.sidebarRailLayout.setContentsMargins(0, 10, 0, 10)
        self.sidebarRailLayout.setSpacing(8)
        self.sidebarRailLayout.addWidget(self.sidebarToggleButton, alignment=Qt.AlignmentFlag.AlignTop)
        self.sidebarRailLayout.addStretch(1)

        self.sidePanelDock = QtWidgets.QWidget()
        self.sidePanelDock.setObjectName("sidePanelDock")
        self.sidePanelDockLayout = QHBoxLayout(self.sidePanelDock)
        self.sidePanelDockLayout.setContentsMargins(0, 0, 0, 0)
        self.sidePanelDockLayout.setSpacing(0)
        self.sidePanelDockLayout.addWidget(self.sidebarRail)
        self.sidePanelDockLayout.addWidget(self.sidePanel, stretch=1)

        self.channelsTab = QtWidgets.QWidget()
        self.channelsLayout = QVBoxLayout(self.channelsTab)
        self.channelsLayout.setContentsMargins(10, 10, 10, 10)
        self.channelsLayout.setSpacing(10)

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
        self.settingsLayout.setContentsMargins(10, 10, 10, 10)
        self.settingsLayout.setSpacing(10)

        self.measurementsTab = QtWidgets.QWidget()
        self.measurementsLayout = QVBoxLayout(self.measurementsTab)
        self.measurementsLayout.setContentsMargins(10, 10, 10, 10)
        self.measurementsLayout.setSpacing(10)

        self.measurementsListWidget = QtWidgets.QListWidget()
        self.measurementsListWidget.itemDoubleClicked.connect(self.jump_to_measurement_item)
        self.measurementsListWidget.currentItemChanged.connect(self.on_measurement_selection_changed)
        self.removeMeasurementButton = QtWidgets.QPushButton("Remove Selected")
        self.removeMeasurementButton.clicked.connect(self.remove_selected_measurement)
        self.removeMeasurementButton.setEnabled(False)
        self.clearMeasurementsButton = QtWidgets.QPushButton("Clear All")
        self.clearMeasurementsButton.clicked.connect(self.clear_measurements)
        self.clearMeasurementsButton.setEnabled(False)
        self.measurementLabelEdit = QtWidgets.QLineEdit()
        self.measurementLabelEdit.setPlaceholderText("Measurement label")
        self.measurementLabelEdit.setEnabled(False)
        self.measurementLabelEdit.returnPressed.connect(self.rename_selected_measurement)
        self.saveMeasurementLabelButton = QtWidgets.QPushButton("Save Label")
        self.saveMeasurementLabelButton.clicked.connect(self.rename_selected_measurement)
        self.saveMeasurementLabelButton.setEnabled(False)
        self.exportMeasurementsButton = QtWidgets.QPushButton("Export to Excel")
        self.exportMeasurementsButton.clicked.connect(self.export_measurements_to_excel)
        self.exportMeasurementsButton.setEnabled(False)
        self.measurementsHint = QtWidgets.QLabel("Completed measurements are saved here. Double-click one to jump to it.")
        self.measurementsHint.setWordWrap(True)
        self.measurementButtonsLayout = QtWidgets.QHBoxLayout()
        self.measurementButtonsLayout.addWidget(self.removeMeasurementButton)
        self.measurementButtonsLayout.addWidget(self.clearMeasurementsButton)
        self.measurementLabelLayout = QtWidgets.QHBoxLayout()
        self.measurementLabelLayout.addWidget(self.measurementLabelEdit)
        self.measurementLabelLayout.addWidget(self.saveMeasurementLabelButton)

        self.gradientCalibrationSpinBox = QtWidgets.QDoubleSpinBox()
        self.gradientCalibrationSpinBox.setDecimals(3)
        self.gradientCalibrationSpinBox.setRange(0.0, 1_000_000.0)
        self.gradientCalibrationSpinBox.setSingleStep(1.0)
        self.gradientCalibrationSpinBox.setSuffix(" Hz/mm @ 100%")
        self.gradientCalibrationSpinBox.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.gradientCalibrationSpinBox.setValue(self.gradientCalibrationHzPerMm)
        self.themeModeComboBox = QtWidgets.QComboBox()
        self.themeModeComboBox.addItem("System", "system")
        self.themeModeComboBox.addItem("Light", "light")
        self.themeModeComboBox.addItem("Dark", "dark")
        theme_index = max(self.themeModeComboBox.findData(self.themeMode), 0)
        self.themeModeComboBox.setCurrentIndex(theme_index)
        self.themeModeComboBox.currentIndexChanged.connect(self.on_theme_mode_changed)
        self.nucleusGammaSpinBox = QtWidgets.QDoubleSpinBox()
        self.nucleusGammaSpinBox.setDecimals(3)
        self.nucleusGammaSpinBox.setRange(0.001, 1_000.0)
        self.nucleusGammaSpinBox.setSingleStep(0.1)
        self.nucleusGammaSpinBox.setSuffix(" MHz/T")
        self.nucleusGammaSpinBox.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.nucleusGammaSpinBox.setValue(self.nucleusGammaMHzPerT)
        self.brukerPwReferenceSpinBox = QtWidgets.QDoubleSpinBox()
        self.brukerPwReferenceSpinBox.setDecimals(6)
        self.brukerPwReferenceSpinBox.setRange(1e-9, 1_000_000_000.0)
        self.brukerPwReferenceSpinBox.setSingleStep(0.1)
        self.brukerPwReferenceSpinBox.setSuffix(" W")
        self.brukerPwReferenceSpinBox.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.brukerPwReferenceSpinBox.setValue(self.brukerPwReferenceWatts)
        self.maxGradientStrengthValue = QtWidgets.QLineEdit()
        self.maxGradientStrengthValue.setReadOnly(True)
        self.derivedSignalStartupPaddingSpinBox = QtWidgets.QDoubleSpinBox()
        self.derivedSignalStartupPaddingSpinBox.setDecimals(3)
        self.derivedSignalStartupPaddingSpinBox.setRange(0.0, 10_000.0)
        self.derivedSignalStartupPaddingSpinBox.setSingleStep(0.1)
        self.derivedSignalStartupPaddingSpinBox.setSuffix(" s")
        self.derivedSignalStartupPaddingSpinBox.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.derivedSignalStartupPaddingSpinBox.setValue(self.derivedSignalStartupPadding)
        self.gradientDisplayUnitsComboBox = QtWidgets.QComboBox()
        self.gradientDisplayUnitsComboBox.addItem("Percent", "percent")
        self.gradientDisplayUnitsComboBox.addItem("Hz/mm", "hz_per_mm")
        self.gradientDisplayUnitsComboBox.addItem("mT/m", "mt_per_m")
        gradient_display_index = max(self.gradientDisplayUnitsComboBox.findData(self.gradientDisplayUnits), 0)
        self.gradientDisplayUnitsComboBox.setCurrentIndex(gradient_display_index)
        self.applyScannerSettingsButton = QtWidgets.QPushButton("Apply Settings")
        self.applyScannerSettingsButton.setObjectName("primaryButton")
        self.applyScannerSettingsButton.clicked.connect(self.apply_scanner_settings)

        appearanceGroup = QtWidgets.QGroupBox("Appearance")
        appearanceLayout = QtWidgets.QFormLayout(appearanceGroup)
        appearanceLayout.addRow("Theme", self.themeModeComboBox)

        scannerGroup = QtWidgets.QGroupBox("Scanner Calibration")
        scannerGroupLayout = QtWidgets.QVBoxLayout(scannerGroup)
        scannerLayout = QtWidgets.QFormLayout()
        self.scannerSettingsHint = QtWidgets.QLabel(
            "Gradient channels are loaded in percent. Bruker pw channels are attenuation in dB and use the "
            "reference power for W conversion."
        )
        self.scannerSettingsHint.setWordWrap(True)
        self.scannerSettingsHint.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.MinimumExpanding,
        )
        scannerGroupLayout.addWidget(self.scannerSettingsHint)
        scannerLayout.addRow("Grad Calibration", self.gradientCalibrationSpinBox)
        scannerLayout.addRow("Nucleus Gamma", self.nucleusGammaSpinBox)
        scannerLayout.addRow("Bruker PW Ref", self.brukerPwReferenceSpinBox)
        scannerLayout.addRow("Max Grad @ 100%", self.maxGradientStrengthValue)
        scannerLayout.addRow("Gradient Display", self.gradientDisplayUnitsComboBox)
        scannerGroupLayout.addLayout(scannerLayout)

        derivedSignalsGroup = QtWidgets.QGroupBox("Derived Signals")
        derivedSignalsLayout = QtWidgets.QFormLayout(derivedSignalsGroup)
        derivedSignalsLayout.addRow("Startup Padding", self.derivedSignalStartupPaddingSpinBox)

        self.settingsLayout.addWidget(appearanceGroup)
        self.settingsLayout.addWidget(scannerGroup)
        self.settingsLayout.addWidget(derivedSignalsGroup)
        self.settingsLayout.addWidget(self.applyScannerSettingsButton)
        self.settingsLayout.addStretch(1)
        self.gradientCalibrationSpinBox.valueChanged.connect(self.update_scanner_settings_display)
        self.nucleusGammaSpinBox.valueChanged.connect(self.update_scanner_settings_display)
        self.update_scanner_settings_display()

        self.measurementsLayout.addWidget(self.measurementsHint)
        self.measurementsLayout.addWidget(self.measurementsListWidget)
        self.measurementsLayout.addLayout(self.measurementLabelLayout)
        self.measurementsLayout.addLayout(self.measurementButtonsLayout)
        self.measurementsLayout.addWidget(self.exportMeasurementsButton)

        self.sideTabs.addTab(self.channelsTab, "Channels")
        self.sideTabs.addTab(self.settingsTab, "Settings")
        self.sideTabs.addTab(self.measurementsTab, "Measurements")
        self.sidePanelLayout.addWidget(self.sideTabs)
        self.refresh_measurements_list()
        self.sidePanelDock.hide()

        plot_layout_item = self.horizontalLayout_2.takeAt(0)
        assert plot_layout_item is not None
        plot_layout = plot_layout_item.layout()
        assert plot_layout is not None
        self.plotAreaWidget = QtWidgets.QWidget()
        self.plotAreaWidget.setObjectName("plotAreaWidget")
        self.plotAreaWidget.setLayout(plot_layout)

        self.mainSplitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        self.mainSplitter.setObjectName("mainSplitter")
        self.mainSplitter.setChildrenCollapsible(False)
        self.mainSplitter.addWidget(self.sidePanelDock)
        self.mainSplitter.addWidget(self.plotAreaWidget)
        self.mainSplitter.setStretchFactor(0, 0)
        self.mainSplitter.setStretchFactor(1, 1)
        self.mainSplitter.splitterMoved.connect(self.on_main_splitter_moved)
        self.mainSplitter.setSizes([max(self.sidebarWidth, 180) + self.sidebarRail.sizeHint().width(), 900])
        self.horizontalLayout_2.addWidget(self.mainSplitter)
        self.update_sidebar_toggle_button()

        qt_app = QApplication.instance()
        assert qt_app is not None
        self.systemPalette = qt_app.palette()
        self.standardPalette = qt_app.style().standardPalette()
        self.apply_theme_settings()

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
