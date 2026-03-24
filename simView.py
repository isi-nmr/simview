import contextlib
import io
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtGui, QtWidgets, uic
from PyQt6.QtCore import QCoreApplication, QDir, QMarginsF, QRectF, QSettings, QSizeF, Qt
from PyQt6.QtGui import QPageLayout, QPageSize, QPainter, QPalette, QPdfWriter
from PyQt6.QtSvg import QSvgRenderer
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

        exportPlotsAction = QtGui.QAction("Export Visible Plots", self)
        exportPlotsAction.triggered.connect(self.export_visible_plots)
        fileMenu.addAction(exportPlotsAction)

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

    def sanitize_filename(self, value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
        return sanitized or "plot"

    def slice_curve_to_range(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_min: float,
        x_max: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if x_data.size == 0 or y_data.size == 0:
            return x_data, y_data

        visible_mask = (x_data >= x_min) & (x_data <= x_max)
        visible_indices = np.flatnonzero(visible_mask)

        if visible_indices.size == 0:
            right_index = int(np.searchsorted(x_data, x_min, side="left"))
            candidate_indices = {min(max(right_index - 1, 0), x_data.size - 1), min(right_index, x_data.size - 1)}
            selected_indices = np.array(sorted(candidate_indices), dtype=int)
            return x_data[selected_indices], y_data[selected_indices]

        start_index = max(int(visible_indices[0]) - 1, 0)
        end_index = min(int(visible_indices[-1]) + 1, x_data.size - 1)
        selected_indices = np.arange(start_index, end_index + 1)
        return x_data[selected_indices], y_data[selected_indices]

    def simplify_curve_indices(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_scale: float,
        y_scale: float,
        tolerance_px: float,
    ) -> np.ndarray:
        point_count = x_data.size
        if point_count <= 2:
            return np.arange(point_count, dtype=int)

        x_screen = x_data * x_scale
        y_screen = y_data * y_scale
        keep_mask = np.zeros(point_count, dtype=bool)
        keep_mask[0] = True
        keep_mask[-1] = True
        stack: list[tuple[int, int]] = [(0, point_count - 1)]
        tolerance_sq = tolerance_px * tolerance_px

        while stack:
            start_index, end_index = stack.pop()
            if end_index <= start_index + 1:
                continue

            start_point = np.array((x_screen[start_index], y_screen[start_index]))
            end_point = np.array((x_screen[end_index], y_screen[end_index]))
            segment = end_point - start_point
            segment_length_sq = float(np.dot(segment, segment))

            interior_slice = slice(start_index + 1, end_index)
            points = np.column_stack((x_screen[interior_slice], y_screen[interior_slice]))
            if points.size == 0:
                continue

            if segment_length_sq <= 1e-12:
                distances_sq = np.sum((points - start_point) ** 2, axis=1)
            else:
                projection = np.clip(np.dot(points - start_point, segment) / segment_length_sq, 0.0, 1.0)
                closest_points = start_point + np.outer(projection, segment)
                distances_sq = np.sum((points - closest_points) ** 2, axis=1)

            max_offset = int(np.argmax(distances_sq))
            max_distance_sq = float(distances_sq[max_offset])
            if max_distance_sq > tolerance_sq:
                split_index = start_index + 1 + max_offset
                keep_mask[split_index] = True
                stack.append((start_index, split_index))
                stack.append((split_index, end_index))

        return np.flatnonzero(keep_mask)

    def downsample_curve_to_viewport(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        viewport_width: int,
        viewport_height: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        *,
        max_point_factor: float = 6.0,
        min_points: int = 3000,
    ) -> tuple[np.ndarray, np.ndarray]:
        if x_data.size <= 2:
            return x_data, y_data

        if not np.all(np.isfinite(x_data)) or not np.all(np.isfinite(y_data)):
            return x_data, y_data

        x_span = max(abs(x_max - x_min), 1e-12)
        y_span = max(abs(y_max - y_min), 1e-12)
        x_scale = max(viewport_width - 1, 1) / x_span
        y_scale = max(viewport_height - 1, 1) / y_span
        max_points = max(int(viewport_width * max_point_factor), min_points)

        if x_data.size <= max_points:
            return x_data, y_data

        low_tolerance = 0.0
        high_tolerance = 0.5
        kept_indices = self.simplify_curve_indices(x_data, y_data, x_scale, y_scale, high_tolerance)

        while kept_indices.size > max_points and high_tolerance < 64.0:
            low_tolerance = high_tolerance
            high_tolerance *= 2.0
            kept_indices = self.simplify_curve_indices(x_data, y_data, x_scale, y_scale, high_tolerance)

        for _ in range(16):
            if kept_indices.size <= max_points:
                break
            mid_tolerance = (low_tolerance + high_tolerance) * 0.5
            kept_indices = self.simplify_curve_indices(x_data, y_data, x_scale, y_scale, mid_tolerance)
            if kept_indices.size > max_points:
                low_tolerance = mid_tolerance
            else:
                high_tolerance = mid_tolerance

        if kept_indices.size > max_points:
            sample_indices = np.linspace(0, kept_indices.size - 1, num=max_points, dtype=int)
            kept_indices = kept_indices[np.unique(sample_indices)]

        return x_data[kept_indices], y_data[kept_indices]

    def build_export_plot(self, plot: CursorPlot) -> pg.PlotWidget:
        source_item = plot.getPlotItem()
        export_plot = pg.PlotWidget()
        export_width = max(plot.width(), 1)
        export_height = max(plot.height(), 1)
        export_plot.resize(export_width, export_height)

        if self.darkMode:
            export_plot.setBackground("black")
        else:
            export_plot.setBackground("white")

        export_item = export_plot.getPlotItem()
        for axis_name in ("left", "right", "bottom", "top"):
            source_axis = source_item.getAxis(axis_name)
            export_item.showAxis(axis_name, show=source_axis.isVisible())

        export_item.getAxis("left").setWidth(60)
        export_item.getAxis("right").setWidth(60)

        for axis_name in ("left", "right", "bottom", "top"):
            source_axis = source_item.getAxis(axis_name)
            if source_axis.labelText:
                export_plot.setLabel(axis_name, source_axis.labelText, units=source_axis.labelUnits)

        if source_item.legend is not None:
            export_plot.addLegend(offset=(10, 10))

        view_range = plot.getViewBox().viewRange()
        x_min, x_max = view_range[0]
        y_min, y_max = view_range[1]

        curve_sources: list[tuple[np.ndarray, np.ndarray, pg.PlotDataItem]] = []
        if getattr(plot, "managed_curves", None):
            for curve in plot.managed_curves:
                curve_sources.append(
                    (
                        np.asarray(curve["x_data"]),
                        np.asarray(curve["y_data"]),
                        curve["item"],
                    ),
                )
        else:
            for data_item in source_item.listDataItems():
                x_data, y_data = data_item.getData()
                if x_data is None or y_data is None:
                    continue
                curve_sources.append((np.asarray(x_data), np.asarray(y_data), data_item))

        for x_data, y_data, data_item in curve_sources:
            if x_data is None or y_data is None:
                continue

            sliced_x, sliced_y = self.slice_curve_to_range(x_data, y_data, x_min, x_max)
            if sliced_x.size == 0 or sliced_y.size == 0:
                continue

            downsampled_x, downsampled_y = self.downsample_curve_to_viewport(
                sliced_x,
                sliced_y,
                export_width,
                export_height,
                x_min,
                x_max,
                y_min,
                y_max,
            )

            plot_kwargs = {}
            for option_name in (
                "pen",
                "fillLevel",
                "fillBrush",
                "stepMode",
                "connect",
                "name",
            ):
                option_value = data_item.opts.get(option_name)
                if option_value is not None:
                    plot_kwargs[option_name] = option_value

            plot_kwargs["symbol"] = None

            export_curve = export_plot.plot(downsampled_x, downsampled_y, **plot_kwargs)
            export_curve.setClipToView(True)
            export_curve.setSkipFiniteCheck(True)

        export_plot.setXRange(x_min, x_max, padding=0)
        export_plot.setYRange(y_min, y_max, padding=0)
        export_plot.getViewBox().setMouseEnabled(x=False, y=False)
        return export_plot

    def export_plot_to_svg(self, plot: CursorPlot, output_path: Path) -> None:
        try:
            from pyqtgraph.exporters import SVGExporter
        except ImportError as exc:
            raise RuntimeError("PyQtGraph SVG exporter is unavailable.") from exc

        export_plot = self.build_export_plot(plot)
        try:
            exporter = SVGExporter(export_plot.plotItem)
            exporter.parameters()["width"] = max(plot.width(), 1)
            exporter.parameters()["height"] = max(plot.height(), 1)
            # PyQtGraph 0.13.x can print harmless QGraphicsObject.paint() warnings
            # from internal helper items even when export succeeds.
            with contextlib.redirect_stderr(io.StringIO()):
                exporter.export(str(output_path))
            self.finalize_svg_export(output_path, max(plot.width(), 1), max(plot.height(), 1))
        finally:
            export_plot.deleteLater()

    def finalize_svg_export(self, output_path: Path, export_width: int, export_height: int) -> None:
        tree = ET.parse(output_path)
        root = tree.getroot()
        view_box = root.get("viewBox")
        if view_box:
            view_box_values = view_box.replace(",", " ").split()
            if len(view_box_values) == 4:
                export_width = int(round(float(view_box_values[2])))
                export_height = int(round(float(view_box_values[3])))

        root.set("width", str(export_width))
        root.set("height", str(export_height))
        root.set("preserveAspectRatio", "xMidYMid meet")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    def export_plot_to_pdf(self, plot: CursorPlot, output_path: Path) -> None:
        temp_svg_path: Path | None = None
        temp_fd, temp_name = tempfile.mkstemp(suffix=".svg", prefix="simview_export_", dir=str(output_path.parent))
        os.close(temp_fd)
        temp_svg_path = Path(temp_name)

        try:
            self.export_plot_to_svg(plot, temp_svg_path)

            svg_renderer = QSvgRenderer(str(temp_svg_path))
            if not svg_renderer.isValid():
                raise RuntimeError(f"Temporary SVG export is invalid: {temp_svg_path}")

            view_box = svg_renderer.viewBoxF()
            if view_box.isEmpty():
                default_size = svg_renderer.defaultSize()
                view_box = QRectF(0, 0, max(default_size.width(), 1), max(default_size.height(), 1))

            page_margin = 12.0
            pdf_writer = QPdfWriter(str(output_path))
            pdf_writer.setResolution(300)
            page_size = QPageSize(
                QSizeF(view_box.width() + 2 * page_margin, view_box.height() + 2 * page_margin),
                QPageSize.Unit.Point,
            )
            pdf_writer.setPageSize(page_size)
            pdf_writer.setPageMargins(QMarginsF(page_margin, page_margin, page_margin, page_margin), QPageLayout.Unit.Point)

            painter = QPainter(pdf_writer)
            try:
                target_rect = QRectF(pdf_writer.pageLayout().paintRect())
                svg_renderer.render(painter, target_rect)
            finally:
                painter.end()
        finally:
            if temp_svg_path is not None:
                temp_svg_path.unlink(missing_ok=True)

    def export_visible_plots(self) -> None:
        visible_plots: list[tuple[int, CursorPlot]] = []
        for index, (container, plot) in enumerate(zip(self.plotContainers, self.plots, strict=False)):
            if container.isVisible():
                visible_plots.append((index, plot))

        if not visible_plots:
            dialog.showErrorMessage("There are no visible plots to export.")
            return

        default_dir = self.settings.value("lastExportFolder", self.dataPath or QDir.homePath())
        default_name = Path(default_dir) / "simview_export.svg"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Visible Plots",
            str(default_name),
            "Vector files (*.svg *.pdf);;SVG files (*.svg);;PDF files (*.pdf)",
        )
        if not file_path:
            return

        target_path = Path(file_path)
        export_format = target_path.suffix.lower()
        if export_format not in {".svg", ".pdf"}:
            target_path = target_path.with_suffix(".svg")
            export_format = ".svg"

        if export_format == ".svg":
            try:
                from pyqtgraph.exporters import SVGExporter
            except ImportError:
                dialog.showErrorMessage("SVG export is unavailable because PyQtGraph SVG exporters could not be loaded.")
                return

        export_dir = target_path.parent
        export_dir.mkdir(parents=True, exist_ok=True)
        self.settings.setValue("lastExportFolder", str(export_dir))

        exported_files: list[Path] = []
        multiple_plots = len(visible_plots) > 1

        for export_index, (plot_index, plot) in enumerate(visible_plots, start=1):
            label = self.channels[plot_index][0]["chanLabel"]
            suffix = f"_{export_index:02d}_{self.sanitize_filename(label)}" if multiple_plots else ""
            output_path = target_path.with_name(f"{target_path.stem}{suffix}{export_format}")

            try:
                if export_format == ".svg":
                    self.export_plot_to_svg(plot, output_path)
                else:
                    self.export_plot_to_pdf(plot, output_path)
            except Exception as exc:
                dialog.showErrorMessage(f"Failed to export plot '{label}': {exc}")
                return
            exported_files.append(output_path)

        if len(exported_files) == 1:
            self.statusBar().showMessage(f"Exported visible plot to {exported_files[0]}", 5000)
        else:
            self.statusBar().showMessage(f"Exported {len(exported_files)} visible plots to {export_dir}", 5000)

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

                currentPlot.add_managed_curve(stepData["t"], stepData["data"], name=line["label"], pen=currentPen)

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
