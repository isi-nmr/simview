import contextlib
import io
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets
from PyQt6.QtCore import QDir, QMarginsF, QRectF, QSizeF
from PyQt6.QtGui import QPageLayout, QPageSize, QPainter, QPdfWriter
from PyQt6.QtSvg import QSvgRenderer

from utils import dialog
from widgets.mulitPlotCursor import CursorPlot


class ExportMixin:
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
            export_item.getAxis(axis_name).enableAutoSIPrefix(False)

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
            for option_name in ("pen", "fillLevel", "fillBrush", "stepMode", "connect", "name"):
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
