import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QResizeEvent, QShowEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication


class TextItemWithBg(pg.TextItem):
    def __init__(self, text:str="", color:str="w", bg_color:QColor|None=None, **kwargs:dict)->None:
        super().__init__(text, color=color, **kwargs)
        if bg_color is None:
            bg_color = QColor(0, 0, 0, 150)
        self.bg_color = bg_color

    def paint(self, p: QPainter, *args:tuple)->None:
        # Draw background rectangle
        rect = self.boundingRect()
        p.fillRect(rect, self.bg_color)
        # Draw the usual text on top
        super().paint(p, *args)


class CursorPlot(pg.PlotWidget):
    """A PlotWidget with its own vertical cursor line."""

    def __init__(self, *args:tuple, darkMode:bool=False, **kwargs:dict)->None:
        self.dataLoaded = False
        self.curve_cache: list[tuple[str, np.ndarray, np.ndarray]] = []
        self.managed_curves: list[dict[str, object]] = []
        self.last_cursor_x: float | None = None
        self._last_refresh_key: tuple[float, float, float, float, int, int] | None = None
        self._resize_in_progress = False
        self._refresh_in_progress = False
        self.annotation_items: list[dict[str, object]] = []
        super().__init__(*args, **kwargs)

        # Create the vertical line cursor
        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=1))
        self.cursor_line.hide()
        self.addItem(self.cursor_line)

        bgColor = QColor(0, 0, 0, 150) if darkMode else QColor(0, 0, 0, 10)

        self.timestamp_label = TextItemWithBg("", anchor=(1, 0), color="r", bg_color=bgColor)
        self.timestamp_label.setZValue(100)
        self.addItem(self.timestamp_label, ignoreBounds=True)

        self.point_label = TextItemWithBg("", anchor=(1, 0), color="r", bg_color=bgColor)
        self.point_label.setZValue(100)

        self.addItem(self.point_label, ignoreBounds=True)

        # Connect scene signals
        self.mouse_proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=90, slot=self.on_mouse_moved)
        self.scene().installEventFilter(self)  # for enter/leave detection

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._handle_refresh_timeout)
        self.getViewBox().sigXRangeChanged.connect(self.schedule_curve_refresh)
        self.getViewBox().sigYRangeChanged.connect(self.schedule_curve_refresh)
        self.getViewBox().sigXRangeChanged.connect(self.refresh_annotation_positions)
        self.getViewBox().sigYRangeChanged.connect(self.refresh_annotation_positions)

        self.temp_region = None
        self.temp_text = None
        # Time measurement state
        self.measure_mode = False
        self.start_x = None
        self.measure_line = None
        self.measure_text = None

        self.zoom_mode = False
        self.zoom_start_x = None
        self.zoom_region_temp = None

        self.timestamp_label.hide()
        self.point_label.hide()

    def get_main_window(self) -> object | None:
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "plots") and hasattr(parent, "update_status"):
                return parent
            parent = parent.parent()
        return None

    def set_interaction_mode(self, mode: str) -> None:
        self.measure_mode = mode == "measure"
        self.zoom_mode = mode == "zoom"
        self.start_x = None
        self.zoom_start_x = None

        if self.temp_region:
            self.removeItem(self.temp_region)
            self.temp_region = None
        if self.temp_text:
            self.removeItem(self.temp_text)
            self.temp_text = None
        if self.zoom_region_temp:
            self.removeItem(self.zoom_region_temp)
            self.zoom_region_temp = None

    def register_curve(self, name: str, x_data: np.ndarray, y_data: np.ndarray) -> None:
        self.curve_cache.append((name, x_data, y_data))

    def apply_theme(self, *, dark_mode: bool) -> None:
        self.setBackground("black" if dark_mode else "white")
        overlay_bg = QColor(0, 0, 0, 150) if dark_mode else QColor(255, 255, 255, 220)
        cursor_bg = QColor(0, 0, 0, 150) if dark_mode else QColor(0, 0, 0, 10)
        axis_color = "white" if dark_mode else "black"
        axis_pen = pg.mkPen(axis_color)

        self.timestamp_label.bg_color = cursor_bg
        self.point_label.bg_color = cursor_bg

        plot_item = self.getPlotItem()
        for axis_name in ("left", "right", "bottom", "top"):
            axis = plot_item.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)
            axis.setTickPen(axis_pen)
            if axis.labelText:
                axis.setLabel(axis.labelText, units=axis.labelUnits, color=axis_color)
        if plot_item.legend is not None:
            plot_item.legend.setLabelTextColor(axis_color)
            for _sample, label in plot_item.legend.items:
                label.setAttr("color", axis_color)
                label.setText(label.text, color=axis_color)

        for annotation in self.annotation_items:
            text_item = annotation["text"]
            line_item = annotation["line"]
            text_item.bg_color = overlay_bg
            line_item.setPen(pg.mkPen("r", style=Qt.PenStyle.DashLine))

    def add_annotation_marker(self, time_value: float, text_value: str, *, color: str = "r") -> None:
        annotation_line = pg.InfiniteLine(
            pos=time_value,
            angle=90,
            pen=pg.mkPen(color, style=Qt.PenStyle.DashLine),
        )
        self.addItem(annotation_line)

        if self.backgroundBrush().color().lightness() < 128:
            bg_color = QColor(0, 0, 0, 150)
        else:
            bg_color = QColor(255, 255, 255, 220)
        annotation_text = TextItemWithBg(
            text_value,
            anchor=(0, 0),
            color=color,
            bg_color=bg_color,
        )
        annotation_text.setZValue(95)
        self.addItem(annotation_text, ignoreBounds=True)

        self.annotation_items.append(
            {
                "time": float(time_value),
                "line": annotation_line,
                "text": annotation_text,
            },
        )
        self.refresh_annotation_positions()

    def refresh_annotation_positions(self, *args: object) -> None:
        if not self.annotation_items:
            return

        x_range, y_range = self.viewRange()
        x_min, x_max = x_range
        y_min, y_max = y_range
        y_span = max(y_max - y_min, 1e-12)
        base_y = y_max - 0.06 * y_span
        row_spacing = 0.08 * y_span
        visible_row = 0

        for annotation in self.annotation_items:
            time_value = float(annotation["time"])
            text_item = annotation["text"]
            if x_min <= time_value <= x_max:
                text_item.show()
                text_y = base_y - visible_row * row_spacing
                if text_y <= y_min:
                    text_y = y_min + 0.02 * y_span
                text_item.setPos(time_value, text_y)
                visible_row += 1
            else:
                text_item.hide()

    def add_managed_curve(self, x_data: np.ndarray, y_data: np.ndarray, **plot_kwargs: object) -> pg.PlotDataItem:
        x_array = np.asarray(x_data)
        y_array = np.asarray(y_data)
        if x_array.size > 1 and y_array.size > 1:
            initial_x = x_array[:2]
            initial_y = y_array[:2]
        else:
            initial_x = x_array
            initial_y = y_array

        plot_curve = self.plot(initial_x, initial_y, **plot_kwargs)
        plot_curve.setClipToView(True)
        plot_curve.setSkipFiniteCheck(True)

        curve_name = str(plot_kwargs.get("name", ""))
        self.register_curve(curve_name, x_array, y_array)
        self.managed_curves.append(
            {
                "item": plot_curve,
                "x_data": x_array,
                "y_data": y_array,
                "cached_render_key": None,
                "cached_render_data": None,
            },
        )
        self.schedule_curve_refresh()
        return plot_curve

    def update_managed_curve(self, index: int, x_data: np.ndarray, y_data: np.ndarray) -> None:
        if index < 0 or index >= len(self.managed_curves):
            return

        self.managed_curves[index]["x_data"] = x_data
        self.managed_curves[index]["y_data"] = y_data
        self.managed_curves[index]["cached_render_key"] = None
        self.managed_curves[index]["cached_render_data"] = None

        if index < len(self.curve_cache):
            curve_name, _, _ = self.curve_cache[index]
            self.curve_cache[index] = (curve_name, x_data, y_data)

        self._last_refresh_key = None
        self.schedule_curve_refresh()

    def schedule_curve_refresh(self, *args: object) -> None:
        if not self.managed_curves:
            return
        if not self.isVisible():
            self._last_refresh_key = None
            return
        if self._refresh_in_progress:
            return
        if self._resize_in_progress:
            self._refresh_timer.start(120)
            return
        self._refresh_timer.start(0)

    def schedule_curve_refresh_delayed(self, delay_ms: int) -> None:
        if not self.managed_curves:
            return
        if not self.isVisible():
            self._last_refresh_key = None
            return
        self._refresh_timer.start(max(delay_ms, 0))

    def _handle_refresh_timeout(self) -> None:
        self._resize_in_progress = False
        self.refresh_visible_curves()

    def refresh_visible_curves(self) -> None:
        if not self.managed_curves:
            return
        if not self.isVisible():
            self._last_refresh_key = None
            return

        main_window = self.get_main_window()
        if main_window is None:
            return

        view_box = self.getViewBox()
        view_range = view_box.viewRange()
        x_min, x_max = view_range[0]
        y_min, y_max = view_range[1]
        view_width = max(round(view_box.sceneBoundingRect().width()), 1)
        view_height = max(round(view_box.sceneBoundingRect().height()), 1)
        refresh_key = (x_min, x_max, y_min, y_max, view_width, view_height)
        if self._last_refresh_key == refresh_key:
            return

        self._refresh_in_progress = True
        try:
            for curve in self.managed_curves:
                full_x = np.asarray(curve["x_data"])
                full_y = np.asarray(curve["y_data"])
                curve_render_key = (
                    refresh_key,
                    full_x.size,
                    full_y.size,
                    float(full_x[0]) if full_x.size else 0.0,
                    float(full_x[-1]) if full_x.size else 0.0,
                )
                cached_render_key = curve.get("cached_render_key")
                cached_render_data = curve.get("cached_render_data")
                if cached_render_key == curve_render_key and cached_render_data is not None:
                    simplified_x, simplified_y = cached_render_data
                else:
                    sliced_x, sliced_y = main_window.slice_curve_to_range(full_x, full_y, x_min, x_max)
                    simplified_x, simplified_y = main_window.downsample_curve_to_viewport(
                        sliced_x,
                        sliced_y,
                        view_width,
                        view_height,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        max_point_factor=4.0,
                        min_points=1200,
                    )
                    curve["cached_render_key"] = curve_render_key
                    curve["cached_render_data"] = (simplified_x, simplified_y)
                curve["item"].setData(simplified_x, simplified_y)

            self._last_refresh_key = refresh_key
        finally:
            self._refresh_in_progress = False

    def notify_measurement(self, dt_seconds: float | None = None) -> None:
        main_window = self.get_main_window()
        if main_window is not None:
            main_window.update_status(measurement=dt_seconds)

    def clear_measurement_overlay(self) -> None:
        self.start_x = None
        if self.temp_region:
            self.removeItem(self.temp_region)
            self.temp_region = None
        if self.temp_text:
            self.removeItem(self.temp_text)
            self.temp_text = None

    def ensure_measurement_overlay(self, start_x: float) -> None:
        self.start_x = float(start_x)
        if self.temp_region is None:
            self.temp_region = pg.LinearRegionItem(
                values=(self.start_x, self.start_x),
                movable=False,
                brush=(50, 50, 200, 50),
            )
            self.addItem(self.temp_region, ignoreBounds=True)
        else:
            self.temp_region.setRegion((self.start_x, self.start_x))

        if self.temp_text is None:
            self.temp_text = pg.TextItem("", color="b", anchor=(0.5, 1))
            self.addItem(self.temp_text, ignoreBounds=True)
        self.temp_text.setText("")

    def update_measurement_overlay(self, end_x: float) -> None:
        if self.start_x is None:
            return
        if self.temp_region is None or self.temp_text is None:
            return

        self.temp_region.setRegion((self.start_x, end_x))
        mid_x = (self.start_x + end_x) / 2
        bottom_y = self.viewRange()[1][0]
        delta_t = abs(end_x - self.start_x)
        self.temp_text.setText(f"Δt = {self.format_time(delta_t)}")
        self.temp_text.setPos(mid_x, bottom_y)

    def get_snapped_event_x(self, x_value: float) -> float:
        main_window = self.get_main_window()
        if main_window is None or not getattr(main_window, "measureSnapToEvents", False):
            return x_value

        nearest_x = x_value
        nearest_distance = float("inf")
        for _, cx, _ in self.curve_cache:
            if cx.size == 0:
                continue

            insert_index = int(np.searchsorted(cx, x_value, side="left"))
            candidate_indices = {min(max(insert_index - 1, 0), cx.size - 1), min(insert_index, cx.size - 1)}
            for candidate_index in candidate_indices:
                candidate_x = float(cx[candidate_index])
                distance = abs(candidate_x - x_value)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_x = candidate_x

        return nearest_x

    def mousePressEvent(self, event:QMouseEvent)->None:
        mouse_point = self.getViewBox().mapSceneToView(event.position())
        modifiers = QApplication.keyboardModifiers()
        shift_pressed = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        main_window = self.get_main_window()

        if self.measure_mode:
            shared_start_x = getattr(main_window, "measurement_start_x", None) if main_window is not None else self.start_x
            if shared_start_x is None:
                start_x = self.get_snapped_event_x(mouse_point.x())
                if main_window is not None:
                    main_window.measurement_start_x = start_x
                    main_window.measurement_source_plot = self
                    for other in main_window.plots:
                        other.ensure_measurement_overlay(start_x)
                else:
                    self.ensure_measurement_overlay(start_x)
                self.notify_measurement(None)
            else:
                end_x = self.get_snapped_event_x(mouse_point.x())
                delta_t = abs(end_x - shared_start_x)
                if main_window is not None:
                    for other in main_window.plots:
                        other.start_x = shared_start_x
                        other.update_measurement_overlay(end_x)
                else:
                    self.start_x = shared_start_x
                    self.update_measurement_overlay(end_x)
                self.notify_measurement(delta_t)
                self.measure_mode = False
                self.start_x = None
                if main_window is not None:
                    main_window.measurement_start_x = None
                    main_window.measurement_source_plot = None
                    for other in main_window.plots:
                        if other == self:
                            continue
                        other.measure_mode = False

        elif self.zoom_mode or shift_pressed:
            self.zoom_start_x = mouse_point.x()
            # create temporary region
            self.zoom_region_temp = pg.LinearRegionItem(
                values=(self.zoom_start_x, self.zoom_start_x),
                movable=False,
                brush=(100, 100, 100, 50),
            )
            self.addItem(self.zoom_region_temp, ignoreBounds=True)

        else:
            super().mousePressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        main_window = self.get_main_window()
        if main_window is None or not self.dataLoaded:
            super().wheelEvent(event)
            return

        angle_delta_y = event.angleDelta().y()
        angle_delta_x = event.angleDelta().x()
        if angle_delta_y == 0 and angle_delta_x == 0:
            super().wheelEvent(event)
            return

        mouse_point = self.getViewBox().mapSceneToView(event.position())
        modifiers = QApplication.keyboardModifiers()
        shift_pressed = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if shift_pressed:
            wheel_steps = angle_delta_y if angle_delta_y != 0 else angle_delta_x
            pan_fraction = -float(wheel_steps) / 1200.0
            main_window.pan_horizontally(main_window.windowWidth * pan_fraction)
        else:
            wheel_steps = angle_delta_y if angle_delta_y != 0 else angle_delta_x
            zoom_factor = 0.8 if wheel_steps > 0 else 1.25
            main_window.zoom_to_cursor(float(mouse_point.x()), zoom_factor)

        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        main_window = self.get_main_window()
        if main_window is not None and event.button() == Qt.MouseButton.LeftButton:
            main_window.resetView()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def enable_measure_mode(self,*, enabled:None|bool=None)->None:
        if enabled is None:
            enabled = True

        """Activate measurement mode and clear previous measurement."""
        self.measure_mode = enabled
        self.clear_measurement_overlay()

    def enable_zoom_mode(self,*, enabled:None|bool=None)->None:
        if enabled is None:
            enabled = True

        self.zoom_mode = enabled
        self.zoom_start_x = None
        if self.zoom_region_temp:
            self.removeItem(self.zoom_region_temp)
            self.zoom_region_temp = None

    def on_mouse_moved(self, event: tuple[QPointF]) -> None:
        pos = event[0]
        if self.zoom_start_x is not None and self.zoom_region_temp is not None:
            mouse_point = self.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()
            self.zoom_region_temp.setRegion((self.zoom_start_x, x))
            return

        """Move the cursor line when mouse moves inside the plot."""
        if not self.sceneBoundingRect().contains(pos):
            return

        mouse_point = self.getViewBox().mapSceneToView(pos)
        x = mouse_point.x()
        if self.last_cursor_x is not None and abs(x - self.last_cursor_x) < 1e-12:
            return
        self.last_cursor_x = x
        self.cursor_line.setPos(mouse_point.x())

        mouse_point = self.plotItem.vb.mapSceneToView(pos)
        x_val = mouse_point.x()
        y_val = mouse_point.y()

        # Loop over all curves to find the nearest point

        label_parts = []
        for name, cx, cy in self.curve_cache:
            nearest_idx = max(np.searchsorted(cx, x_val, side="right") - 1, 0)
            y_value = cy[min(nearest_idx + 1, cy.size - 1)]
            label_parts.append(f"{name}:{y_value:.2f}")

        self.point_label.setText(" ".join(label_parts))
        self.point_label.setPos(x_val, y_val)

        main_window = self.get_main_window()
        if main_window is not None:
            for other in main_window.plots:
                other.cursor_line.setPos(mouse_point.x())
                # Update timestamp label at bottom-right corner
                view_rect = other.viewRect()
                other.timestamp_label.setPos(view_rect.right(), view_rect.bottom())
                other.timestamp_label.setText(f"t = {x * 1e3:.2f} ms")
            main_window.update_status(cursor_time=x)

        # Update timestamp label at bottom-right corner
        view_rect = self.viewRect()
        self.timestamp_label.setPos(view_rect.right(), view_rect.bottom())
        self.timestamp_label.setText(f"t = {x * 1e3:.2f} ms")

        # Update measurement region dynamically
        shared_start_x = getattr(main_window, "measurement_start_x", None) if main_window is not None else self.start_x
        if self.measure_mode and shared_start_x is not None:
            end_x = self.get_snapped_event_x(mouse_point.x())
            delta_t = abs(end_x - shared_start_x)
            if main_window is not None:
                for other in main_window.plots:
                    other.start_x = shared_start_x
                    other.update_measurement_overlay(end_x)
            else:
                self.start_x = shared_start_x
                self.update_measurement_overlay(end_x)
            self.notify_measurement(delta_t)

    def mouseReleaseEvent(self, event:QMouseEvent)->None:
        if self.zoom_start_x is not None and self.zoom_region_temp is not None:
            # finalize zoom
            start, end = self.zoom_region_temp.getRegion()
            if start != end:
                self.setXRange(min(start, end), max(start, end), padding=0)

                main_window = self.get_main_window()
                if main_window is not None:
                    main_window.windowWidth = max(start, end) - min(start, end)
                    main_window.tPos = (max(start, end) + min(start, end)) / 2
                    main_window.updateView()

            # remove temporary region
            self.removeItem(self.zoom_region_temp)
            self.zoom_region_temp = None
            self.zoom_start_x = None
        else:
            super().mouseReleaseEvent(event)

    def format_time(self, dt_seconds:float)->str:
        """Return a human-readable time string for Δt."""
        if dt_seconds >= 1:
            return f"{dt_seconds:.3f} s"
        if dt_seconds >= 1e-3:
            return f"{dt_seconds * 1e3:.3f} ms"
        if dt_seconds >= 1e-6:
            return f"{dt_seconds * 1e6:.3f} µs"
        return f"{dt_seconds * 1e9:.3f} ns"

    def leaveEvent(self, event:QMouseEvent)->None:
        """Hide cursor and label when mouse leaves the widget."""
        self.last_cursor_x = None
        self.cursor_line.hide()
        self.timestamp_label.hide()
        self.point_label.hide()

        if self.dataLoaded:
            self.hideCursor()
            main_window = self.get_main_window()
            if main_window is not None:
                for other in main_window.plots:
                    other.hideCursor()
                    other.timestamp_label.setText("")
                main_window.update_status(cursor_time=None)

        super().leaveEvent(event)

    def enterEvent(self, event:QMouseEvent)->None:
        """Hide cursor and label when mouse leaves the widget."""

        self.timestamp_label.show()
        self.point_label.show()

        if self.dataLoaded:
            self.showCursor()
            main_window = self.get_main_window()
            if main_window is not None:
                for other in main_window.plots:
                    other.showCursor()

        super().enterEvent(event)

    def showEvent(self, event: QShowEvent) -> None:
        self._last_refresh_key = None
        self.schedule_curve_refresh()
        super().showEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._last_refresh_key = None
        self._resize_in_progress = True
        self.schedule_curve_refresh_delayed(120)
        super().resizeEvent(event)

    def showCursor(self)->None:
        self.dataLoaded = True
        self.cursor_line.show()

    def hideCursor(self)->None:
        self.cursor_line.hide()
