import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QMouseEvent, QPainter


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
        self.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.scene().installEventFilter(self)  # for enter/leave detection

        self.dataLoaded = False

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

    def mousePressEvent(self, event:QMouseEvent)->None:
        mouse_point = self.getViewBox().mapSceneToView(event.position())
        if self.measure_mode:
            if self.start_x is None:
                if hasattr(self.parent().parent().parent(), "plots"):
                    for other in self.parent().parent().parent().plots:
                        if other == self:
                            continue
                        other.measure_mode = False

                # First click → start measuring
                self.start_x = mouse_point.x()

                # Create temporary region
                self.temp_region = pg.LinearRegionItem(
                    values=(self.start_x, self.start_x),
                    movable=False,
                    brush=(50, 50, 200, 50),
                )
                self.addItem(self.temp_region, ignoreBounds=True)

                # Text for delta time
                self.temp_text = pg.TextItem("", color="b", anchor=(0.5, 1))
                self.addItem(self.temp_text, ignoreBounds=True)

            else:
                # Second click → finish measurement
                self.measure_mode = False
                self.start_x = None  # reset start for next measurement
                if hasattr(self.parent().parent().parent(), "plots"):
                    for other in self.parent().parent().parent().plots:
                        if other == self:
                            continue
                        other.start_x = None
                        other.measure_mode = False

        elif self.zoom_mode:
            self.zoom_start_x = mouse_point.x()
            # create temporary region
            self.zoom_region_temp = pg.LinearRegionItem(
                values=(self.zoom_start_x, self.zoom_start_x),
                movable=False,
                brush=(100, 100, 100, 50),
            )
            self.addItem(self.zoom_region_temp, ignoreBounds=True)

        elif not self.zoom_mode and not self.measure_mode:
            self.enable_zoom_mode()
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

    def enable_measure_mode(self,*, enabled:None|bool=None)->None:
        if enabled is None:
            enabled = True

        """Activate measurement mode and clear previous measurement."""
        self.measure_mode = enabled
        self.start_x = None

        # Remove previous measurement items
        if self.temp_region:
            self.removeItem(self.temp_region)
            self.temp_region = None
        if self.temp_text:
            self.removeItem(self.temp_text)
            self.temp_text = None

    def enable_zoom_mode(self,*, enabled:None|bool=None)->None:
        if enabled is None:
            enabled = True

        self.zoom_mode = enabled
        self.zoom_start_x = None
        if self.zoom_region_temp:
            self.removeItem(self.zoom_region_temp)
            self.zoom_region_temp = None

    def get_curves(self)->list:
        """
        Returns a list of tuples: (PlotDataItem, x_data, y_data)
        """
        curves = []
        for item in self.plotItem.listDataItems():
            x, y = item.getData()
            curves.append((item, x, y))
        return curves

    def on_mouse_moved(self, pos:QPointF)->None:
        if self.zoom_mode and self.zoom_start_x is not None:
            mouse_point = self.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()
            self.zoom_region_temp.setRegion((self.zoom_start_x, x))
            return

        """Move the cursor line when mouse moves inside the plot."""
        if self.sceneBoundingRect().contains(pos):
            mouse_point = self.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()
            self.cursor_line.setPos(mouse_point.x())

            mouse_point = self.plotItem.vb.mapSceneToView(pos)
            x_val = mouse_point.x()
            y_val = mouse_point.y()

            # Loop over all curves to find the nearest point

            yS = []
            names = []
            label = ""
            for _, (curve, cx, cy) in enumerate(self.get_curves()):
                # Find nearest index

                mask = cx <= x_val
                if np.any(mask):  # noqa: SIM108
                    # Pick the last index where cx is less than or equal to x_val
                    nearest_idx = np.where(mask)[0][-1]
                else:
                    # Fallback: all cx are greater, so pick the first one
                    nearest_idx = 0

                # nearest_idx = np.abs(cx - x_val).argmin()
                yS.append(cy[np.minimum(nearest_idx + 1, cy.size - 1)])
                names.append(curve.name() or "Unnamed")

                label += f"{names[-1]}:{yS[-1]:.2f} "

            self.point_label.setText(label)
            self.point_label.setPos(x_val, y_val)

        if hasattr(self.parent().parent().parent(), "plots"):
            for other in self.parent().parent().parent().plots:
                other.cursor_line.setPos(mouse_point.x())
                # Update timestamp label at bottom-right corner
                view_rect = other.viewRect()
                other.timestamp_label.setPos(view_rect.right(), view_rect.bottom())
                other.timestamp_label.setText(f"t = {x * 1e3:.2f} ms")

        # Update timestamp label at bottom-right corner
        view_rect = self.viewRect()
        self.timestamp_label.setPos(view_rect.right(), view_rect.bottom())
        self.timestamp_label.setText(f"t = {x * 1e3:.2f} ms")

        # Update measurement region dynamically
        if self.measure_mode and self.start_x is not None:
            end_x = mouse_point.x()
            # Update LinearRegionItem
            self.temp_region.setRegion((self.start_x, end_x))
            # Update text at midpoint
            mid_x = (self.start_x + end_x) / 2
            bottom_y = self.viewRange()[1][0]  # bottom of Y-axis
            delta_t = abs(end_x - self.start_x)

            self.temp_text.setText(f"Δt = {self.format_time(delta_t)}")
            self.temp_text.setPos(mid_x, bottom_y)

    def mouseReleaseEvent(self, event:QMouseEvent)->None:
        if self.zoom_mode and self.zoom_start_x is not None:
            # finalize zoom
            start, end = self.zoom_region_temp.getRegion()
            if start != end:
                self.setXRange(min(start, end), max(start, end), padding=0)

                if hasattr(self.parent().parent().parent(), "plots"):
                    self.parent().parent().parent().windowWidth = max(start, end) - min(start, end)
                    self.parent().parent().parent().tPos = (max(start, end) + min(start, end)) / 2

                    self.parent().parent().parent().updateView()

            # remove temporary region
            self.removeItem(self.zoom_region_temp)
            self.zoom_region_temp = None
            self.zoom_start_x = None
            self.zoom_mode = False
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
        self.cursor_line.hide()
        self.timestamp_label.hide()
        self.point_label.hide()

        if self.dataLoaded:
            self.hideCursor()
            if hasattr(self.parent().parent().parent(), "plots"):
                for other in self.parent().parent().parent().plots:
                    other.hideCursor()
                    other.timestamp_label.setText("")

        super().leaveEvent(event)

    def enterEvent(self, event:QMouseEvent)->None:
        """Hide cursor and label when mouse leaves the widget."""

        self.timestamp_label.show()
        self.point_label.show()

        if self.dataLoaded:
            self.showCursor()
            if hasattr(self.parent().parent().parent(), "plots"):
                for other in self.parent().parent().parent().plots:
                    other.showCursor()

        super().leaveEvent(event)

    def showCursor(self)->None:
        self.dataLoaded = True
        self.cursor_line.show()

    def hideCursor(self)->None:
        self.cursor_line.hide()
