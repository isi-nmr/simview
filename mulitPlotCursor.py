
from PyQt6 import  QtCore
import pyqtgraph as pg
import numpy as np

class CursorPlot(pg.PlotWidget):
    """A PlotWidget with its own vertical cursor line."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create the vertical line cursor
        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=1))
        self.cursor_line.hide()
        self.addItem(self.cursor_line)

        self.timestamp_label = pg.TextItem("", anchor=(1, 0), color='r')
        self.addItem(self.timestamp_label, ignoreBounds=True)
        
        # Connect scene signals
        self.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.scene().installEventFilter(self)  # for enter/leave detection

        
        self.dataLoaded=False

        self.temp_region = None
        self.temp_text = None
        # Time measurement state
        self.measure_mode = False
        self.start_x = None
        self.measure_line = None
        self.measure_text = None


    def mousePressEvent(self, event):
        if self.measure_mode:
            mouse_point = self.getViewBox().mapSceneToView(event.position())
            if self.start_x is None:
                # First click → start measuring
                self.start_x = mouse_point.x()

                # Create temporary region
                self.temp_region = pg.LinearRegionItem(values=(self.start_x, self.start_x),
                                                       movable=False, brush=(50, 50, 200, 50))
                self.addItem(self.temp_region,ignoreBounds=True)

                # Text for delta time
                self.temp_text = pg.TextItem("", color='b', anchor=(0.5, 1))
                self.addItem(self.temp_text, ignoreBounds=True)
            else:
                # Second click → finish measurement
                self.measure_mode = False
                self.start_x = None  # reset start for next measurement
        else:
            super().mousePressEvent(event)
        

    def enable_measure_mode(self, enabled=True):
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
            
            
               
        
    def on_mouse_moved(self, pos):
        """Move the cursor line when mouse moves inside the plot."""
        if self.sceneBoundingRect().contains(pos):
            mouse_point = self.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()
            self.cursor_line.setPos(mouse_point.x())


        if hasattr(self.parent().parent(), 'plots'):
            for other in self.parent().parent().plots:
                other.cursor_line.setPos(mouse_point.x())
                # Update timestamp label at bottom-right corner
                view_rect = other.viewRect()
                other.timestamp_label.setPos(view_rect.right(), view_rect.bottom())
                other.timestamp_label.setText(f"t = {x*1e3:.2f} ms")
                
        # Update timestamp label at bottom-right corner
        view_rect = self.viewRect()
        self.timestamp_label.setPos(view_rect.right(), view_rect.bottom())
        self.timestamp_label.setText(f"t = {x*1e3:.2f} ms")


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


    def format_time(self,dt_seconds):
        """Return a human-readable time string for Δt."""
        if dt_seconds >= 1:
            return f"{dt_seconds:.3f} s"
        elif dt_seconds >= 1e-3:
            return f"{dt_seconds*1e3:.3f} ms"
        elif dt_seconds >= 1e-6:
            return f"{dt_seconds*1e6:.3f} µs"
        else:
            return f"{dt_seconds*1e9:.3f} ns"


    def leaveEvent(self, event):
        """Hide cursor and label when mouse leaves the widget."""
        self.cursor_line.hide()
        self.timestamp_label.setText("")

        if self.dataLoaded:
            self.hideCursor()
            if hasattr(self.parent().parent(), 'plots'):
                for other in self.parent().parent().plots: 
                    other.hideCursor()
                    other.timestamp_label.setText("")
        
        super().leaveEvent(event)


    def enterEvent(self, event):
        """Hide cursor and label when mouse leaves the widget."""
        
        if self.dataLoaded:
            self.showCursor()
            if hasattr(self.parent().parent(), 'plots'):
                for other in self.parent().parent().plots: 
                    other.showCursor()

        super().leaveEvent(event)
        
        
    def showCursor(self):
        self.dataLoaded=True
        self.cursor_line.show()
        
    def hideCursor(self):
        self.cursor_line.hide()        