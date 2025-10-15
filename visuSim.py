import sys
from pathlib import Path
from simUtils import readGrads, readRFEvents
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, uic, QtGui
from PyQt6.QtCore import Qt, QSettings, QDir, QCoreApplication,QPoint,QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
)
import os

from mulitPlotCursor import CursorPlot


class GUIapp(QMainWindow):
    windowWidth = 1e-2
    sliderScaler = 1

    highlightRect = None
    plots=[]
    
    def __init__(
        self,
        simPath="",
        outputPath="testIns",
        config=None
    ):
        super().__init__()
        path = Path(__file__).resolve().parent / "visusimForm.ui"
        uic.loadUi(path, self)

        self.dataPath = simPath
        self.setMouseTracking(True)

        
        self.plotGrads = CursorPlot()
        self.plotRF = CursorPlot()
        self.phasePlot = CursorPlot()
        self.imageLayout.insertWidget(0, self.phasePlot)
        self.imageLayout.insertWidget(1, self.plotRF)
        self.imageLayout.insertWidget(2, self.plotGrads)

        self.plots.append(self.phasePlot)
        self.plots.append(self.plotRF)
        self.plots.append(self.plotGrads)
        
        

    

        # Get the ViewBox of the PlotWidget
        vb = self.plotGrads.getViewBox()
        vb.setMouseEnabled(x=False, y=True)
        # vb.setLimits(xMin=0, xMax=10)  # restrict horizontal panning

        vb = self.plotRF.getViewBox()
        vb.setMouseEnabled(x=False, y=True)
        # vb.setLimits(xMin=0, xMax=10)  # restrict horizontal panning

        vb = self.phasePlot.getViewBox()
        vb.setMouseEnabled(x=False, y=True)
        # vb.setLimits(xMin=0, xMax=10)  # restrict horizontal panning

        penR = pg.mkPen(color=(200, 0, 0), width=1.5)
        penG = pg.mkPen(color=(0, 200, 0), width=1.5)
        penB = pg.mkPen(color=(0, 0, 200), width=1.5)
        penY = pg.mkPen(color=(200, 200, 0), width=1.5)

        self.time = []
        self.navigatorH = []
        self.navigatorB = []
        self.tPos = self.windowWidth * 0.5

        self.plotGrads.setLabel("left", "Gradients")
        self.plotRF.setLabel("left", "RF Amp + Rx")
        self.phasePlot.setLabel("left", "RF Phase")
        self.phasePlot.setYRange(0, 360)
        self.plotGrads.setLabel("bottom", "Time (s)")
        self.plotGrads.addLegend(offset=(10, 10))

        self.lineGx = self.plotGrads.plot(
            self.time, self.navigatorH, name="Gx", pen=penG
        )
        self.lineGy = self.plotGrads.plot(
            self.time, self.navigatorH, name="Gy", pen=penR
        )
        self.lineGz = self.plotGrads.plot(
            self.time, self.navigatorH, name="Gz", pen=penB
        )

        self.lineRFA = self.plotRF.plot(
            self.time, self.navigatorH, name="RFA", pen=penR
        )
        self.lineRFP = self.plotRF.plot(
            self.time, self.navigatorH, name="RFP", pen=penG
        )

        self.lineRx = self.plotRF.plot(self.time, self.navigatorH, name="Rx", pen=penG)

        self.lineTxPh = self.phasePlot.plot(
            self.time, self.navigatorH, name="Tx Phase", pen=penR
        )
        self.lineTxPow = self.phasePlot.plot(
            self.time, self.navigatorH, name="Tx Power", pen=penY
        )
        self.lineRxPh = self.phasePlot.plot(
            self.time, self.navigatorH, name="Rx Phase", pen=penG
        )

        self.plotGrads.setBackground("w")
        self.plotRF.setBackground("w")
        self.phasePlot.setBackground("w")

        self.tSlider = QtWidgets.QSlider(Qt.Orientation.Horizontal)

        self.jumpPButton = QtWidgets.QPushButton()
        self.jumpNButton = QtWidgets.QPushButton()

        self.jumpPButton.setText("+")
        self.jumpNButton.setText("-")

        self.jumpPButton.setFixedHeight(50)  # pixels
        self.jumpNButton.setFixedHeight(50)

        self.jumpNButton.clicked.connect(self.jumpXNeg)
        self.jumpPButton.clicked.connect(self.jumpXPos)

        self.tSlider.sliderReleased.connect(self.changeXRange)
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
        self.zoomInButton.setFixedHeight(50)
        self.zoomOutButton.setFixedHeight(50)
        self.zoomInButton.clicked.connect(self.zoomIn)
        self.zoomOutButton.clicked.connect(self.zoomOut)

        
        self.measureButton = QtWidgets.QPushButton("Measure dur")
        self.measureButton.clicked.connect(self.activate_measure)
        
        # Layout
        buttonLayout = QtWidgets.QHBoxLayout()
        buttonLayout.addWidget(self.jumpNButton)
        buttonLayout.addWidget(self.jumpPButton)
        buttonLayout.addWidget(self.zoomOutButton)
        buttonLayout.addWidget(self.zoomInButton)
        buttonLayout.addWidget(self.measureButton)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(buttonLayout)
        layout.addWidget(self.tSlider)

        self.imageLayout.insertLayout(3, layout)

        # Menu bar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("File")

        # Add "Open Folder" action
        openFolderAction = QtGui.QAction("Open Folder", self)
        openFolderAction.triggered.connect(self.open_folder)
        fileMenu.addAction(openFolderAction)

        # Create settings object
        self.settings = QSettings("MR_ISIBrno", "BrukerSimView")

        # Read a value (with a default)
        self.dataPath = self.settings.value("lastFolder", QDir.homePath())



    def activate_measure(self):
        self.plots[-1].enable_measure_mode(True)


    def open_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Analysis Folder", self.dataPath
        )
        if folder_path:
            self.dataPath = folder_path

            if not os.path.exists(self.dataPath + "/" + "_GCube.xml"):
                self.showErrorMessage("No GCube file found in folder!")
                return

            if not os.path.exists(self.dataPath + "/" + "_FCube1.xml"):
                self.showErrorMessage("No GCube file found in folder!")
                return

            self.settings.setValue("lastFolder", self.dataPath)

            self.loadData()

    def zoomIn(self):
        self.windowWidth *= 0.8
        self.updateView()

    def zoomOut(self):
        self.windowWidth /= 0.8
        self.updateView()

    def askYesNo(self, text):
        # Create a message box
        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Question)  # Set the icon to Question
        msg_box.setWindowTitle("Confirmation")  # Set the title of the message box
        msg_box.setText(text)  # Main question
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No
        )  # Add Yes and No buttons

        user_choice = msg_box.exec()

        return user_choice == QtWidgets.QMessageBox.StandardButton.Yes

    def showErrorMessage(self, errorMessage):
        # Create a message box
        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(
            QtWidgets.QMessageBox.Icon.Critical
        )  # Set the icon to Critical (error)
        msg_box.setWindowTitle("Error")  # Set the title of the message box
        msg_box.setText("An error occurred!")  # Set the main text
        msg_box.setInformativeText(errorMessage)  # Optional additional text
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Ok
        )  # Add standard buttons
        msg_box.exec()  # Display the message box

    def changeXRange(self):
        self.tPos = self.tSlider.value() * self.sliderScaler
        self.updateView()

    def jumpXPos(self):
        self.tPos = np.minimum(self.tMax, self.tPos + self.windowWidth * 0.5)
        self.tSlider.setValue(int(self.tPos / self.sliderScaler))
        self.updateView()

    def jumpXNeg(self):
        self.tPos = np.maximum(self.tMin, self.tPos - self.windowWidth * 0.5)
        self.tSlider.setValue(int(self.tPos / self.sliderScaler))
        self.updateView()

    def updateView(self):
        rangePos = self.tPos + self.windowWidth * 0.5
        rangeNeg = self.tPos - self.windowWidth * 0.5

        self.plotRF.setXRange(rangeNeg, rangePos)
        self.plotGrads.setXRange(rangeNeg, rangePos)
        self.phasePlot.setXRange(rangeNeg, rangePos)

    def loadData(self):
        progress = QtWidgets.QProgressDialog(
            "Parsing simulation data ...", "Cancel", 0, 100, self
        )
        progress.setWindowTitle("Please Wait")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)  # show immediately
        progress.show()
        progress.setValue(5)
        QCoreApplication.processEvents()

        progress.setLabelText("Reading RF Events")

        if progress.wasCanceled():
            return
        self.ncos, info = readRFEvents(self.dataPath)
        self.info = info

        self.setWindowTitle(f"{self.dataPath} originPPG: {info['pulProg']}")

        if progress.wasCanceled():
            return
        progress.setValue(40)
        progress.setLabelText("Reading gradients")

        self.gradTime, self.grads = readGrads(self.dataPath)

        progress.setValue(50)

        progress.setLabelText("Preparing plots gradients")

        self.tMax = np.max([self.ncos["1"]["t"][-1]])
        self.tMin = 0
        self.sliderScaler = self.tMax / self.tSlider.maximum()

        self.initPlots()
        progress.setValue(90)
        progress.close()
        QtWidgets.QMessageBox.information(self, "Done", "Loading finished!")
        
        
        for plot in self.plots:
            plot.showCursor()

    def makeStepArrs(self, tArr, multArr):
        stepTime = np.repeat(tArr, 2)[1:]
        stepGrads = np.repeat(multArr, 2, -1)[:, :-1]

        return stepTime, stepGrads


    def convertToStep(self,dict):
        
        dict["t"] = np.repeat(dict["t"], 2)[1:]
        
        for key in dict:
            if not isinstance(dict[key],np.ndarray):
                continue
            
            if key =="t":
                continue
            
            dict[key] = np.repeat(dict[key], 2, -1)[ :-1]
    
        return dict    

    def initPlots(self):
        stepTime, stepGrads = self.makeStepArrs(self.gradTime, self.grads)

        # stepRFTime, stepRFs = self.makeStepArrs(self.RFTime, self.TxEv)

        # stepRxTime, stepRxs = self.makeStepArrs(self.RxTime, self.RxEv)
        
        # stepTxPTime, stepTxEvs = self.makeStepArrs(self.TxPTime, self.TxPEv)
        self.ncosStep={}
        
        for nco in self.ncos:
            self.ncosStep[nco] = self.convertToStep(self.ncos[nco])
        
        nco1Time = self.ncosStep["1"]["t"]
        nco1Amp = self.ncosStep["1"]["am"]
        
        
        
        
        nco3Time = self.ncosStep["3"]["t"]
        nco3Rgp = self.ncosStep["3"]["rgp"] 
        nco3p0 = self.ncosStep["3"]["p0"] 
        
        
            
        pow = self.ncosStep["1"]["pw"]
        
        
        
        sf = self.ncosStep["1"]["sf"]
        t = self.ncosStep["1"]["t"]

        # Compute differences to find where frequency changes
        dsf = sf - sf[np.where(sf>0)[0][0]]
        
        whenChange = np.abs(np.diff(dsf, prepend=0)) > 0

        # Extract change values and corresponding time points
        sfChanges = dsf[whenChange]
        tsfChanges = t[whenChange]

        # Plot vertical markers and labels for frequency changes
        for ind, sfChange in enumerate(sfChanges):
            x = tsfChanges[ind]
            freq = sfChange

            # Optional: draw a vertical line marker
            line = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine))
            self.plotRF.addItem(line)

            # Add a text label slightly above the bottom
            text = pg.TextItem(f"f = {freq*1e3:.2f} kHz", anchor=(0, 0), color='r')
            text.setPos(x, 110)  # adjust vertical offset if needed
            self.plotRF.addItem(text)


        self.lineGx.setData(stepTime, stepGrads[0, :])
        self.lineGy.setData(stepTime, stepGrads[1, :])
        self.lineGz.setData(stepTime, stepGrads[2, :])

        self.lineRFA.setData(nco1Time, nco1Amp)
        self.lineRFP.setData(nco3Time, nco3p0)

        self.lineRx.setData(nco3Time, nco3Rgp * 100)

        self.lineTxPh.setData(nco1Time, self.ncosStep["1"]["p0"])
        self.lineTxPow.setData(nco1Time, pow)

        self.updateView()

        self.tSlider.setValue(int(self.tPos / self.sliderScaler))


if __name__ == "__main__":
    inputArgs = sys.argv

    app = QApplication(sys.argv)  # ✅ Must be first
    gui = GUIapp("/mnt/c/Users/vitou/Documents/mrScanSim/")  # ✅ Now it's safe
    gui.show()
    sys.exit(app.exec())
