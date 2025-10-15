import sys
from pathlib import Path
from simUtils import readGrads, readRFEvents
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, uic, QtGui
from PyQt6.QtCore import Qt, QSettings, QDir, QCoreApplication
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

import os

from mulitPlotCursor import CursorPlot
import re


class GUIapp(QMainWindow):
    windowWidth = 1e-2
    sliderScaler = 1

    highlightRect = None
    plots = []
    channels = []
    checkBoxes = []
    displayChannels = []
    plotGroups = []
    plotContainers = []

    def __init__(self, simPath=None):
        super().__init__()
        path = Path(__file__).resolve().parent / "visusimForm.ui"
        uic.loadUi(path, self)

        self.dataPath = simPath
        self.setMouseTracking(True)

        self.tPos = self.windowWidth * 0.5

        # self.plotGrads.setBackground("w")
        # self.plotRF.setBackground("w")
        # self.phasePlot.setBackground("w")

        self.tSlider = QtWidgets.QSlider(Qt.Orientation.Horizontal)

        self.jumpPButton = QtWidgets.QPushButton()
        self.jumpNButton = QtWidgets.QPushButton()

        self.jumpPButton.setText("->")
        self.jumpNButton.setText("<-")

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

        buttonLayout.addWidget(self.zoomOutButton)
        buttonLayout.addWidget(self.zoomInButton)
        buttonLayout.addWidget(self.jumpPButton)

        buttonLayout.addWidget(self.measureButton)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(buttonLayout)
        layout.addWidget(self.tSlider)
        self.navigation = layout

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

        for plot in self.plots:
            plot.setXRange(rangeNeg, rangePos)

    def loadData(self):
        self.plotGroups = []
        self.plots = []
        self.checkBoxes = []
        self.channels = []
        self.displayChannels = []

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

        for nco in self.ncos:
            for key in self.ncos[nco]:
                if key == "t" or key == "sf":
                    continue

                if re.match(r"p\d", key):
                    plotType = "phase"
                elif key == "pw":
                    plotType = "power"
                else:
                    plotType = "mag"

                channelDes = {
                    "label": "NCO_" + nco + "_" + key,
                    "type": "NCO",
                    "ind": nco,
                    "key": key,
                    "plotType": plotType,
                    "t": self.ncos[nco]["t"],
                    "data": self.ncos[nco][key],
                }
                
                channelDes["annotations"]=[]
                
                if key =="am":
                    sf = self.ncos[nco]["sf"]
                    t = self.ncos[nco]["t"]

                    # Compute differences to find where frequency changes
                    dsf = sf - sf[np.where(sf>0)[0][0]]

                    whenChange = np.abs(np.diff(dsf, prepend=0)) > 0

                    # Extract change values and corresponding time points
                    sfChanges = dsf[whenChange]
                    tsfChanges = t[whenChange]
                    if len(sfChanges>0):
                        channelDes["annotations"].append({"name":"sf","t":tsfChanges,"vals":sfChanges*1e3,"units":"kHz"})


                    
                
                self.channels.append(channelDes)
                self.displayChannels.append(self.channels[-1]["label"])

        self.channels.append(
            {
                "label": "grads_1_Gx",
                "type": "grads",
                "ind": str(0),
                "key": "Gx",
                "plotType": "mag",
                "t": self.gradTime,
                "data": self.grads[0],
                "annotations":[]
            }
        )
        self.displayChannels.append(self.channels[-1]["label"])
        self.channels.append(
            {
                "label": "grads_2_Gy",
                "type": "grads",
                "ind": str(1),
                "key": "Gy",
                "plotType": "mag",
                "t": self.gradTime,
                "data": self.grads[1],
                 "annotations":[]
            }
        )
        self.displayChannels.append(self.channels[-1]["label"])
        self.channels.append(
            {
                "label": "grads_3_Gz",
                "type": "grads",
                "ind": str(2),
                "key": "Gz",
                "plotType": "mag",
                "t": self.gradTime,
                "data": self.grads[2],
                 "annotations":[]
            }
        )
        self.displayChannels.append(self.channels[-1]["label"])

        self.registerCheckBoxes()

        self.initPlots()
        progress.setValue(90)
        progress.close()
        QtWidgets.QMessageBox.information(self, "Done", "Loading finished!")

        for plot in self.plots:
            plot.showCursor()

    def registerCheckBoxes(self):
        if hasattr(self, "checkBoxLayout"):
            while self.checkBoxLayout.count():
                item = self.checkBoxLayout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        self.checkBoxLayout = QtWidgets.QHBoxLayout()

        for channel in self.channels:
            checkBox = QtWidgets.QCheckBox()
            checkBox.setText(channel["label"])
            checkBox.setChecked(True)
            checkBox.stateChanged.connect(self.checkBoxChanged)
            self.checkBoxLayout.addWidget(checkBox)
            self.checkBoxes.append(checkBox)

    def checkBoxChanged(self):
        self.displayChannels = []

        for checkBox in self.checkBoxes:
            if not checkBox.isChecked():
                self.plotContainers[checkBox.contID].hide()
            else:
                self.plotContainers[checkBox.contID].show()

    def makeStepArrs(self, tArr, multArr):
        stepTime = np.repeat(tArr, 2)[1:]
        stepGrads = np.repeat(multArr, 2, -1)[:, :-1]

        return stepTime, stepGrads

    def convertToStep(self, dict, key):
        newDict = {}
        newDict["t"] = np.repeat(dict["t"], 2)[1:]

        if not isinstance(dict[key], np.ndarray):
            newDict[key] = dict[key]

        else:
            newDict[key] = np.repeat(dict[key], 2, -1)[:-1]

        return newDict

    def initPlots(self):
        self.imageLayout.removeItem(self.navigation)

        penR = pg.mkPen(color=(200, 0, 0), width=1.5)
        penG = pg.mkPen(color=(0, 200, 0), width=1.5)
        penB = pg.mkPen(color=(0, 0, 200), width=1.5)
        penY = pg.mkPen(color=(200, 200, 0), width=1.5)

        pens = [penR, penG, penB, penY]
        hasGrads = False
        
        for chanInd, channel in enumerate(self.channels):
            if (channel["type"] == "grads" and not hasGrads) or channel[
                "type"
            ] != "grads":
                currentPlot = CursorPlot()
                phaseContainer = QWidget()
                phaseContainer_layout = QVBoxLayout(phaseContainer)
                phaseContainer_layout.setContentsMargins(0, 0, 0, 0)
                phaseContainer_layout.addWidget(currentPlot)
                vb = currentPlot.getViewBox()
                vb.setMouseEnabled(x=False, y=True)
                self.plots.append(currentPlot)
                self.plotContainers.append(phaseContainer)
                self.imageLayout.addWidget(phaseContainer, stretch=1)

            
            self.checkBoxes[chanInd].contID = len(self.plotContainers)-1
            
            if chanInd >= len(self.channels) - 1:
                currentPlot.setLabel("bottom", "Time (s)")

            stepData = self.convertToStep(channel, "data")

            if np.sum(np.abs(stepData["data"])) == 0:
                phaseContainer.hide()
                self.checkBoxes[chanInd].blockSignals(True)
                self.checkBoxes[chanInd].setChecked(False)
                self.checkBoxes[chanInd].blockSignals(False)

            if channel["plotType"] == "phase":
                currentPlot.setYRange(0, 360)

            if channel["type"]=="grads" and not hasGrads:
                currentPlot.addLegend(offset=(10, 10))
                currentPlot.setLabel("bottom", "Time (s)")
                
            if channel["type"]!="grads":
                currentPlot.setLabel("left", channel["label"])
                currentPlot.plot(
                    stepData["t"],
                    stepData["data"],
                    name=channel["label"],
                    pen=pens[chanInd % 4],
                )
            else:
                hasGrads=True
                currentPlot.setLabel("left", "Grads")
                currentPlot.plot(
                    stepData["t"],
                    stepData["data"],
                    name=channel["key"],
                    pen=pens[chanInd % 4],
                )    
            
            
            if len(channel["annotations"])>0:
                
                for annotation in channel["annotations"]:
                    for ind, t in enumerate(annotation["t"]):
                        
                        line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine))
                        currentPlot.addItem(line)  

                        text = pg.TextItem(f"f = {annotation["vals"][ind]:.2f} {annotation["units"]}", anchor=(0, 0), color='r')
                        text.setPos(t, 110)  # adjust vertical offset if needed
                        currentPlot.addItem(text)
            
        self.updateView()

        self.tSlider.setValue(int(self.tPos / self.sliderScaler))

        self.imageLayout.addLayout(self.navigation)
        self.imageLayout.addLayout(self.checkBoxLayout)


if __name__ == "__main__":
    inputArgs = sys.argv

    app = QApplication(sys.argv)  # ✅ Must be first
    gui = GUIapp("/mnt/c/Users/vitou/Documents/mrScanSim/")  # ✅ Now it's safe
    gui.show()
    sys.exit(app.exec())
