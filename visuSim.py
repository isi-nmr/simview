import sys
from pathlib import Path
from simUtilsBrkr import readBrkrChannels
from simUtilsNMRScopeB import readNMRScopeBChannels
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, uic, QtGui
from PyQt6.QtCore import Qt, QSettings, QDir, QCoreApplication
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

import os

from mulitPlotCursor import CursorPlot


class GUIapp(QMainWindow):
    windowWidth = 1e-2
    sliderScaler = 1

    highlightRect = None
    plots = []
    channels = []
    checkBoxes = []
    plotContainers = []

    def __init__(self, simPath=None):
        super().__init__()
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
        self.measureButton.setFixedHeight(50)

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
        
        
        self.leftMenu = QVBoxLayout()
        
        self.horizontalLayout_2.insertLayout(0,self.leftMenu)


    def activate_measure(self):
        self.plots[-1].enable_measure_mode(True)

    def open_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Simulation output folder", self.dataPath
        )
        if folder_path:
            self.dataPath = folder_path
            if os.path.exists(self.dataPath + "/" + "pulse_seq.json"):
                # NMRScopeBPath
                self.settings.setValue("lastFolder", self.dataPath)
                self.loadData()
                return
            
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
        self.plots = []
        self.checkBoxes = []
        self.channels = []


        progress = QtWidgets.QProgressDialog(
            "Parsing simulation data ...", "Cancel", 0, 100, self
        )
        progress.setWindowTitle("Please Wait")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)  # show immediately
        progress.show()
        progress.setValue(5)
        QCoreApplication.processEvents()

        if os.path.exists(self.dataPath + "/" + "pulse_seq.json"):
            self.channels = readNMRScopeBChannels(self.dataPath, progress, self)        
        else:
            self.channels = readBrkrChannels(self.dataPath, progress, self)
        
        self.tMax = 0
        for channel in self.channels:
            self.tMax = np.maximum(self.tMax, channel[0]["t"][-1].item())

        self.tMin = 0
        self.sliderScaler = self.tMax / self.tSlider.maximum()

        self.registerCheckBoxes()

        self.initPlots()
        progress.setValue(90)
        progress.close()
        QtWidgets.QMessageBox.information(self, "Done", "Loading finished!")

        for plot in self.plots:
            plot.showCursor()

    def registerCheckBoxes(self):
        if hasattr(self, "leftMenu"):
            while self.leftMenu.count():
                item = self.leftMenu.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        for channel in self.channels:
            checkBox = QtWidgets.QCheckBox()
            checkBox.setText(channel[0]["chanLabel"])
            checkBox.setChecked(True)
            checkBox.stateChanged.connect(self.checkBoxChanged)
            self.leftMenu.addWidget(checkBox)
            self.checkBoxes.append(checkBox)

    def checkBoxChanged(self):


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
        
        penDict = {"r":penR,"g":penG,"b":penB,"y":penY}
        
        hasGrads = False

        for chanInd, channel in enumerate(self.channels):

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
            
            plotItem = currentPlot.getPlotItem()
            plotItem.showAxis('left', True)
            plotItem.showAxis('right', True)
            axis = currentPlot.getPlotItem().getAxis('right')
            axis.setWidth(50)  # fixed width in pixels
            axis = currentPlot.getPlotItem().getAxis('left')
            axis.setWidth(50)  # fixed width in pixels
                
            self.checkBoxes[chanInd].contID = len(self.plotContainers) - 1
            
            
            if chanInd >= len(self.channels) - 1:
                currentPlot.setLabel("bottom", "Time (s)")

            if channel[0]["plotType"] == "phase":
                currentPlot.setYRange(0, 360)
            
            if len(channel)>1:
                currentPlot.addLegend(offset=(10, 10))
            
            for line in channel:
                stepData = self.convertToStep(line, "data")

                if np.sum(np.abs(stepData["data"])) == 0:
                    phaseContainer.hide()
                    self.checkBoxes[chanInd].blockSignals(True)
                    self.checkBoxes[chanInd].setChecked(False)
                    self.checkBoxes[chanInd].blockSignals(False)

                currentPlot.setLabel("right", line["label"])
                
                
                currentPen = line.get("pen",pens[chanInd % 4])
                
                currentPlot.plot(
                    stepData["t"],
                    stepData["data"],
                    name=line["label"],
                    pen=currentPen,
                )

                if len(line["annotations"]) > 0:
                    for annotation in line["annotations"]:
                        for ind, t in enumerate(annotation["t"]):
                            line = pg.InfiniteLine(
                                pos=t,
                                angle=90,
                                pen=pg.mkPen("r", style=Qt.PenStyle.DashLine),
                            )
                            currentPlot.addItem(line)

                            text = pg.TextItem(
                                f"f = {annotation['vals'][ind]:.2f} {annotation['units']}",
                                anchor=(0, 0),
                                color="r",
                            )
                            text.setPos(t, 110)  # adjust vertical offset if needed
                            currentPlot.addItem(text)

        self.updateView()

        self.tSlider.setValue(int(self.tPos / self.sliderScaler))

        self.imageLayout.addLayout(self.navigation)


if __name__ == "__main__":
    inputArgs = sys.argv

    app = QApplication(sys.argv)  # ✅ Must be first
    gui = GUIapp("/mnt/c/Users/vitou/Documents/mrScanSim/")  # ✅ Now it's safe
    gui.show()
    sys.exit(app.exec())
