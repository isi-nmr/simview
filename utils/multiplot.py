import numpy as np
import pyqtgraph as pg

from widgets.mulitPlotCursor import CursorPlot


def makePens()->tuple[list, dict]:
    penR = pg.mkPen(color=(240, 0, 0), width=1.5)
    penG = pg.mkPen(color=(0, 240, 0), width=1.5)
    penB = pg.mkPen(color=(0, 0, 240), width=1.5)
    penY = pg.mkPen(color=(145, 125, 0), width=1.5)
    penC = pg.mkPen(color=(0, 200, 200), width=1.5)  # Cyan
    penM = pg.mkPen(color=(200, 0, 200), width=1.5)  # Magenta
    penO = pg.mkPen(color=(255, 128, 0), width=1.5)  # Orange
    penP = pg.mkPen(color=(150, 0, 150), width=1.5)  # Purple
    penTeal = pg.mkPen(color=(0, 150, 120), width=1.5)  # Teal
    penPink = pg.mkPen(color=(255, 105, 180), width=1.5)  # Pink
    penGray = pg.mkPen(color=(128, 128, 128), width=1.5)  # Gray
    penBrown = pg.mkPen(color=(139, 69, 19), width=1.5)  # Brown

    pens = [penR, penG, penB, penY, penC, penM, penO, penP, penTeal, penPink, penGray, penBrown]

    penDict = {"r": penR, "g": penG, "b": penB, "y": penY}

    return pens, penDict


def addAnnotations(line:dict, currentPlot: CursorPlot)->None:
    for annotation in line["annotations"]:
        if "texts" in annotation:
            color = str(annotation.get("color", "r"))
            for ind, t in enumerate(annotation["t"]):
                text_value = annotation["texts"][ind] if ind < len(annotation["texts"]) else str(annotation["texts"][-1])
                currentPlot.add_annotation_marker(
                    float(t),
                    str(text_value),
                    color=color,
                )
            continue

        for ind, t in enumerate(annotation["t"]):
            currentPlot.add_annotation_marker(
                float(t),
                f"f = {annotation['vals'][ind]:.2f} {annotation['units']}",
                color="r",
            )

def makeStepArrs(tArr:np.ndarray, multArr:np.ndarray)->np.ndarray:
    stepTime = np.repeat(tArr, 2)[1:]
    stepGrads = np.repeat(multArr, 2, -1)[:, :-1]

    return stepTime, stepGrads

def convertToStep(dict:dict, key:str)->dict:
    newDict = {}
    newDict["t"] = np.repeat(dict["t"], 2)[1:]

    if not isinstance(dict[key], np.ndarray):
        newDict[key] = dict[key]

    else:
        newDict[key] = np.repeat(dict[key], 2, -1)[:-1]

    return newDict
