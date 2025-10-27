import re

import numpy as np
import xmltodict
from PyQt6.QtWidgets import QMainWindow, QProgressDialog


def getGradEvents(dict: dict) -> tuple[np.ndarray, np.ndarray]:
    time = np.zeros(len(dict["pulseprogram"]["ev"]))

    tUnit = float(dict["pulseprogram"]["@timeunit"])

    grads = np.zeros((3, len(dict["pulseprogram"]["ev"])))
    ind = 0
    for event in dict["pulseprogram"]["ev"]:
        if "@g1" not in event:
            continue

        time[ind] = float(event["@t"]) * tUnit
        grads[:, ind] = [float(event["@g1"]), float(event["@g2"]), float(event["@g3"])]
        ind = ind + 1

    return time[:ind], grads[:, :ind]


def readGrads(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path + "/" + "_GCube.xml") as f:
        gCube = xmltodict.parse(f.read())

    time, grads = getGradEvents(gCube)

    return time, grads


def initNco(nEvents: int) -> dict:
    return {
        "t": np.zeros(nEvents),
        "p0": np.zeros(nEvents),
        "p1": np.zeros(nEvents),
        "p2": np.zeros(nEvents),
        "am": np.zeros(nEvents),
        "pw": np.zeros(nEvents),
        "sf": np.zeros(nEvents),
        "rgp": np.zeros(nEvents),
    }


def turnOffNco(ncos: dict, ncoNumber: int, t: np.ndarray, ind: int) -> None:
    ncos[ncoNumber]["t"][ind] = t
    ncos[ncoNumber]["p0"][ind] = float(ncos[ncoNumber]["p0"][ind - 1]) if ind > 0 else 0
    ncos[ncoNumber]["p1"][ind] = float(ncos[ncoNumber]["p1"][ind - 1]) if ind > 0 else 0
    ncos[ncoNumber]["p2"][ind] = float(ncos[ncoNumber]["p2"][ind - 1]) if ind > 0 else 0
    ncos[ncoNumber]["am"][ind] = 0
    ncos[ncoNumber]["pw"][ind] = 0
    ncos[ncoNumber]["sf"][ind] = float(ncos[ncoNumber]["sf"][ind - 1]) if ind > 0 else 0
    ncos[ncoNumber]["rgp"][ind] = 0


def getRFEvents(dict: dict) -> tuple[dict, dict]:
    ncos = {}

    info = {}
    info["pulProg"] = dict["pulseprogram"]["@path"].split("/")[-1]
    ind = 0
    tUnit = float(dict["pulseprogram"]["@timeunit"])

    indMaxes = {}

    ncoNumber = None

    for event in dict["pulseprogram"]["ev"]:
        if "@t" not in event:
            print("Skip event")
            continue

        if "@nco" in event:
            if event["@nco"] == "0":
                turnOffNco(
                    ncos,
                    ncoNumber,
                    float(event["@t"]) * tUnit,
                    indMaxes[str(ncoNumber)],
                )
                indMaxes[ncoNumber] += 1
                continue
            ncoNumber = event["@nco"]
        else:
            ncoNumber = str(1)

        if str(ncoNumber) not in ncos:
            ncos[str(ncoNumber)] = initNco(len(dict["pulseprogram"]["ev"]))
            indMaxes[str(ncoNumber)] = 0

        ind = indMaxes[str(ncoNumber)]

        ncos[ncoNumber]["t"][ind] = float(event["@t"]) * tUnit

        if "@p0" in event:
            ncos[ncoNumber]["p0"][ind] = float(event["@p0"])
        else:
            ncos[ncoNumber]["p0"][ind] = ncos[ncoNumber]["p0"][ind - 1] if ind > 0 else 0

        if "@p1" in event:
            ncos[ncoNumber]["p1"][ind] = float(event["@p1"])
        else:
            ncos[ncoNumber]["p1"][ind] = ncos[ncoNumber]["p1"][ind - 1] if ind > 0 else 0

        if "@p2" in event:
            ncos[ncoNumber]["p2"][ind] = float(event["@p2"])
        else:
            ncos[ncoNumber]["p2"][ind] = ncos[ncoNumber]["p2"][ind - 1] if ind > 0 else 0

        if "@pw" in event:
            ncos[ncoNumber]["pw"][ind] = float(event["@pw"])
        else:
            ncos[ncoNumber]["pw"][ind] = ncos[ncoNumber]["pw"][ind - 1] if ind > 0 else 0

        if "@am" in event:
            ncos[ncoNumber]["am"][ind] = float(event["@am"])
        else:
            ncos[ncoNumber]["am"][ind] = ncos[ncoNumber]["am"][ind - 1] if ind > 0 else 0

        if "@sf" in event:
            ncos[ncoNumber]["sf"][ind] = float(event["@sf"])
        else:
            ncos[ncoNumber]["sf"][ind] = ncos[ncoNumber]["sf"][ind - 1] if ind > 0 else 0

        if "@rgp" in event:
            ncos[ncoNumber]["rgp"][ind] = 1 if "0--0" in event["@rgp"] else 0

        elif "@ln" in event:
            if event["@ln"] == "8000001":
                ncos[ncoNumber]["rgp"][ind] = 0
            else:
                ncos[ncoNumber]["rgp"][ind] = ncos[ncoNumber]["rgp"][ind - 1] if ind > 0 else 0
        else:
            ncos[ncoNumber]["rgp"][ind] = ncos[ncoNumber]["rgp"][ind - 1] if ind > 0 else 0

        indMaxes[str(ncoNumber)] += 1

    for ncoKey, subDict in ncos.items():
        for key, value in subDict.items():
            if isinstance(value, np.ndarray):
                subDict[key] = value[: indMaxes[ncoKey]]

    return ncos, info


def readRFEvents(path: str) -> tuple[dict, dict]:
    with open(path + "/" + "_FCube1.xml") as f:
        gCube = xmltodict.parse(f.read())

    ncos, info = getRFEvents(gCube)

    return ncos, info


def readBrkrChannels(path: str, progress: QProgressDialog, app: QMainWindow)->dict:
    progress.setLabelText("Reading RF Events")

    if progress.wasCanceled():
        return None

    ncos, info = readRFEvents(path)

    app.setWindowTitle(f"{path} originPPG: {info['pulProg']}")

    if progress.wasCanceled():
        return None
    progress.setValue(40)
    progress.setLabelText("Reading gradients")

    gradTime, grads = readGrads(path)

    progress.setValue(50)

    progress.setLabelText("Preparing plots gradients")

    channels = []

    for nco in ncos:
        for key in ncos[nco]:
            if key in {"t", "sf"}:
                continue

            if re.match(r"p\d", key):
                plotType = "phase"
            elif key == "pw":
                plotType = "power"
            else:
                plotType = "mag"

            channelDes = {
                "chanLabel": "NCO_" + nco + "_" + key,
                "label": "NCO_" + nco + "_" + key,
                "type": "NCO",
                "ind": nco,
                "key": key,
                "plotType": plotType,
                "t": ncos[nco]["t"],
                "data": ncos[nco][key],
            }

            channelDes["annotations"] = []

            if key == "am":
                sf = ncos[nco]["sf"]
                t = ncos[nco]["t"]

                # Compute differences to find where frequency changes
                dsf = sf - sf[np.where(sf > 0)[0][0]]

                whenChange = np.abs(np.diff(dsf, prepend=0)) > 0

                # Extract change values and corresponding time points
                sfChanges = dsf[whenChange]
                tsfChanges = t[whenChange]
                if len(sfChanges > 0):
                    channelDes["annotations"].append(
                        {
                            "name": "sf",
                            "t": tsfChanges,
                            "vals": sfChanges * 1e3,
                            "units": "kHz",
                        }
                    )

            channels.append([channelDes])

    channels.append(
        [
            {
                "chanLabel": "Gradients",
                "label": "Gx",
                "type": "grads",
                "ind": str(0),
                "key": "Gx",
                "plotType": "mag",
                "t": gradTime,
                "data": grads[0],
                "annotations": [],
                "pen": "g",
            },
            {
                "label": "Gy",
                "type": "grads",
                "ind": str(1),
                "key": "Gy",
                "plotType": "mag",
                "t": gradTime,
                "data": grads[1],
                "annotations": [],
                "pen": "r",
            },
            {
                "label": "Gz",
                "type": "grads",
                "ind": str(2),
                "key": "Gz",
                "plotType": "mag",
                "t": gradTime,
                "data": grads[2],
                "annotations": [],
                "pen": "b",
            },
        ]
    )

    return channels
