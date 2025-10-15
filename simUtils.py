import xmltodict
import numpy as np


def getGradEvents(dict):
    time = np.zeros((len(dict["pulseprogram"]["ev"])))

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


def readGrads(path):
    with open(path + "/" + "_GCube.xml") as f:
        gCube = xmltodict.parse(f.read())

    time, grads = getGradEvents(gCube)

    return time, grads


def initNco(nEvents):
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


def turnOffNco(ncos,ncoNumber,t,ind):
    
    
    ncos[ncoNumber]["t"][ind] = t
    ncos[ncoNumber]["p0"][ind] = float(ncos[ncoNumber]["p0"][ind-1])if ind>0 else 0
    ncos[ncoNumber]["p1"][ind] = float(ncos[ncoNumber]["p1"][ind-1])if ind>0 else 0
    ncos[ncoNumber]["p2"][ind] = float(ncos[ncoNumber]["p2"][ind-1])if ind>0 else 0
    ncos[ncoNumber]["am"][ind] = 0
    ncos[ncoNumber]["pw"][ind] = 0
    ncos[ncoNumber]["sf"][ind] = float(ncos[ncoNumber]["sf"][ind-1])if ind>0 else 0
    ncos[ncoNumber]["rgp"][ind] = 0
    
    
        
        
        
        
    

def getRFEvents(dict):
    ncos = {}

    info = {}
    info["pulProg"] = dict["pulseprogram"]["@path"].split("/")[-1]
    ind = 0
    tUnit = float(dict["pulseprogram"]["@timeunit"])
    
    indMaxes={}
    
    ncoNumber= None
    
    for event in dict["pulseprogram"]["ev"]:
        if "@t" not in event:
            print("Skip event")
            continue




        if "@nco" in event:
            if event["@nco"] == "0":
                turnOffNco(ncos,ncoNumber,float(event["@t"])*tUnit,indMaxes[str(ncoNumber)])
                indMaxes[ncoNumber]+=1
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
            ncos[ncoNumber]["p0"][ind] = (
                ncos[ncoNumber]["p0"][ind - 1] if ind > 0 else 0
            )

        if "@p1" in event:
            ncos[ncoNumber]["p1"][ind] = float(event["@p1"]) 
        else:
            ncos[ncoNumber]["p1"][ind] = (
                ncos[ncoNumber]["p1"][ind - 1] if ind > 0 else 0
            )

        if "@p2" in event:
            ncos[ncoNumber]["p2"][ind] = float(event["@p2"])
        else:
            ncos[ncoNumber]["p2"][ind] = (
                ncos[ncoNumber]["p2"][ind - 1] if ind > 0 else 0
            )

        if "@pw" in event:
            ncos[ncoNumber]["pw"][ind] = float(event["@pw"]) 
        else:
            ncos[ncoNumber]["pw"][ind] = (
                ncos[ncoNumber]["pw"][ind - 1] if ind > 0 else 0
            )

        if "@am" in event:
            ncos[ncoNumber]["am"][ind] = float(event["@am"]) 
        else:
            ncos[ncoNumber]["am"][ind] = (
                ncos[ncoNumber]["am"][ind - 1] if ind > 0 else 0
            )

        if "@sf" in event:
            ncos[ncoNumber]["sf"][ind] = float(event["@sf"]) 
        else:
            ncos[ncoNumber]["sf"][ind] = (
                ncos[ncoNumber]["sf"][ind - 1] if ind > 0 else 0
            )

        if "@rgp" in event:
            ncos[ncoNumber]["rgp"][ind] = 1 if '0--0' in event["@rgp"] else 0
        
        
        elif  "@ln" in event:
            if event["@ln"]=="8000001":
                ncos[ncoNumber]["rgp"][ind] = 0
            else:
                ncos[ncoNumber]["rgp"][ind] = (
                    ncos[ncoNumber]["rgp"][ind - 1] if ind > 0 else 0
                )                    
        else:    
            ncos[ncoNumber]["rgp"][ind] = (
                ncos[ncoNumber]["rgp"][ind - 1] if ind > 0 else 0
            )

        indMaxes[str(ncoNumber)]+=1
         
         
    for ncoKey in ncos:
        for key in ncos[ncoKey]:
            if isinstance(ncos[ncoKey][key], np.ndarray):
                
                ncos[ncoKey][key] = ncos[ncoKey][key][:indMaxes[ncoKey]]

    return ncos, info


def readRFEvents(path):
    with open(path + "/" + "_FCube1.xml") as f:
        gCube = xmltodict.parse(f.read())

    ncos, info = getRFEvents(gCube)

    return ncos, info
