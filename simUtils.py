import xmltodict
import numpy as np

def getGradEvents(dict):
    
    time = np.zeros((len(dict["pulseprogram"]["ev"])))
    
    tUnit = float(dict["pulseprogram"]["@timeunit"])
    
    grads = np.zeros((3,len(dict["pulseprogram"]["ev"])))
    ind=0
    for event in dict["pulseprogram"]["ev"]:
        if "@g1" not in event:
                continue
        
        time[ind] = float(event["@t"])*tUnit
        grads[:,ind] = [float(event["@g1"]),float(event["@g2"]),float(event["@g3"])]
        ind=ind+1

    return time[:ind],grads[:,:ind]

def readGrads(path):
    
    with open(path+"/"+"_GCube.xml") as f:
        gCube = xmltodict.parse(f.read())

    time,grads =  getGradEvents(gCube)
    
    
    return time, grads    




def getRFEvents(dict):
    
    time = np.zeros((len(dict["pulseprogram"]["ev"])))
    timeRx = np.zeros((len(dict["pulseprogram"]["ev"])))
    
    tUnit = float(dict["pulseprogram"]["@timeunit"])
    
    rfs = np.zeros((2,len(dict["pulseprogram"]["ev"])))
    
    rxs = np.zeros((1,len(dict["pulseprogram"]["ev"])))
    
    indTx=0
    indRx=0
    
    indTxP=0
    
    timeTxP= np.zeros_like(time)
    TxPs = np.zeros((2,len(dict["pulseprogram"]["ev"])))
    
    info={}
    info["pulProg"] = dict["pulseprogram"]["@path"].split("/")[-1]
    
    for event in dict["pulseprogram"]["ev"]:
        if ("@g" not in event or "@p0" not in event) and "@nco" not in event and "@p1" not in event and "@pw" not in event:
                continue
        
        if "@nco" in event:
            if "@rgp" in event and "@t" in event:
                timeRx[indRx] = float(event["@t"])*tUnit
                rxs[0,indRx] = 1 if "1" in event["@rgp"] else 1
                
                indRx+=1
                continue
            else:
                if "@t" not in event:
                    continue
                timeRx[indRx] = float(event["@t"])*tUnit
                rxs[0,indRx] = 0
                indRx+=1
                continue    
            
            
        if "@pw" in event and "@p1" in event:
            timeTxP[indTxP] = float(event["@t"])*tUnit
            
            TxPs[:,indTxP] = [float(event["@p1"]),float(event["@pw"])]
            
            indTxP+=1
            continue
        
        if "@p1" in event:
            continue    
        
        if "@am" not in event:
            continue            
        
        time[indTx] = float(event["@t"])*tUnit
        rfs[:,indTx] = [float(event["@p0"]),float(event["@am"])]
        indTx=indTx+1

    return time[:indTx],rfs[:,:indTx], timeRx[:indRx],rxs[:,:indRx],timeTxP[:indTxP],TxPs[:,:indTxP],info

def readRFEvents(path):
    
    with open(path+"/"+"_FCube1.xml") as f:
        gCube = xmltodict.parse(f.read())

    time,TxEvents,rxTime,RxEvents,TxPTime,TxPs,info =  getRFEvents(gCube)
    
    
    return time, TxEvents,rxTime,RxEvents,TxPTime,TxPs,info