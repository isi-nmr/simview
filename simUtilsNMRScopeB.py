import json
import re
import numpy as np

def readNMRScopeBChannels(path,progress,app):
    
    with open(path+"/pulse_seq.json") as f:
        data = json.load(f)
        
    channels = []
    
    # Move gradients to the end...
    end_keys = ["gx", "gy", "gz"]

    # Rebuild the dictionary
    new_data = {k: v for k, v in data.items() if k not in end_keys}
    for k in end_keys:
        if k in data:
            new_data[k] = data[k]
            
    app.setWindowTitle(f"{path}")
    
    
    time = np.array(data["time"])*1e-6
    
    for channelName in new_data:
        if channelName == "time":
            continue

        if "phase" in channelName or "_p" in channelName:
            plotType = "phase"
        else:
            plotType = "mag"
        
        dataNpy = np.array(new_data[channelName])
        
        if dataNpy.size != time.size:
            print (f"Array length does not match the time vector for channel {channelName}. Found {dataNpy.size} samples, time vector has {time.size} samples skipping...")
            continue
        
                
        channelDes = {
            "chanLabel": channelName,
            "label": channelName,
            "type": "grads" if re.match(r"g\w",channelName) else "NCO",
            "ind": str(0),
            "key": channelName,
            "plotType": plotType,
            "t": time,
            "data":dataNpy,
        }
        channelDes["annotations"] = []            
        channels.append([channelDes])
    
    return channels