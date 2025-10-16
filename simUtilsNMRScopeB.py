import json
import re
import numpy as np

def readNMRScopeBChannels(path_data,progress,app):
    
    if type(path_data) is str:
        with open(path_data+"/pulse_seq.json") as f:
            data = json.load(f)
        
        app.setWindowTitle(f"{path_data}")
    else:
        data = path_data
               
    channels = []
    
    # Move gradients to the end...
    end_keys = ["gx", "gy", "gz"]

    # Rebuild the dictionary
    new_data = {k: v for k, v in data.items() if k not in end_keys}
    for k in end_keys:
        if k in data:
            new_data[k] = data[k]
            
    
    
    
    time = np.array(data["time"]["val"])*1e-3
    
    for channelName in new_data:
        if channelName == "time":
            continue

        if "phase" in channelName or "_p" in channelName:
            plotType = "phase"
        else:
            plotType = "mag"
        
        dataNpy = np.array(new_data[channelName]["val"])
        
        if dataNpy.size != time.size:
            print (f"Array length does not match the time vector for channel {channelName}. Found {dataNpy.size} samples, time vector has {time.size} samples skipping...")
            continue
        
        
        if new_data[channelName]["units"] != '':
            unitLabel = f"({new_data[channelName]["units"]})"
        else:
            unitLabel = "(-)"
                
        channelDes = {
            "chanLabel": channelName +" "+unitLabel,
            "label": channelName,
            "type": "grads" if re.match(r"g\w",channelName) else "NCO",
            "ind": str(0),
            "key": channelName,
            "plotType": plotType,
            "units": new_data[channelName]["units"],
            "t": time,
            "data":dataNpy,
            "show": new_data[channelName]["show"]=="yes"
        }
        channelDes["annotations"] = []            
        channels.append([channelDes])
    
    return channels