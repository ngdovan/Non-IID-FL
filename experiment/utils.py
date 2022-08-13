import json

def read_log(file_name):
    file = open(file_name, "r")
    data = []
    #order = ["time", "url", "type", "message"]
    order = ["time", "message"]


    for line in file.readlines():
        details = line.split("INFO")
        details = [x.strip() for x in details]
        structure = {key:value for key, value in zip(order, details)}
        if ">>" in structure["message"]:
            data.append(structure)
        
    #for entry in data:
    #    print(json.dumps(entry, indent = 4))

    global_train_acc = []
    global_test_acc = []

    for entry in data:
        print(entry["message"])
        if "Global Model Train accuracy" in entry["message"]:
            details = entry["message"].split(":")
            details = [x.strip() for x in details]
            global_train_acc.append(float(details[1]))
        elif "Global Model Test accuracy" in entry["message"]:
            details = entry["message"].split(":")
            details = [x.strip() for x in details]
            global_test_acc.append(float(details[1]))
    
    return global_train_acc, global_test_acc