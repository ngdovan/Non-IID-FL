import json

file_name = "/home/ngdovan/codesample/NonIID-Bench/Non-IID-FL/logs/experiment_log-2022-08-11-22:36-43.log"
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
        
#print(acc for acc in global_train_acc)
for acc in global_train_acc:
    print(acc)
    
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()