from train import train
import json

CRED = "\033[91m"
CEND = "\033[0m"

totals = {}
result = {}
index = 0
for nPass in range(2):
     for epochs in range(15, 40, 1):
          for nJet in list(x for x in range(7, 11+1)):
               #for nNodes in range(30, 100, 10):
               #for nNodesD in range(2, 20, 2):
               #for nHLayers in range(1, 3):
               #for nHLayersD in range(1, 5):
               #for drop_out in range(5, 9, 1):
               if nPass == 0:
                    index += 1.0
               else:
                    config = {"minNJetBin":7, "maxNJetBin":11, "gr_lambda":0, "nNodes":70, "nNodesD":10,  "nHLayers":1, 
                              "nHLayersD":1, "drop_out":0.7, "batch_size":2048, "epochs":epochs, "lr":0.001, "verbose":0, "Mask":True, "Mask_nJet":nJet}
                    name = ""
                    for key in sorted(config.keys()):
                         name += key+"_"+str(config[key])+"_"
                         
                    _, metric = train(config)
               
                    total = 0.0
                    for key in metric:
                         total += metric[key]
               
                    totals[name] = total
                    result[name] = {"total":total, "metric": metric, "nJet": nJet}

     print CRED + "Total number of trainings: " + str(index) + CEND
     print CRED + "Estimated time: " + str(index/60.0) + " hours or " + str(index/60.0/24.0) + " days" + CEND

bestKey = min(totals, key=totals.get)

print CRED + "-----------------------------------------------------------------------------------------------------------------" + CEND
print CRED + "Best Training" + CEND
print CRED + bestKey + CEND, result[bestKey]
print CRED + "Total number of trainings: " + str(index) + CEND
print CRED + "-----------------------------------------------------------------------------------------------------------------" + CEND

with open("megatrain.json",'w') as trainingOutput:
     json.dump(result, trainingOutput)
