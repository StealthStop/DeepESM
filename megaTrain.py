from train import train
import json

CRED = "\033[91m"
CEND = "\033[0m"

totals = {}
result = {}
index = 0.0
for nPass in range(2):
     for epochs in range(10, 35, 1):
          for nNodes in range(60, 80, 1):
               for nHLayers in range(1, 3):
                    for drop_out in range(5, 9, 1):
                         #for nNodesD in range(2, 20, 2):
                         #for nHLayersD in range(1, 5):
                         if nPass == 0:
                              index += 1.0
                         else:
                              config = {"minNJetBin":7, "maxNJetBin":11, "gr_lambda":0, "nNodes":nNodes, "nNodesD":10,  "nHLayers":nHLayers, 
                                        "nHLayersD":1, "drop_out":float(drop_out)/10.0, "batch_size":2048, "epochs":epochs, "lr":0.001, "verbose":0, "Mask":False, "Mask_nJet":7}
                              name = ""
                              for key in sorted(config.keys()):
                                   name += key+"_"+str(config[key])+"_"
                         
                              _, metric = train(config)
               
                              total = 0.0
                              for key in metric:
                                   total += metric[key]
               
                              totals[name] = total
                              result[name] = {"total":total, "metric": metric}

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
