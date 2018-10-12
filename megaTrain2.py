from train import train
from multiprocessing import Pool
import json

def red(string):
     CRED = "\033[91m"
     CEND = "\033[0m"
     return CRED + string + CEND

def parallel_train(config):
     name = ""
     for key in sorted(config.keys()):
          name += key+"_"+str(config[key])+"_"
          
     _, metric = train(config)
                    
     total = 0.0
     for key in metric:
          total += metric[key]

     return (name, total, {"total":total, "metric": metric})

if __name__ == '__main__':     
     configList = []
     index = 0
     for epochs in range(10, 35, 1):
          for nNodes in range(60, 80, 1):
               for nHLayers in range(1, 3):
                    for drop_out in range(5, 9, 1):
                         #for nNodesD in range(2, 20, 2):
                         #for nHLayersD in range(1, 5):
                              index += 1.0
                              config = {"minNJetBin":7, "maxNJetBin":11, "gr_lambda":0, "nNodes":nNodes, "nNodesD":10,  "nHLayers":nHLayers, 
                                        "nHLayersD":1, "drop_out":float(drop_out)/10.0, "batch_size":2048, "epochs":epochs, "lr":0.001, "verbose":0, "Mask":False, "Mask_nJet":7}
                              configList.append(config)
     print red("Total number of trainings: " + str(index))
     print red("Estimated time: " + str(index/60.0) + " hours or " + str(index/60.0/24.0) + " days")
                       
     totals = {}
     result = {}
     pool = Pool(processes=2)
     outPut = pool.map(parallel_train, configList)          

     for t in outPut:
          totals[t[0]] = t[1]
          result[t[0]] = t[2]
     bestKey = min(totals, key=totals.get)
     
     print red("-----------------------------------------------------------------------------------------------------------------")
     print red("Best Training")
     print red(bestKey), result[bestKey]
     print red("Total number of trainings: " + str(index))
     print red("-----------------------------------------------------------------------------------------------------------------")
     
     with open("megatrain.json",'w') as trainingOutput:
          json.dump(result, trainingOutput)
