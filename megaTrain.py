from train import Train
from multiprocessing import Pool
import os
import json

def red(string):
     CRED = "\033[91m"
     CEND = "\033[0m"
     return CRED + string + CEND

def parallel_train(config):
     name = ""
     for key in sorted(config.keys()):
          name += key+"_"+str(config[key])+"_"
          
     command = "python train.py '"+str(config)+"'"
     print command
     os.system(command)
     
     #total = 0.0
     #for key in metric:
     #     total += metric[key]
     #
     #return (name, total, {"total":total, "metric": metric})
     return (name, 0.0, {"total":0.0, "metric": 0.0})

if __name__ == '__main__':     
     configList = []
     index = 0
     totals = {}
     result = {}
     for epochs in range(110, 130+5, 5):
          for nNodes in range(60, 80+5, 5):
               for nNodesD in range(5, 15+5, 5):
                    for nHLayers in range(1, 2+1, 1):
                         for nHLayersD in range(1, 2+1, 1):                    
                              for drop_out in range(6, 8+1, 1):
                                   for Lambda in range(0, 3+1, 1):
                                        index += 1.0
                                        config = {"minNJetBin":7, "maxNJetBin":11, "gr_lambda":Lambda, "nNodes":nNodes, "nNodesD":nNodesD,  "nHLayers":nHLayers, 
                                                  "nHLayersD":nHLayersD, "drop_out":float(drop_out)/10.0, "batch_size":2048, "epochs":epochs, "lr":0.001, "verbose":0, "Mask":False, "Mask_nJet":7}
                                        configList.append(config)
     print red("Total number of trainings: " + str(index))
     print red("Estimated time: " + str(index/60.0) + " hours or " + str(index/60.0/24.0) + " days")
                       
     #Parallel training
     #pool = Pool(processes=1)
     #outPut = pool.map(parallel_train, configList)          
     #for t in outPut:
     #     totals[t[0]] = t[1]
     #     result[t[0]] = t[2]

     #Training one by one
     for config in configList:
          outPut = parallel_train(config)
          totals[outPut[0]] = outPut[1]
          result[outPut[0]] = outPut[2]

     bestKey = min(totals, key=totals.get)
     
     print red("-----------------------------------------------------------------------------------------------------------------")
     print red("Best Training")
     print red(bestKey), result[bestKey]
     print red("Total number of trainings: " + str(index))
     print red("-----------------------------------------------------------------------------------------------------------------")
     
     with open("megatrain.json",'w') as trainingOutput:
          json.dump(result, trainingOutput)
