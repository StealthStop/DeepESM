#from train import Train
import subprocess
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

     #############################################
     with open("temp.json",'w') as f:
          json.dump(config, f)
     command = "run \"./train.py temp.json\""
     print( command )
     os.system(command)
     os.system("rm temp.json")
     return (name, 0.0, {"total":0.0, "metric": 0.0})
     #############################################

     ###############################################
     #print(config)
     #t = Train()
     #metric = t.train(config)
     #del t
     #
     #total = 0.0
     #for key in metric:
     #     total += metric[key]
     #
     #return (name, total, {"total":total, "metric": metric})
     ###############################################

if __name__ == '__main__':     
     configList = []
     index = 0
     totals = {}
     result = {}
     for cor_lambda in [0.0, 10.0, 50.0, 200.0, 500.0, 1000.0, 2000.0]:
          index += 1.0
          config = {"gr_lambda": 0.0, "cor_lambda": float(cor_lambda), "nNodes":100, "nNodesD":1, "nNodesM":100,
                    "nHLayers":1, "nHLayersD":1, "nHLayersM":1, "drop_out":0.3,
                    "batch_size":16384, "epochs":60, "lr":0.001}
          configList.append(config)

     for epoch in [40, 50, 65, 70, 80]:
          index += 1.0
          config = {"gr_lambda": 0.0, "cor_lambda": 100.0, "nNodes":100, "nNodesD":1, "nNodesM":100,
                    "nHLayers":1, "nHLayersD":1, "nHLayersM":1, "drop_out":0.3,
                    "batch_size":16384, "epochs":epoch, "lr":0.001}
          configList.append(config)

     #for epochs in range(110, 130+5, 5):
     #     for nNodes in range(60, 80+5, 5):
     #          for nNodesD in range(5, 15+5, 5):
     #               for nHLayers in range(1, 2+1, 1):
     #                    for nHLayersD in range(1, 2+1, 1):                    
     #                         for drop_out in range(6, 8+1, 1):
     #                              for Lambda in range(0, 3+1, 1):
     #                                   index += 1.0
     #                                   config = {"minNJetBin":7, "maxNJetBin":11, "lambda":Lambda, "nNodes":nNodes, "nNodesD":nNodesD,  "nHLayers":nHLayers, 
     #                                             "nHLayersD":nHLayersD, "drop_out":float(drop_out)/10.0, "batch_size":2048, "epochs":epochs, "lr":0.001, "verbose":0, "Mask":False, "Mask_nJet":7}
     #                                   configList.append(config)

     timePerTraining = 10.0 #min
     totalTime = timePerTraining*index #min
     print( red("Total number of trainings: " + str(index)) )
     print( red("Estimated time: " +str(totalTime)+ " minuets or " + str(totalTime/60.0) + " hours or " + str(totalTime/60.0/24.0) + " days") )
                       
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
     
     print( red("-----------------------------------------------------------------------------------------------------------------") )
     print( red("Best Training") )
     print( red(bestKey), result[bestKey] )
     print( red("Total number of trainings: " + str(index)) )
     print( red("-----------------------------------------------------------------------------------------------------------------") )
     
     #with open("Megatrain.json",'w') as trainingOutput:
     #     json.dump(result, trainingOutput)

