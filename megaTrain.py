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
    command = "run \"python train.py --json temp.json --minMass 550 --valMass 550 --year 2016\""
    print( command )
    os.system(command)

    metric = None
    with open("temp.json", "r") as f:
         metric = json.load(f)

    total = 0.0
    for key in metric:
         if type(metric[key]) is str: continue
         total += metric[key]
    
    os.system("rm temp.json")
    return (name, total, {"total":total, "metric": metric})
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

#    for epoch in [60]:
#       for comb in [0.5,1.0,1.5]:
#           for disc in [0.5,1.0,1.5]:
#              for gr in [1.0, 1.2, 1.5]:
#                  for sgcorr in [0.0, 500.0, 750.0, 1000.0]:
#                      for bgcorr in [0.0, 500.0, 750.0, 1000.0]:
#                          for reg in [0.001,0.005,0.01]:
#                              index += 1.0
#                              config = {"atag" : "megaTrain2016_20201222", "disc_comb_lambda":float(comb), "disc_lambda":float(disc), "reg_lambda": float(reg), "gr_lambda": float(gr), "bg_cor_lambda": float(bgcorr), "sg_cor_lambda": float(sgcorr), "nNodes":100, "nNodesD":1, "nNodesM":100, "nHLayers":1, "nHLayersD":1, "nHLayersM":1, "drop_out":0.3, "batch_size":16384, "epochs": int(epoch), "lr":0.001}
#                              configList.append(config)

    for i in range(0,100):
        index += 1.0
        config = {"atag" : "Golden%d"%(i), "disc_comb_lambda": 1.0, "disc_lambda": 0.5, "reg_lambda": 0.01, "gr_lambda": 1.2, "bg_cor_lambda": 1000.0, "sg_cor_lambda": 0.0, "nNodes":100, "nNodesD":1, "nNodesM":100, "nHLayers":1, "nHLayersD":1, "nHLayersM":1, "drop_out":0.3, "batch_size":16384, "epochs": 60, "lr":0.001}
        configList.append(config)

    timePerTraining = 10.0 #min
    totalTime = timePerTraining*index #min
    print( red("Total number of trainings: " + str(index)) )
    print( red("Estimated time: " +str(totalTime)+ " minutes or " + str(totalTime/60.0) + " hours or " + str(totalTime/60.0/24.0) + " days") )
                      
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
