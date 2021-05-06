#from train import Train
import os
import json
import subprocess
from multiprocessing import Pool


def red(string):
    CRED = "\033[91m"
    CEND = "\033[0m"
    return CRED + string + CEND

def parallel_train(config,command):
    name = ""
    for key in sorted(config.keys()):
         name += key+"_"+str(config[key])+"_"

    with open("temp.json",'w') as f:
         json.dump(config, f)

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


if __name__ == '__main__':     
    configList = []
    index = 0
    totals = {}
    result = {}

    # ---------------------------------------
    # Make a list for command line options
    # ---------------------------------------
    commandList = []
    commands = {
        "Rpv" : "run \"python train.py --json temp.json --minMass 300 --valMass 550 --year 2016 --tree myMiniTree_0l --seed 527725 --model RPV --valModel RPV\"",
        #"RpvSyy" : "run \"python train.py --json temp.json --minMass 300 --valMass 550 --year 2016 --tree myMiniTree_0l --seed 527725 --model RPV_SYY --valModel RPV\"",
        #"RpvSyySHH" : "run \"python train.py --json temp.json --minMass 300 --valMass 550 --year 2016 --tree myMiniTree_0l --seed 527725 --model RPV_SYY_SHH --valModel RPV\"",
    }
    
    # -----------
    # 0-Lepton
    # -----------
    for model,command in commands.items():
        index += 1.0
        hyperconfig = {"atag" : "%s550"%model, "disc_comb_lambda": 0.0, "gr_lambda": 1.0, "disc_lambda": 10.0, "bg_cor_lambda": 1000.0, "sg_cor_lambda" : 1000.0, "reg_lambda": 0.001, "nNodes":100, "nNodesD":1, "nNodesM":100, "nHLayers":1, "nHLayersD":1, "nHLayersM":1, "drop_out":0.3, "batch_size":10000, "epochs":15, "lr":0.001}
        configList.append(hyperconfig)
        commandList.append(command)

    # -----------
    # 1-Lepton
    # -----------
    #for i in range(0,100):
    #    index += 1.0
    #    config = {"atag" : "Golden%d"%(i), "disc_comb_lambda": 1.0, "disc_lambda": 0.5, "reg_lambda": 0.01, "gr_lambda": 1.2, "bg_cor_lambda": 1000.0, "sg_cor_lambda": 0.0, "nNodes":100, "nNodesD":1, "nNodesM":100, "nHLayers":1, "nHLayersD":1, "nHLayersM":1, "drop_out":0.3, "batch_size":16384, "epochs": 60, "lr":0.001}
    #    configList.append(config)

    timePerTraining = 10.0 #min
    totalTime = timePerTraining*index #min
    print( red("Total number of trainings: " + str(index)) )
    print( red("Estimated time: " +str(totalTime)+ " minutes or " + str(totalTime/60.0) + " hours or " + str(totalTime/60.0/24.0) + " days") )

    # ----------------------                
    # Training one by one
    # ---------------------- 
    for config,command in zip(configList,commandList):
         outPut = parallel_train(config,command)
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
