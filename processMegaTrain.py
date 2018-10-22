import os
import json
import shutil

#path = ""
path = "TT+TTJets_GoodJetsandGoodLepton/"

with open(path + "megatrain.json", "r") as f:
    data = json.load(f)

#Performance
wOT = 1
wP = 1 - wOT
key = min( [(wOT*data[x]["metric"]["OverTrain"] + wP*data[x]["metric"]["Performance"],   x) for x in data.keys()] )[1]

outputDir = "Training/"
print key
print data[key]["metric"]
if os.path.exists(outputDir):
    print "Removing old training files: ", outputDir
    shutil.rmtree(outputDir)
os.makedirs(outputDir) 
os.system("cp "+path+key+"/keras_frozen.pb "+outputDir+"/keras_frozen.pb")
os.system("cp "+path+key+"/config.json "+outputDir+"/config.json")
