import os
import json
import shutil

#path = ""
#path = "TT_noSW_perNJet/"
path = "TT_TTJetsInclusvie_noSW_perNJet/"
#path = "TTJets_noSW_perNJet/"
with open(path + "megatrain.json", "r") as f:
    data = json.load(f)

#OverTrain Performance
wOT = 0.8
wP = 1 - wOT
for nJet in range(7, 11+1):
    key = min( [(wOT*data[x]["metric"]["OverTrain"] + wP*data[x]["metric"]["Performance"],   x) for x in data.keys() if data[x]["nJet"] == nJet] )[1]
    outputDir = "Training/nJet"+str(nJet)
    print "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    print "nJet:", nJet
    print key
    print data[key]["metric"]
    os.system("ls "+path+key)
    if os.path.exists(outputDir):
        print "Removing old training files: ", outputDir
        shutil.rmtree(outputDir)
    os.makedirs(outputDir) 
    os.system("cp "+path+key+"/keras_frozen.pb "+outputDir+"/keras_frozen_nJet"+str(nJet)+".pb")
    os.system("cp "+path+key+"/config.json "+outputDir+"/config_nJet"+str(nJet)+".json")
    print ""
