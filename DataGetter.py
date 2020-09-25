import uproot
import numpy as np
import pandas as pd
from glob import glob
import time 

def getSamplesToRun(names):
    s = glob(names)
    if len(s) == 0:
        raise Exception("No files find that correspond to: "+names)
    return s

# Takes training vars, signal and background files and returns training data
def get_data(signalDataSet, backgroundDataSet, config, doBgWeight = False, doSgWeight = False):
    dgSig = DataGetter.DefinedVariables(config["allVars"], signal = True,  background = False)
    dgBg = DataGetter.DefinedVariables(config["allVars"],  signal = False, background = True)

    dataSig = dgSig.importData(samplesToRun = tuple(signalDataSet), treename = "myMiniTree", maxNJetBin=config["maxNJetBin"])
    dataBg = dgBg.importData(samplesToRun = tuple(backgroundDataSet), treename = "myMiniTree", maxNJetBin=config["maxNJetBin"])

    # Change the weight to 1 if needed
    if config["doSgWeight"]: dataSig["Weight"] = config["lumi"]*dataSig["Weight"]
    else: dataSig["Weight"] = np.full(dataSig["Weight"].shape, 1)
    if config["doBgWeight"]: dataBg["Weight"] = config["lumi"]*dataBg["Weight"]
    else: dataBg["Weight"] = np.full(dataBg["Weight"].shape, 1)

    minLen = min(len(dataSig["data"]),len(dataBg["data"]))

    # Put signal and background data together in trainData dictionary
    trainDataArray = [dataSig,dataBg]
    trainData = {}
    for data in trainDataArray:
        for key in data:
            if key in trainData:
                trainData[key] = np.vstack([trainData[key], data[key][:minLen]])
            else:
                trainData[key] = data[key][:minLen]

    # Randomly shuffle the signal and background 
    np.random.seed(int(time.time())) 
    perms = np.random.permutation(trainData["data"].shape[0])
    for key in trainData:
        trainData[key] = trainData[key][perms]

    # Get the rescale inputs to have unit variance centered at 0 between -1 and 1
    def scale(data):
        # Get the masks for the different stop masses
        for m in range(config["minStopMass"],config["maxStopMass"]+50,50):
            mask = ~np.ma.masked_where(data["masses"] != m, data["masses"]).mask
            data["mask_m"+str(m)] = mask[:,0]        
        # Get the masks for the different nJet bins
        for i in range(len(data["domain"][0])):
            mask = (1 - data["domain"][:,i]).astype(bool)
            data["mask_nJet_%02d" % (config["minNJetBin"]+i)] = ~np.array(mask)
            mask = (data["domain"][:,i]).astype(bool)
            data["mask_stuff_%02d" % (config["minNJetBin"]+i)] = ~np.array(mask)            
        if config["Mask"]:
            mask = data["mask_stuff_%02d" % (config["Mask_nJet"])]
            for key in data:
                data[key] = data[key][mask]
        data["mean"] = np.mean(data["data"], 0)
        data["std"] = np.std(data["data"], 0)
        data["scale"] = 1.0 / data["std"]
    scale(trainData)
    scale(dataSig)
    scale(dataBg)
    return trainData, dataSig, dataBg

class DataGetter:
    #The constructor simply takes in a list and saves it to self.l
    def __init__(self, variables, signal = False, background = False):
        self.l = variables
        self.signal = signal
        self.background = background
        self.columnHeaders = None
        self.data = None

    #Simply accept a list and pass it to the constructor
    @classmethod
    def DefinedVariables(cls, variables, signal = False, background = False):
        return cls(variables, signal, background)

    def getList(self):
        return self.l

    def getData(self):
        return self.data

    def getColumnHeaders(self, samplesToRun, treename):
        if self.columnHeaders is None:
            try:
                sample = samplesToRun[0]                
                f = uproot.open(sample)
                self.columnHeaders = f[treename].pandas.df().columns.tolist()
                f.close()
            except IndexError as e:
                print(e)
                raise IndexError("No sample in samplesToRun")
        return self.columnHeaders

    def checkVariables(self, variables):
        for v in variables:            
            if not v in self.columnHeaders:
                raise ValueError("Variable not found in input root file: %s"%v)
        
    def getDataSets(self, samplesToRun, treename):
        dsets = []
        if len(samplesToRun) == 0:
            raise IndexError("No sample in samplesToRun")
        for filename in samplesToRun:
            try:
                f = uproot.open(filename)
                #dsets.append( f[treename].pandas.df(branches=variables) )
                dsets.append( f[treename].pandas.df() )
                f.close()
            except Exception as e:
                print("Warning: \"%s\" has issues" % filename, e)
                continue
        return dsets
    
    def importData(self, samplesToRun, treename = "myMiniTree", maxNJetBin = 11):
        #variables to train
        variables = self.getList()
        self.getColumnHeaders(samplesToRun, treename)
        self.checkVariables(variables)
        
        #load data files and get data
        dsets = self.getDataSets(samplesToRun, treename)
        self.data = pd.concat(dsets)
        self.data = self.data.dropna()

        #setup and get training data
        npyInputData = self.data[variables].astype(float).values

        #setup and get labels
        npyInputAnswers = np.zeros((npyInputData.shape[0], 2))
        if self.signal:
            npyInputAnswers[:,0] = 1
        else:
            npyInputAnswers[:,1] = 1
        #unique, counts = np.unique(npyInputAnswers, return_counts=True)
        #print(dict(zip(unique, counts)))

        #setup and get domains
        domainColumnNames = ["NGoodJets_pt30"]
        inputDomains = self.data[domainColumnNames]
        tempInputDomains = inputDomains.astype(int)
        tempInputDomains[tempInputDomains > maxNJetBin] = maxNJetBin 
        minNJetBin = tempInputDomains.min().values[0]
        numDomains = maxNJetBin + 1 - minNJetBin
        npyNJet = tempInputDomains.astype(float).values
        tempInputDomains = tempInputDomains - tempInputDomains.min()
        npyInputDomain = np.zeros((npyInputData.shape[0], numDomains))
        npyInputDomain[np.arange(npyInputDomain.shape[0]), tempInputDomains.values.flatten()] = 1

        #setup and get weights
        wgtColumnNames = ["totalEventWeight"]
        npyInputSampleWgts = self.data[wgtColumnNames].values

        #setup and get masses
        massNames = ["mass"]
        npyMasses = self.data[massNames].values
        npyMasses[npyMasses == 173.0] = 0.0

        #sample weight for signal masses
        unique, counts = np.unique(npyMasses, return_counts=True)
        #print(dict(zip(unique, counts)))
        d = dict(zip(unique, counts))
        m = max(value for key, value in d.items())
        d = {key: round(float(m)/float(value),3) for key, value in d.items()}

        factor = 1.0
        for key in d.keys():
            if key in [300.0, 350.0, 400.0]:
                d[key] = 1.0
            elif key in [0.0, 1.0, 173.0]:
                d[key] = 1.25
            else:
                d[key] = round(factor*d[key], 3)

        npySW = np.copy(npyMasses)
        for key, value in d.items(): npySW[npySW == key] = factor*value

        return {"data":npyInputData, "labels":npyInputAnswers, "domain":npyInputDomain, "Weight":npyInputSampleWgts, "nJet":npyNJet, "masses":npyMasses, "sample_weight":npySW}

