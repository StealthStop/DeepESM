import numpy as np
import pandas as pd
import dask.array as da
import h5py

# Takes training vars, signal and background files and returns training data
def get_data(signalDataSet, backgroundDataSet, config, doBgWeight = False, doSgWeight = False):
    dgSig = DataGetter.DefinedVariables(config["allVars"], signal = True,  background = False)
    dgBg = DataGetter.DefinedVariables(config["allVars"],  signal = False, background = True)
    
    dataSig = dgSig.importData(samplesToRun = tuple(signalDataSet), maxNJetBin=config["maxNJetBin"])
    dataBg = dgBg.importData(samplesToRun = tuple(backgroundDataSet), maxNJetBin=config["maxNJetBin"])
    # Change the weight to 1 if needed
    if config["doSgWeight"]: dataSig["Weight"] = 35900*dataSig["Weight"]
    else: dataSig["Weight"] = np.full(dataSig["Weight"].shape, 1)
    if config["doBgWeight"]: dataBg["Weight"] = 35900*dataBg["Weight"]
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
    perms = np.random.permutation(trainData["data"].shape[0])
    for key in trainData:
        trainData[key] = trainData[key][perms]

    # Get the rescale inputs to have unit variance centered at 0 between -1 and 1
    def scale(data):
        # Get the masks for the different nJet bins (7 is hard coded njet start point...should fix this)
        for i in range(len(data["domain"][0])):
            mask = (1 - data["domain"][:,i]).astype(bool)
            data["mask_%02d" % (7+i)] = ~np.array(mask)
        if config["Mask"]:
            mask = data["mask_%02d" % (config["Mask_nJet"])]
            for key in data:
                data[key] = data[key][mask]
        data["mean"] = np.mean(data["data"], 0)
        data["std"] = np.std(data["data"], 0)
        data["scale"] = 1.0 / np.std(data["data"], 0)
    scale(trainData)
    scale(dataSig)
    scale(dataBg)
    return trainData, dataSig, dataBg    

class DataGetter:

    #The constructor simply takes in a list and saves it to self.list
    def __init__(self, variables, signal = False, background = False):
        self.list = variables
        self.signal = signal
        self.background = background

    #Simply accept a list and pass it to the constructor
    @classmethod
    def DefinedVariables(cls, variables, signal = False, background = False):
        return cls(variables, signal, background)

    def getList(self):
        return self.list

    def getColumnHeaders(self, sample, name, att):
        f = h5py.File(sample, "r")
        columnHeaders = f[name].attrs[att]
        f.close()
        return columnHeaders

    def getDataSets(self, samplesToRun, name):
        dsets = []
        for filename in samplesToRun:
            try:
                dsets.append( h5py.File(filename, mode='r')[name] )
            except:
                print "Warning: \"%s\" is empty" % filename
                continue
        return dsets
    
    def importData(self, samplesToRun, maxNJetBin = 11):
        #variables to train
        variables = self.getList()

        for fNum in range(len(samplesToRun)):
            try:
                columnHeaders = self.getColumnHeaders(samplesToRun[fNum], "EventShapeVar", "column_headers")
                break
            except:
                pass

        for v in variables:
            if not v in columnHeaders:
                print "Error: Variable not found: %s"%v
                exit()

        #load data files 
        dsets = self.getDataSets(samplesToRun, "EventShapeVar")
        arrays = [da.from_array(dset, chunks=(65536, 1024)) for dset in dsets]
        x = da.concatenate(arrays, axis=0)
         
        #setup and get data
        dataColumns = np.array([np.flatnonzero(columnHeaders == v)[0] for v in variables])
        data = x[:,dataColumns]
        npyInputData = data.compute()
        #print data.shape
        
        #setup and get labels
        npyInputAnswers = np.zeros((npyInputData.shape[0], 2))
        if self.signal:
            npyInputAnswers[:,0] = 1
        else:
            npyInputAnswers[:,1] = 1
        
        #setup and get domains
        domainColumnNames = ["NGoodJets_double"]
        #maxNJetBin = 11
        domainColumns = np.array([np.flatnonzero(columnHeaders == v)[0] for v in domainColumnNames])
        inputDomains = x[:,domainColumns]
        tempInputDomains = inputDomains.astype(int)
        tempInputDomains = da.reshape(tempInputDomains, [-1])
        tempInputDomains[tempInputDomains > maxNJetBin] = maxNJetBin 
        minNJetBin = tempInputDomains.min().compute()
        numDomains = maxNJetBin + 1 - minNJetBin
        tempInputDomains = tempInputDomains - tempInputDomains.min()
        d =  np.zeros((npyInputData.shape[0], numDomains))
        d[np.arange(d.shape[0]), tempInputDomains] = 1
            
        #setup and get weights
        wgtColumnNames = ["Weight"]
        wgtColumns = np.array([np.flatnonzero(columnHeaders == v)[0] for v in wgtColumnNames])
        npyInputSampleWgts = x[:,wgtColumns].compute()

        #NJet
        npyNJet = np.zeros((npyInputData.shape[0], 1))
        for i in range(0, len(d)):
            nJet = minNJetBin
            for j in range(len(d[i])):
                if d[i][j] == 1:
                    break
                else:
                    nJet +=1
            npyNJet[i][0] = int(nJet)
            
        return {"data":npyInputData, "labels":npyInputAnswers, "domain":d, "Weight":npyInputSampleWgts, "nJet":npyNJet}
