import numpy as np
import pandas as pd
import dask.array as da
import h5py

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
    
    def importData(self, samplesToRun, maxNJetBin = 11):
        #variables to train
        variables = self.getList()

        f = h5py.File(samplesToRun[0], "r")
        columnHeaders = f["EventShapeVar"].attrs["column_headers"]
        f.close()

        for v in variables:
            if not v in columnHeaders:
                print "Variable not found: %s"%v

        #load data files 
        dsets = [h5py.File(filename, mode='r')['EventShapeVar'] for filename in samplesToRun]
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
