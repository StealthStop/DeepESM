import numpy as np
import pandas as pd
import dask.array as da

class DataGetter:

    #The constructor simply takes in a list and saves it to self.list
    def __init__(self, variables, signal = True, background = True, domain = None, bufferData = False):
        self.list = variables
        self.signal = signal
        self.domain = domain
        self.background = background
        self.bufferData = bufferData
        self.dataMap = {}

    #Simply accept a list and pass it to the constructor
    @classmethod
    def DefinedVariables(cls, variables, signal = True, background = True, bufferData = False):
        return cls(variables, signal, background, bufferData)

    def getList(self):
        return self.list
    
    def prescaleBackground(self, input, answer, prescale):
      return np.vstack([input[answer == 1], input[answer != 1][::prescale]])
    
    def importData(self, samplesToRun, prescale = True, ptReweight=False, randomize = True):

        #check if this file was cached 
        if (samplesToRun, prescale, ptReweight) in self.dataMap:
            npyInputData, npyInputAnswers, npyInputWgts, npyInputSampleWgts = self.dataMap[samplesToRun, prescale, ptReweight]

        else:
            #variables to train
            vars = self.getList()
        
            inputData = np.empty([0])
            npyInputWgts = np.empty([0])
        
            import h5py

            variables = vars

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
            maxNJetBin = 11
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
            wgtColumnNames = ["sampleWgt"]
            wgtColumns = np.array([np.flatnonzero(columnHeaders == v)[0] for v in wgtColumnNames])
            npyInputSampleWgts = x[:,wgtColumns].compute()
            npyInputWgts = npyInputSampleWgts

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
            
            return {"data":npyInputData, "labels":npyInputAnswers, "domain":d, "weights":npyInputWgts, "w":npyInputSampleWgts, "nJet":npyNJet}
