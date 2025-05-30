import gc
import sys
import math
import numpy as np
import tensorflow as tf
import uproot4 as uproot
import tensorflow.keras as K

import datetime
import time

def timeStamp():
    return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

class DataLoader(K.utils.Sequence):

    def __init__(self, config, signalDataSet, backgroundDataSet, categorization=0):

        np.random.seed(config["seed"]) 

        self.config = config
        self.datasets = [tuple(backgroundDataSet), tuple(signalDataSet)]

        # We use "variables" to store all needed branches from the inputs
        # auxVars hold variables used as labels in the network
        self.variables = self.config["trainVars"] + self.config["auxVars"]

        # These sample factor variables store how many events
        # per category should be drawn per batch
        self.sigSampleFactor = 0
        self.bkgSampleFactor = 0

        # Will hold per event information for inputs, Njets, regression label, model, etc.
        self.data = {}

        self.df = {}

        self.columnHeaders = None
        self.numBkgEvents = 0.0
        self.numSigEvents = 0.0

        # Stores the mean and inverse std dev for each input variable
        # For use by the lambda layer when constructing the model
        self.means = None
        self.scales = None

        # Keeps track of the number of unique categories for 
        # background and signal. Can depend on mass bins, njets bins
        # Used for figuring out how many events from each category
        # to be used per batch
        self.numBkgCategories = 1
        self.numSigCategories = 1

        self.batchIndexContainer = None 

        # This call performs the entire setup of the DataLoader instance
        self.importData()

    # Flatten the dictionary of numpy arrays over the mass and njet categories
    # for use in the Validation part of the framework
    # If a process, mass, and/or Njets category is specified, that combination
    # returned while flattening over the other categories
    def getFlatData(self, year = None, process = None, mass = None, Njets = None):
        
        self.data = None
        gc.collect()
        flatDictionary = {}

        mask = (np.ones(self.df[list(self.df.keys())[0]].shape[0]))==1
        if process != None:
            mask &= (self.df[self.config["modelLabel"]]==process)

        if mass != None:
            mask &= (self.df[self.config["massLabel"]]==mass)

        if Njets != None:
            mask &= (self.df[self.config["domainLabel"]]==Njets)

        if year != None and year != "Run2" and year != "2016All":
            mask &= (self.df["year"]==year)

        flatDictionary["vars"]    = self.config["trainVars"]
        flatDictionary["njets"]   = self.df[self.config["domainLabel"]][mask]
        flatDictionary["inputs"]  = self.df["inputs"][mask]
        flatDictionary["massReg"] = self.df[self.config["regressionLabel"]][mask]
        flatDictionary["mass"]    = self.df[self.config["massLabel"]][mask]
        flatDictionary["weight"]  = self.df[self.config["weightLabel"]][mask]*self.config["lumi"]
        flatDictionary["model"]   = self.df[self.config["modelLabel"]][mask]
        flatDictionary["label"]   = (self.df[self.config["modelLabel"]][mask]>=100).astype("float16")

        return flatDictionary

    def getSigWeight(self):
        return float(self.sigWeight)

    def getBkgWeight(self):
        return float(self.bkgWeight)

    def getNumBkgEvents(self):
        return float(self.numBkgEvents)

    def getNumSigEvents(self):
        return float(self.numSigEvents)
    
    def getDataMeans(self):
        return self.means

    def getDataScales(self):
        return self.scales

    # Returns the shapes of the unique layers output in the NN model
    # Order: Mass regression, Domain (Njets), Double DisCo, Input
    def getShapes(self):
        return 1, self.config["maxNJetBin"] - self.config["minNJetBin"] + 1, 1, len(self.config["trainVars"])
        #return 0, self.config["maxNJetBin"] - self.config["minNJetBin"] + 1, 4, len(self.config["trainVars"])
   
    def getColumnHeaders(self):
        if self.columnHeaders is None:
            try:
                print(self.datasets[0])
                sample = self.datasets[0][0]                
                f = uproot.open(sample)
                theVars = [v for v in self.variables]
                self.columnHeaders = f[self.config["tree"]].arrays(expressions=theVars, library="np")
                f.close()
            except IndexError as e:
                raise IndexError("Problem getting variable names:", e)

        return self.columnHeaders
    
    # Process is 0 for background and 1 for signal
    def getDataSets(self, process = -1):
        if process != 0 and process != 1:
            raise IndexError("Must specify to load signal or background")

        dsets = []
        years = []
        if len(self.datasets[process]) == 0:
            raise IndexError("No sample in samplesToRun")

        max_entries_bg = None
        max_entries_sg = None
        if self.config["debug"]:
            max_entries_bg = 500
            max_entries_sg = 100

        variations = ["", "JECup", "JECdown", "JERup", "JERdown"]
        if self.config["debug"] or not self.config["useJECs"]:
            variations = [""]

        for suffix in variations:
            theVars = [v+suffix if v not in ["mass", "model", "Weight"] else v for v in self.variables]

            selection = None
            if "_1l" in self.config["tree"] or "_0l" in self.config["tree"]:
                print("%s [INFO]: Selecting 1L events with >= %d jets"%(timeStamp(), self.config["nJets"]))
                selection = "(%s>=%s)"%(self.config["domainLabel"]+suffix,self.config["nJets"])

            for filename in self.datasets[process]:
            
                year = filename.split("MyAnalysis_")[-1].split("_")[0]
                try:
                    f = uproot.open(filename)
                    if "RPV" in filename or "SYY" in filename:
                        tempnpf = f[self.config["tree"]+suffix].arrays(expressions=theVars, cut=selection, library="np", entry_stop=max_entries_sg)
                    else:
                        tempnpf = f[self.config["tree"]+suffix].arrays(expressions=theVars, cut=selection, library="np", entry_stop=max_entries_bg)
                        
                    f.close()

                    tempVars = list(tempnpf.keys())
                    npf = {}
                    for var in tempVars:
                        newVar = var.replace("JERup", "").replace("JECup", "").replace("JERdown", "").replace("JECdown", "")

                        if ("Jet_m_" in newVar or "Stop1_m_" in newVar or "Stop2_m_" in newVar) and self.config["scaleLog"]:
                            npf[var] = np.log(1 + tempnpf[var]).astype('float16')

                        elif ("Jet_pt_" in newVar or "Stop1_pt_" in newVar or "Stop2_pt" in newVar) and self.config["scaleJetPt"]:
                            npf[newVar] = tempnpf[var] /tempnpf["HT_trigger_pt30"+suffix].astype('float16')
                        else:
                            npf[newVar] = tempnpf[var].astype('float16')

                    print("%s [INFO]: Loaded \"%s\" from input file \"%s\""%(timeStamp(), self.config["tree"]+suffix,filename))

                    dsets.append(npf)
                    years.append(year)
                except Exception as e:
                    print("%s [WARNING]: Skipping tree \"%s\" in file \"%s\" !" %(timeStamp(), self.config["tree"]+suffix, filename) , e)
                    continue

        return dsets, years

    # This special function is called per-batch and constructs the batch
    # Based on how many background and signal events should appear as well as the 
    # different categories specified.
    def __getitem__(self, index):

        batchInputs   = None 
        batchMassReg  = None 

        offset = 0
        for mnjet, maskDict in self.data.items():
            
            mixer = np.random.default_rng().choice(maskDict["mask"].shape[0], size=maskDict["factor"], replace=False)
            self.batchIndexContainer[offset:offset+maskDict["factor"]] = maskDict["mask"][mixer]
            offset += maskDict["factor"]

        batchInputs  = self.df["inputs"][self.batchIndexContainer]
        batchMass    = self.df[self.config['massLabel']][self.batchIndexContainer]
        if self.config["scaleLog"]:
            batchMassReg = np.log(1 + self.df[self.config["regressionLabel"]][self.batchIndexContainer])
        else:
            batchMassReg = self.df[self.config["regressionLabel"]][self.batchIndexContainer]/self.config["massScale"]

        model      = self.df[self.config["modelLabel"]][self.batchIndexContainer]
        mass       = self.df[self.config["massLabel"]][self.batchIndexContainer]
        labelSig     = (model>=100).astype("float16")
        #labelSigLow     = (np.logical_and(model>=100, mass <= 400)).astype("float32")
        #labelSigMid    = (np.logical_and(np.logical_and(model>=100, mass > 400), mass <= 850)).astype("float32")
        #labelSigHigh    = (np.logical_and(model>=100, mass > 850)).astype("float32")
        #labelBkg   = (model<100).astype("float32")
        #batchDisCo = np.vstack((labelSigHigh, labelSigMid, labelSigLow, labelBkg, labelSigHigh, labelSigMid, labelSigLow, labelBkg)).T
        batchDisCo = np.vstack((labelSig, labelSig)).T
    
        #return batchInputs, tuple((batchDisCo, batchDisCo, batchMassReg))
        return batchInputs, tuple((batchDisCo, batchDisCo, batchDisCo, batchMassReg))
        #return batchInputs, tuple((batchDisCo, batchDisCo, batchDisCo, batchDisCo, batchMassReg))

    # Required function that tells tensorflow how many batches per epoch
    def __len__(self):
        return math.ceil((self.numBkgEvents+self.numSigEvents) / self.config["batch"])
    
    def importData(self):
    
        # Load events from the inputs ROOT files
        # and store them in numpy dataframes
        self.getColumnHeaders()

        # Hold numpy array of vars loaded from ROOT files
        # for each process/model/variation
        dsets = []
        years = []
        temp1, temp2 = self.getDataSets(0)
        dsets += temp1; years += temp2
        temp1, temp2 = self.getDataSets(1)
        dsets += temp1; years += temp2
       
        # Mix the events around when formatting inputs
        # to be read in while training
        mixer = list(range(0, len(dsets)))
        np.random.shuffle(mixer)

        # Set up a list to hold the mean and std dev for each input variable
        self.means  = [None for var in self.config["trainVars"]]
        self.scales = [None for var in self.config["trainVars"]]

        # Here determine the total number of events from all data sets
        # Then pre allocate numpy arrays to hold the variables and avoid
        # excessive copying
        totalNevts = 0
        for dset in dsets:
            totalNevts += len(dset[self.config["trainVars"][0]])

        nVars = len(self.config["trainVars"])

        # Allocate array for inputs and labels
        self.df["inputs"] = np.zeros((totalNevts, nVars), dtype="float16")
        for var in self.config["auxVars"]:
            self.df[var] = np.zeros(totalNevts, dtype="float16")

        self.df["year"] = np.empty(totalNevts, dtype="U11")

        # Use an offset to move the "pointer" were we write in values
        # to the preallocated arrays
        offset = 0

        # Calculated the mean and inverse std dev for the input variables
        for i,Var in enumerate(self.config["trainVars"]):
            var = [x[Var] for x in dsets]
            varVals = np.concatenate(var).ravel()
            self.means[i]  = np.mean(varVals, dtype=np.float64)
            self.scales[i] = 1/np.std(varVals, dtype=np.float64)

        # Start with a loop over the data sets that are mixed up
        for iMix in mixer:

            # Total number of events from current data set
            nEvents = len(dsets[iMix][self.config["trainVars"][0]])

            # Fill in the array holding info on all events 
            self.df["inputs"][offset:offset+nEvents,:] = np.vstack([dsets[iMix][var] for var in self.config["trainVars"]]).T
            for var in self.config["auxVars"]:
                self.df[var][offset:offset+nEvents] = dsets[iMix][var]

            self.df["year"][offset:offset+nEvents] = [years[iMix]]*nEvents

            # Remove dataset and release memory as soon as possible
            dsets[iMix] = None
            gc.collect()

            # Move pointer before filling with next data set
            offset += nEvents

        # Make a mask for any njets events we do not want to use
        combMaskNjets = None
        if self.config["Mask"]:
            for njets in self.config["Mask_nJet"]:
                if combMaskNjets == None:
                    combMaskNjets = (self.df[self.config["domainLabel"]] != njets)
                else:
                    combMaskNjets &= (self.df[self.config["domainLabel"]] != njets)
        else:
            combMaskNjets = (self.df[self.config["domainLabel"]] != -1)

        # Get a dictionary with mass points mapped to number of events
        temp = self.df[self.config["massLabel"]][combMaskNjets]
        unique, counts = np.unique(temp, return_counts=True)
        massDict = dict(zip(unique, counts))

        # Get a dictionary with process mapped to number of events
        # The "model" field of the dataframe is utilized as follows:
        # Nominal           POWHEG ttbar - 0
        #                 MADGRAPH ttbar - 1 
        # erdOn             POWHEG ttbar - 2
        # hdampUp           POWHEG ttbar - 3
        # hdampDown         POWHEG ttbar - 4
        # underlyingEvtUp   POWHEG ttbar - 5
        # underlyingEvtDown POWHEG ttbar - 6
        # fsrUp             POWHEG ttbar - 7
        # fsrDown           POWHEG ttbar - 8
        # isrUp             POWHEG ttbar - 9
        # isrDown           POWHEG ttbar - 10 

        # RPV signal - 100
        # SYY signal - 101
        # SHH signal - 102

        # JECup   - (add 10)
        # JECdown - (add 20)
        # JERup   - (add 30)
        # JERdown - (add 40)

        temp = self.df[self.config["modelLabel"]][combMaskNjets]
        unique, counts = np.unique(temp, return_counts=True)
        procDict = dict(zip(unique, counts))

        self.numBkgEvents = self.df[self.config["massLabel"]][combMaskNjets&(self.df[self.config["modelLabel"]]<100)].shape[0]
        self.bkgWeight = self.df[self.config["weightLabel"]][combMaskNjets&(self.df[self.config["modelLabel"]]<100)][0]
        self.numSigEvents = self.df[self.config["massLabel"]][combMaskNjets&(self.df[self.config["modelLabel"]]>=100)].shape[0]
        self.sigWeight = self.df[self.config["weightLabel"]][combMaskNjets&(self.df[self.config["modelLabel"]]>=100)][0]

        minNJetBin = self.df[self.config["domainLabel"]][combMaskNjets].min()
        numDomains = int(self.config["maxNJetBin"] + 1 - minNJetBin)

        evenSplit = 1
        if self.config["procCats"]:
            self.numBkgCategories = 1
            self.numSigCategories = 1
        if self.config["massCats"]:
            # Exclude bkg mass point from list
            self.numSigCategories *= len(massDict.keys())-1
        if self.config["njetsCats"]:
            self.numSigCategories *= numDomains
            self.numBkgCategories *= numDomains

        if self.config["procCats"] or self.config["massCats"] or self.config["njetsCats"]:
            evenSplit = 2

        self.sigSampleFactor = int(int(self.config["batch"]/evenSplit) / self.numSigCategories)
        self.bkgSampleFactor = int(int(self.config["batch"]/evenSplit) / self.numBkgCategories) 

        trueBatchSize = 0
        for p in procDict.keys(): 

            pcond = np.ones(self.df[self.config["modelLabel"]].shape[0]).astype(bool)
            pcond &= combMaskNjets

            isBackground = (p<100)

            process = None
            if not self.config["procCats"]:
                process = "EVTS"
                pcond &= (self.df[self.config["modelLabel"]]>=0)
            else:
                if isBackground:
                    process = "BKG"
                    pcond &= (self.df[self.config["modelLabel"]]<100)
                else:
                    process = "SIG"
                    pcond &= (self.df[self.config["modelLabel"]]>=100)

            for m in massDict.keys():

                # Skipped cross-matched mass and process
                if (m != 173.0 and isBackground) or (m == 173.0 and not isBackground):
                    continue

                mcond = np.ones(self.df[self.config["modelLabel"]].shape[0]).astype(bool)

                # Shift ttbar mass to 0 just for internal simplicity
                shiftedMass = 0
                if self.config["massCats"]:
                    mcond &= (self.df[self.config["massLabel"]]==m)
                    if not isBackground: 
                        shiftedMass = int(m)

                for n in np.arange(minNJetBin, self.config["maxNJetBin"]+1, dtype="float16"):

                    gc.collect()
                    theKey = ""
                    if   self.config["procCats"]:
                        theKey += str(process)
                    else:
                        theKey += "EVTS"
                    if self.config["massCats"]:
                        theKey += "_%d"%(int(shiftedMass))
                    if self.config["njetsCats"]:
                        theKey += "_%d"%(int(n))

                    # Get out if key already in data dict
                    # i.e. don't waste time with all those
                    # arrays down below
                    if theKey in self.data:
                        continue

                    ncond = np.ones(self.df[self.config["modelLabel"]].shape[0]).astype(bool)

                    # Depending on how finely we want to categorize and balance events
                    # The mask to pick out the correct events is defined accordingly
                    if self.config["njetsCats"]:
                        if n == float(self.config["maxNJetBin"]):
                            ncond &= ((self.df[self.config["domainLabel"]]>=n))
                        else:
                            ncond &= ((self.df[self.config["domainLabel"]]==n))

                    mask = np.where(pcond&mcond&ncond)[0]

                    factor = -1
                    maskSize = mask.shape[0]
                    if "BKG" in theKey or ("EVTS" in theKey and "_0" in theKey):
                        factor = self.bkgSampleFactor
                    else:
                        factor = self.sigSampleFactor

                    # Determine and get random indices to grab events for the batch
                    if factor > maskSize:
                        factor = maskSize

                    trueBatchSize += factor

                    # Store a mask of indices for all events of a given category
                    self.data[theKey] = {"mask" : mask, "factor" : factor}

        self.batchIndexContainer = np.zeros((trueBatchSize), dtype="int32")

        print("%s [INFO]: Bkg (sig) categories: %d (%d)"%(timeStamp(), self.numBkgCategories, self.numSigCategories))    
        print("%s [INFO]: Bkg (sig) sample factor: %d (%d)"%(timeStamp(), self.bkgSampleFactor, self.sigSampleFactor))
        print("%s [INFO]: Loading %d background and %d signal events"%(timeStamp(), self.numBkgEvents, self.numSigEvents))
