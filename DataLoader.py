import time
import math
import itertools
import numpy as np
import tensorflow as tf
import uproot4 as uproot
import tensorflow.keras as K

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

        # This call performs the entire setup of the DataLoader instance
        self.importData()

    # Flatten the dictionary of numpy arrays over the mass and njet categories
    # for use in the Validation part of the framework
    # If a process, mass, and/or Njets category is specified, that combination
    # returned while flattening over the other categories
    def getFlatData(self, process = None, mass = None, Njets = None):
        
        flatDictionary = {}
        for key, d in self.data.items():

            mask = (np.ones(d[list(d.keys())[0]].shape[0]))==1

            if process != None:
                mask &= (d[self.config["modelLabel"]]==process)

            if mass != None:
                mask &= (d[self.config["massLabel"]]==mass)

            if Njets != None:
                mask &= (d[self.config["domainLabel"]]==Njets)

            for label, data in d.items():

                if label not in flatDictionary:
                    flatDictionary[label] = data[mask]
                else:
                    flatDictionary[label] = np.append(flatDictionary[label], data[mask], axis=0)
           
        return flatDictionary

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
        return 1, self.config["maxNJetBin"] - self.config["minNJetBin"] + 1, 2, len(self.config["trainVars"])
   
    def getColumnHeaders(self):
        if self.columnHeaders is None:
            try:
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
        if len(self.datasets[process]) == 0:
            raise IndexError("No sample in samplesToRun")

        max_entries = None
        if self.config["debug"]:
            max_entries = 100

        variations = ["", "JECup", "JECdown", "JERup", "JERdown"]
        if self.config["debug"] or not self.config["useJECs"]:
            variations = [""]

        for suffix in variations:
            theVars = [v+suffix if v not in ["mass", "model", "Weight"] else v for v in self.variables]

            selection = None
            if "_1l" in self.config["tree"] or "_0l" in self.config["tree"]:
                print("Selecting 1L events with >= %d jets"%(self.config["nJets"]))
                selection = "(%s>=%s)"%(self.config["domainLabel"]+suffix,self.config["nJets"])

            for filename in self.datasets[process]:
            
                try:
                    f = uproot.open(filename)
                    tempnpf = f[self.config["tree"]+suffix].arrays(expressions=theVars, cut=selection, library="np", entry_stop=max_entries)
                    tempVars = list(tempnpf.keys())
                    npf = {}
                    for var in tempVars:
                        newVar = var.replace("JERup", "").replace("JECup", "").replace("JERdown", "").replace("JECdown", "")

                        if newVar == var:
                            npf[newVar] = tempnpf[var] 
                        else:
                            npf[newVar] = tempnpf.pop(var)

                    print("Loaded \"%s\" from input file \"%s\""%(self.config["tree"]+suffix,filename))

                    dsets.append(npf)
                    f.close()
                except Exception as e:
                    print("Skipping tree \"%s\" in file \"%s\" !" %(self.config["tree"]+suffix, filename) , e)
                    continue
        return dsets

    # This special function is called per-batch and constructs the batch
    # Based on how many background and signal events should appear as well as the 
    # different categories specified.
    def __getitem__(self, index):

        batchInputs   = np.empty([1,1]) 
        batchMassReg  = np.empty([1,1]) 
        batchModel    = np.empty([1,1])

        initialized = False
        for mnjet, data in self.data.items():
            
            batchIndices = None; factor = -1
            if "BKG" in mnjet or ("EVTS" in mnjet and "_0" in mnjet):
                factor = self.bkgSampleFactor
            else:
                factor = self.sigSampleFactor

            # Determine and get random indices to grab events for the batch
            if factor > data["domain"].shape[0]:
                factor = data["domain"].shape[0]
            batchIndices = np.random.choice(data["inputs"].shape[0], size=factor, replace=False)

            if not initialized:
                batchInputs  = data["inputs"][batchIndices]
                batchMassReg = data["massReg"][batchIndices]
                batchDisCo   = data["label"][batchIndices]
                initialized = True
            else:
                batchInputs  = np.append(batchInputs,  data["inputs"][batchIndices],  axis=0)
                batchMassReg = np.append(batchMassReg, data["massReg"][batchIndices], axis=0)
                batchDisCo   = np.append(batchDisCo,   data["label"][batchIndices],   axis=0)

        return batchInputs, tuple((batchDisCo, batchDisCo, batchMassReg))

    # Required function that tells tensorflow how many batches per epoch
    def __len__(self):
        return math.ceil((self.numBkgEvents+self.numSigEvents) / self.config["batch"])
    
    def importData(self):
    
        # Load events from the inputs ROOT files
        # and store them in numpy dataframes
        self.getColumnHeaders()

        df = {} 

        bgdsets = self.getDataSets(0)
        sgdsets = self.getDataSets(1)
        trainlists = [[] for var in self.config["trainVars"]]
        ziplists   = []

        # First do optimal loop to make input array of arrays
        for dset in bgdsets+sgdsets:
            iVar = 0
            templists = [[] for var in self.config["trainVars"]]
            for var in self.config["trainVars"]:
                trainlists[iVar] += dset[var].tolist()
                templists[iVar]  += dset[var].tolist()
                iVar             += 1

            # On the fly zip up results to make an array of inputs per event
            ziplists += list(map(np.array, zip(*templists)))

        # Calculated the mean and inverse std dev for the input variables
        self.means  = [np.mean(v) for v in trainlists]
        self.scales = [1.0 / np.std(v) for v in trainlists]

        df["inputs"] = np.array(ziplists)

        # Now put together arrays of aux vars
        for var in self.config["auxVars"]:
            auxlist = []
            for dset in bgdsets+sgdsets:
                auxlist += dset[var].tolist()
            df[var] = np.array(list(itertools.chain(auxlist)))

        # Make a mask for any njets events we do not want to use
        combMaskNjets = None
        if self.config["Mask"]:
            for njets in self.config["Mask_nJet"]:
                if combMaskNjets == None:
                    combMaskNjets = (df[self.config["domainLabel"]] != njets)
                else:
                    combMaskNjets &= (df[self.config["domainLabel"]] != njets)
        else:
            combMaskNjets = (df[self.config["domainLabel"]] != -1)

        # Get a dictionary with mass points mapped to number of events
        temp = df[self.config["massLabel"]][combMaskNjets]
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

        temp = df[self.config["modelLabel"]][combMaskNjets]
        unique, counts = np.unique(temp, return_counts=True)
        procDict = dict(zip(unique, counts))

        self.numBkgEvents = df[self.config["massLabel"]][combMaskNjets&(df[self.config["modelLabel"]]<100)].shape[0]
        self.numSigEvents = df[self.config["massLabel"]][combMaskNjets&(df[self.config["modelLabel"]]>=100)].shape[0]

        minNJetBin = df[self.config["domainLabel"]][combMaskNjets].min()
        numDomains = int(self.config["maxNJetBin"] + 1 - minNJetBin)

        for p in procDict.keys(): 

            pcond = np.ones(df[self.config["modelLabel"]].shape[0]).astype(bool)
            pcond &= combMaskNjets

            isBackground = (p<100)

            process = None
            if not self.config["procCats"]:
                process = "EVTS"
                pcond &= (df[self.config["modelLabel"]]>=0)
            else:
                if isBackground:
                    process = "BKG"
                    pcond &= (df[self.config["modelLabel"]]<100)
                else:
                    process = "SIG"
                    pcond &= (df[self.config["modelLabel"]]>=100)

            for m in massDict.keys():

                # Skipped cross-matched mass and process
                if (m != 173.0 and isBackground) or (m == 173.0 and not isBackground):
                    continue

                mcond = np.ones(df[self.config["modelLabel"]].shape[0]).astype(bool)

                # Shift ttbar mass to 0 just for internal simplicity
                shiftedMass = 0
                if self.config["massCats"]:
                    mcond &= (df[self.config["massLabel"]]==m)
                    if not isBackground: 
                        shiftedMass = int(m)

                for n in np.arange(minNJetBin, self.config["maxNJetBin"]+1, dtype=float):

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

                    ncond = np.ones(df[self.config["modelLabel"]].shape[0]).astype(bool)

                    # Depending on how finely we want to categorize and balance events
                    # The mask to pick out the correct events is defined accordingly
                    if self.config["njetsCats"]:
                        if n == float(self.config["maxNJetBin"]):
                            ncond &= ((df[self.config["domainLabel"]]>=n))
                        else:
                            ncond &= ((df[self.config["domainLabel"]]==n))

                    # Aquire all the needed information from the dataframes and store as numpy arrays
                    njets  = df[self.config["domainLabel"]][pcond&mcond&ncond]
                    inputs = df["inputs"][pcond&mcond&ncond]
                    reg    = df[self.config["regressionLabel"]][pcond&mcond&ncond]
                    massp  = df[self.config["massLabel"]][pcond&mcond&ncond]
                    weight = df[self.config["weightLabel"]][pcond&mcond&ncond]*self.config["lumi"]
                    model  = df[self.config["modelLabel"]][pcond&mcond&ncond]

                    labelSig   = (model>=100).astype(float)
                    labelBkg   = (model<100).astype(float)
                    labelTemp  = np.vstack((labelSig, labelBkg)).T
                    labelDisCo = np.concatenate((labelTemp, labelTemp), axis=1)

                    #setup and get domains
                    njetsTemp = njets
                    domain = np.zeros((njets.shape[0], numDomains))

                    njetsTemp[njetsTemp>self.config["maxNJetBin"]] = self.config["maxNJetBin"]
                    njetsTemp = njetsTemp - minNJetBin
    
                    domain[np.arange(njetsTemp.shape[0], dtype=int), njetsTemp.astype(int)] = 1

                    perms = np.random.permutation(domain.shape[0])

                    self.data[theKey] = {"domain" : domain[perms], "label" : labelDisCo[perms], "inputs" : inputs[perms], "massReg" : reg[perms], "mass" : massp[perms], "model" : model[perms], "weight" : weight[perms], "njets" : njets[perms]}

        evenSplit = 1
        if self.config["procCats"]:
            self.numBkgCategories = 1
            self.numSigCategories = 1
        if self.config["massCats"]:
            self.numSigCategories *= len(massDict.keys())
        if self.config["njetsCats"]:
            self.numSigCategories *= numDomains
            self.numBkgCategories *= numDomains

        if self.config["procCats"] or self.config["massCats"] or self.config["njetsCats"]:
            evenSplit = 2

        self.sigSampleFactor = int(int(self.config["batch"]/evenSplit) / self.numSigCategories)
        self.bkgSampleFactor = int(int(self.config["batch"]/evenSplit) / self.numBkgCategories) 

        print("Bkg (sig) categories: %d (%d)"%(self.numBkgCategories, self.numSigCategories))    
        print("Bkg (sig) sample factor: %d (%d)"%(self.bkgSampleFactor, self.sigSampleFactor))
        print("Loading %d background and %d signal events"%(self.numBkgEvents, self.numSigEvents))
