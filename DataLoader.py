import uproot
import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K

class DataLoader(K.utils.Sequence):

    def __init__(self, config, signalDataSet, backgroundDataSet, categorization=0):

        np.random.seed(config["seed"]) 

        self.config = config
        self.datasets = [tuple(backgroundDataSet), tuple(signalDataSet)]

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

        # Depending on final state, different pt requirements
        # and resultant objects are used
        ptCut = "pt30"
        if "0l" in config["tree"]:
            ptCut = "pt45"

        # Labels for extracting relevant information from the
        # dataframes constructed from the inputs ROOT files
        self.massLabel = "mass"
        self.domainLabel = "NGoodJets_%s_double"%(ptCut)
        self.regressionLabel = "stop1_ptrank_mass"
        self.modelLabel = "model"
        self.weightLabel = "Weight"

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
                mask &= (d["model"]==process)

            if mass != None:
                mask &= (d["mass"]==mass)

            if Njets != None:
                mask &= (d["njets"]==Njets)

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
    
    def getList(self):
        return self.config["allVars"]
    
    def getData(self):
        return self.data

    def getDataMeans(self):
        return self.means

    def getDataScales(self):
        return self.scales

    # Returns the shapes of the unique layers output in the NN model
    # Order: Mass regression, Domain (Njets), Double DisCo, Input
    def getShapes(self):
        return 1, self.config["maxNJetBin"] - self.config["minNJetBin"] + 1, 2, len(self.config["allVars"])
   
    def getColumnHeaders(self):
        if self.columnHeaders is None:
            try:
                sample = self.datasets[0][0]                
                f = uproot.open(sample)
                self.columnHeaders = f[self.config["tree"]].pandas.df().columns.tolist()
                f.close()
            except IndexError as e:
                print(e)
                raise IndexError("No sample in samplesToRun")
        return self.columnHeaders
    
    def checkVariables(self):
        for v in self.config["allVars"]:            
            if not v in self.columnHeaders:
                raise ValueError("Variable not found in input root file: %s"%v)
        
    # Process is 0 for background and 1 for signal
    def getDataSets(self, process = -1):
        if process != 0 and process != 1:
            raise IndexError("Must specify to load signal or background")

        dsets = []
        if len(self.datasets[process]) == 0:
            raise IndexError("No sample in samplesToRun")
        for filename in self.datasets[process]:
            for suffix in ["", "JECup", "JECdown", "JERup", "JERdown"]:
                try:
                    f = uproot.open(filename)
                    pdf = f[self.config["tree"]+suffix].pandas.df()

                    columns = list(pdf.columns)
                    newColumns = {header : header.replace(suffix, "") for header in columns}
                    pdf.rename(columns=newColumns)

                    dsets.append( pdf )
                    f.close()
                except Exception as e:
                    #print("Skipping tree \"%s\" !" %(self.config["tree"]+suffix) , e)
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
                batchInputs  = np.append(batchInputs,  data["inputs"][batchIndices], axis=0)
                batchMassReg = np.append(batchMassReg, data["massReg"][batchIndices], axis=0)
                batchDisCo   = np.append(batchDisCo,   data["label"][batchIndices], axis=0)

        return batchInputs, tuple((batchDisCo, batchDisCo, batchMassReg))

    # Required function that tells tensorflow how many batches per epoch
    def __len__(self):
        return math.ceil((self.numBkgEvents+self.numSigEvents) / self.config["batch"])
    
    def importData(self):
    
        # Load events from the inputs ROOT files
        # and store them in pandas dataframes temporarily
        variables = self.getList()
        self.getColumnHeaders()
        self.checkVariables()
        
        bgdsets = self.getDataSets(0)
        df = pd.concat(bgdsets)

        sgdsets = self.getDataSets(1)
        df = df.append(pd.concat(sgdsets))
        df = df.dropna()

        # Depending on channel, different pt used for jets and derived quantities
        ptCut = "pt30"
        if "0l" in self.config["tree"]:
            ptCut = "pt45"
            
        # Make a mask for any njets events we do not want to use
        combMaskNjets = None
        if self.config["Mask"]:
            for njets in self.config["Mask_nJet"]:
                if combMaskNjets == None:
                    combMaskNjets = (df[self.domainLabel] != njets)

                else:
                    combMaskNjets &= (df[self.domainLabel] != njets)
        else:
            combMaskNjets = (df[self.domainLabel] != -1)

        # Get a dictionary with mass points mapped to number of events
        temp = df[combMaskNjets][self.massLabel].values
        unique, counts = np.unique(temp, return_counts=True)
        massDict = dict(zip(unique, counts))

        # Get a dictionary with process mapped to number of events
        # The "model" field of the dataframe is utilized as follows:
        # Nominal           POWHEG ttbar - 0
        # erdOn             POWHEG ttbar - 1
        # hdampUp           POWHEG ttbar - 2
        # hdampDown         POWHEG ttbar - 3
        # underlyingEvtUp   POWHEG ttbar - 4
        # underlyingEvtDown POWHEG ttbar - 5
        # fsrUp             POWHEG ttbar - 2
        # fsrDown           POWHEG ttbar - 3
        # isrUp             POWHEG ttbar - 4
        # isrDown           POWHEG ttbar - 5

        # RPV signal - 100
        # SYY signal - 101
        # SHH signal - 102

        # JECup   - (add 10)
        # JECdown - (add 20)
        # JERup   - (add 30)
        # JERdown - (add 40)

        temp = df[combMaskNjets]["model"].values
        unique, counts = np.unique(temp, return_counts=True)
        procDict = dict(zip(unique, counts))

        # Calculated the mean and inverse std dev for the input variables
        self.means = np.mean(df[self.config["allVars"]].astype(float).values, 0)
        self.scales = 1.0 / np.std(df[self.config["allVars"]].astype(float).values, 0)

        self.numBkgEvents = df[combMaskNjets&(df["model"]<100)][self.massLabel].shape[0]
        self.numSigEvents = df[combMaskNjets&(df["model"]>=100)][self.massLabel].shape[0]

        minNJetBin = df[combMaskNjets][self.domainLabel].min()
        numDomains = int(self.config["maxNJetBin"] + 1 - minNJetBin)

        for p in procDict.keys(): 

            pcond = np.ones(df["model"].shape[0]).astype(bool)
            pcond &= combMaskNjets.values

            isBackground = (p<100)

            process = None
            if not self.config["procCats"]:
                process = "EVTS"
                pcond &= (df["model"]>=0)
            else:
                if isBackground:
                    process = "BKG"
                    pcond &= (df["model"]<100)
                else:
                    process = "SIG"
                    pcond &= (df["model"]>=100)

            for m in massDict.keys():

                # Skipped cross-matched mass and process
                if (m != 173.0 and isBackground) or (m == 173.0 and not isBackground):
                    continue

                mcond = np.ones(df["model"].shape[0]).astype(bool)

                # Shift ttbar mass to 0 just for internal simplicity
                shiftedMass = 0
                if self.config["massCats"]:
                    mcond &= ((df[self.massLabel]==m).values)
                    if not isBackground: 
                        shiftedMass = int(m)

                for n in np.arange(minNJetBin, self.config["maxNJetBin"]+1, dtype=float):

                    ncond = np.ones(df["model"].shape[0]).astype(bool)

                    # Depending on how finely we want to categorize and balance events
                    # The mask to pick out the correct events is defined accordingly
                    if self.config["njetsCats"]:
                        if n == float(self.config["maxNJetBin"]):
                            ncond &= ((df[self.domainLabel]>=n).values)
                        else:
                            ncond &= ((df[self.domainLabel]==n).values)

                    # Aquire all the needed information from the dataframes and store as numpy arrays
                    njets  = df[pcond&mcond&ncond][self.domainLabel].values
                    inputs = df[pcond&mcond&ncond][self.config["allVars"]].values
                    reg    = df[pcond&mcond&ncond][[self.regressionLabel]].values
                    massp  = df[pcond&mcond&ncond][self.massLabel].values
                    weight = df[pcond&mcond&ncond][self.weightLabel].values*self.config["lumi"]
                    model  = df[pcond&mcond&ncond][self.modelLabel].values

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

                    theKey = ""
                    if   self.config["procCats"]:
                        theKey += str(process)
                    else:
                        theKey += "EVTS"
                    if self.config["massCats"]:
                        theKey += "_%d"%(int(shiftedMass))
                    if self.config["njetsCats"]:
                        theKey += "_%d"%(int(n))

                    perms = np.random.permutation(domain.shape[0])
                    if theKey not in self.data:
                        self.data[theKey] = {"domain" : domain[perms], "label" : labelDisCo[perms], "inputs" : inputs[perms], "massReg" : reg[perms], "mass" : massp[perms], "model" : model[perms], "weight" : weight[perms], "njets" : njets[perms]}
                    else:
                        break

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
