import uproot
import numpy as np
import pandas as pd
from glob import glob

def getSamplesToRun(names):
    s = glob(names)
    if len(s) == 0:
        raise Exception("No files find that correspond to: "+names)
    return s

# Takes training vars, signal and background files and returns training data
def get_data(signalDataSet, backgroundDataSet, config, sgSampleFactor = 1, bgSampleFactor = 1, doSgWeight = False, doBgWeight = False):
    np.random.seed(config["seed"]) 

    dgSig = DataGetter.DefinedVariables(config["allVars"], signal = True,  background = False)
    dgBg = DataGetter.DefinedVariables(config["allVars"],  signal = False, background = True)

    dataBg = None; dataSig = None
    if    config["Mask"]: dataBg, dataSig = dgSig.importData(bgSamplesToRun = tuple(backgroundDataSet), sgSamplesToRun = tuple(signalDataSet), treename = config["tree"], doReweight=config["doReweight"], maxNJetBin=config["maxNJetBin"], njetsMask=config["Mask_nJet"])
    else:                 dataBg, dataSig = dgSig.importData(bgSamplesToRun = tuple(backgroundDataSet), sgSamplesToRun = tuple(signalDataSet), treename = config["tree"], doReweight=config["doReweight"], maxNJetBin=config["maxNJetBin"])


    # Change the weight to 1 if needed
    if config["doSgWeight"]: dataSig["Weight"] = config["lumi"]*dataSig["Weight"]
    else: dataSig["Weight"] = np.full(dataSig["Weight"].shape, 1)
    if config["doBgWeight"]: dataBg["Weight"] = config["lumi"]*dataBg["Weight"]
    else: dataBg["Weight"] = np.full(dataBg["Weight"].shape, 1)

    # Randomly shuffle the signal and background before mixing them together
    np.random.seed(config["seed"]) 
    perms = np.random.permutation(dataSig["data"].shape[0])
    for key in dataSig:
        dataSig[key] = dataSig[key][perms]

    perms = np.random.permutation(dataBg["data"].shape[0])
    for key in dataSig:
        dataBg[key] = dataBg[key][perms]

    print("Loading %dx %d background and %dx %d signal events"%(bgSampleFactor, len(dataBg["data"]), sgSampleFactor, len(dataSig["data"])))

    # Put signal and background data together in trainData dictionary
    trainDataArray = []
    for sf in range(0, bgSampleFactor):
        trainDataArray.append(dataBg)
    for sf in range(0, sgSampleFactor):
        trainDataArray.append(dataSig)

    np.random.shuffle(trainDataArray)

    trainData = {}
    for data in trainDataArray:
        for key in data:
            if key in trainData:
                trainData[key] = np.vstack([trainData[key], data[key][:]])
            else:
                trainData[key] = data[key][:]

    # Randomly shuffle the signal and background once put together
    perms = np.random.permutation(trainData["data"].shape[0])
    for key in trainData:
        trainData[key] = trainData[key][perms]

    # Get the rescale inputs to have unit variance centered at 0 between -1 and 1
    def scale(data):
        # Get the masks for different signal models
        for s in [0.0, 1.0, 2.0, 3.0]:
            mask = np.ma.getmaskarray(np.ma.masked_where((data["model"] == s) | (data["model"] == 0.0), data["model"]))
            
            maskName = None
            if   s == 1.0: maskName = "mask_RPV"
            elif s == 2.0: maskName = "mask_SYY"
            elif s == 3.0: maskName = "mask_SHH"
            elif s == 0.0: maskName = "mask_TT" 
            data[maskName]  = mask[:,0]

        # Get the masks for the different stop masses
        #for m in range(config["minStopMass"],config["maxStopMass"]+50,50):
        for m in range(300,1450,50):
            mask = np.ma.getmaskarray(np.ma.masked_where(data["masses"] == m, data["masses"]))
            data["mask_m"+str(m)] = mask[:,0]        

        # Get the masks for the different nJet bins
        for i in range(len(data["domain"][0])):
            mask = (1 - data["domain"][:,i]).astype(bool)
            data["mask_nJet_%02d" % (config["minNJetBin"]+i)] = ~np.array(mask)
            mask = (data["domain"][:,i]).astype(bool)
            data["mask_stuff_%02d" % (config["minNJetBin"]+i)] = ~np.array(mask)
        
        # Mask the njets in the training
        if config["Mask"]:
            combMaskNjets = None
            for njets in config["Mask_nJet"]:
                mask = data["mask_stuff_%02d" % njets]
                if combMaskNjets is None:
                    combMaskNjets = mask
                else:
                    combMaskNjets &= mask
            for key in data:
                data[key] = data[key][combMaskNjets]

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
                model = None
                if   "RPV" in filename: model = 1.0 
                elif "SYY" in filename: model = 2.0
                elif "SHH" in filename: model = 3.0
                else:                   model = 0.0

                pdf = f[treename].pandas.df()
                pdf.insert(0, "model", model)
                dsets.append( pdf )
                f.close()
            except Exception as e:
                print("Warning: \"%s\" has issues" % filename, e)
                continue
        return dsets

    def importDataWorker(self, variables, maxNJetBin, df, index, treename, njetsMask = [-1]):

        # Depending on channel, different pt used for jets
        ptCut = "pt30"
        if "0l" in treename:
            ptCut = "pt45"
            
        # Common column names to signal and background
        wgtColumnNames = ["Weight"]; massNames = ["mass"]; domainColumnNames = ["NGoodJets_%s_double"%(ptCut)]; njetsNames = ["NGoodJets_%s_double"%(ptCut)]; modelNames = ["model"]; recoMassNames = ["stop1_ptrank_mass"]

        combMaskNjets = None
        for njets in njetsMask:
            mask = df["NGoodJets_%s_double"%(ptCut)] != njets
            if combMaskNjets is None:
                combMaskNjets = mask
            else:
                combMaskNjets &= mask       
 
        npyNjetsFilter = df[combMaskNjets][massNames].values
        unique, counts = np.unique(npyNjetsFilter, return_counts=True)
        masses = dict(zip(unique, counts)).keys()

        #setup and get weights
        npyInputSampleWgts = df[wgtColumnNames].values

        # Use this npyMasses for excluding 7 jet events
        # Changes weights
        npyMassesFilter = df[combMaskNjets][massNames].values
        #npyMassesFilter[npyMassesFilter == 173.0] = 0.0

        dnjets = {}
        njetBinCounts = {"6" : 0, "7" : 0, "8" : 0, "9" : 0, "10" : 0, "11" : 0, "12" : 0, "13" : 0, "14" : 0, "15" : 0, "16" : 0, "17" : 0, "18" : 0, "19" : 0, "20" : 0}
        for mass in masses:
            npyNjetsFilter = df[combMaskNjets & (df["mass"]==mass)][njetsNames].values
            unique, counts = np.unique(npyNjetsFilter, return_counts=True)
            NjetsDict = dict(zip(unique, counts))
            for Njets, c in NjetsDict.items():
                njetBinCounts[str(int(Njets))] += c
                if mass == 173.0: dnjets[Njets] = c
                else:             dnjets[mass+Njets] = c

        totalRelevantEvents = 0
        for Njets in range(6, 21):
            if Njets not in njetsMask:
                totalRelevantEvents += njetBinCounts[str(Njets)]

        print("Total non-masked events to train on: %d"%(totalRelevantEvents))
            
        npyModels = df[modelNames].values
        npyMasses = df[massNames].values
        npyMassesReco = df[recoMassNames].values
        npyNjets  = df[njetsNames].values
        #npyMasses[npyMasses == 173.0] = 0.0

        #signalMasses = np.arange(175, 1400, 50)
        #signalMasses = np.append(signalMasses, 0)

        # WARNING!!! ANY OF THIS LITTLE SECTION HERE MOST LIKELY MESSES UP REWEIGHTING
        # GOOD THING WE ARE NOT USING REWEIGHTING
        """
        for i in range(0, len(npyMassesReco)):
            if npyMasses[i][0] != 173.0: continue
    
            #npyMasses[i] = np.random.choice(signalMasses, 1)
            npyMassesReco[i] = np.random.uniform(signalMasses[0], signalMasses[-1])
        """

        npyMassNjets = npyMasses+npyNjets

        #setup and get training data
        npyInputData = df[variables].astype(float).values

        #setup and get labels
        npyInputAnswers = np.zeros((npyInputData.shape[0], 2))
        npyInputAnswers[:,index] = 1

        #setup and get domains
        inputDomains = df[domainColumnNames]
        tempInputDomains = inputDomains.astype(int)
        tempInputDomains[tempInputDomains > maxNJetBin] = maxNJetBin 
        minNJetBin = tempInputDomains.min().values[0]
        numDomains = maxNJetBin + 1 - minNJetBin
        npyNJet = tempInputDomains.astype(float).values
        tempInputDomains = tempInputDomains - tempInputDomains.min()
   
        #sample weight for background masses
        npyInputDomain = np.zeros((npyInputData.shape[0], numDomains))
        npyInputDomain[np.arange(npyInputDomain.shape[0]), tempInputDomains.values.flatten()] = 1
        unique, counts = np.unique(npyMassesFilter, return_counts=True)

        #sample weight for signal masses
        d = dict(zip(unique, counts))
        mmin = -999; mmax = 10e10
        cmin = 0.0; cmax = 0.0
        for massNjets, c in dnjets.items():
            if c > mmin: mmin = c
            if c < mmax: mmax = c 
        for massNjets, c in dnjets.items():
            cmin += mmin
            cmax += mmax
            dnjets[massNjets] = round(float(mmin)/float(c),3)
            #dnjets[massNjets] = round(float(mmax)/float(c),3)

        return npyInputData, npyInputAnswers, npyInputDomain, npyInputSampleWgts, npyNJet, npyMasses, npyMassesReco, npyModels, npyMassNjets, cmax, dnjets

    def importData(self, bgSamplesToRun, sgSamplesToRun, treename = "myMiniTree", doReweight = False, maxNJetBin = 11, njetsMask = [-1]):

        #variables to train
        variables = self.getList()
        self.getColumnHeaders(bgSamplesToRun, treename)
        self.checkVariables(variables)
        
        # Do somethings for the background
        #load BG data files and get data
        # I guess use
        bgdsets = self.getDataSets(bgSamplesToRun, treename)
        dataBG = pd.concat(bgdsets)
        dataBG = dataBG.dropna()

        npyInputDataBG, npyInputAnswersBG, npyInputDomainBG, npyInputSampleWgtsBG, npyNJetBG, npyMassesBG, npyMassesRecoBG, npyModelsBG, npyMassNjetsBG, countsBG, dBG = self.importDataWorker(variables, maxNJetBin, dataBG, 1, treename, njetsMask=njetsMask)

        #########################################################################################

        # Now do same sort of things for signal
        #load SG data files and get data
        sgdsets = self.getDataSets(sgSamplesToRun, treename)
        dataSG = pd.concat(sgdsets)
        dataSG = dataSG.dropna()

        npyInputDataSG, npyInputAnswersSG, npyInputDomainSG, npyInputSampleWgtsSG, npyNJetSG, npyMassesSG, npyMassesRecoSG, npyModelsSG, npyMassNjetsSG, countsSG, dSG = self.importDataWorker(variables, maxNJetBin, dataSG, 0, treename, njetsMask=njetsMask)

        # Do some final changes to the weights for background and signal
        factor = 1.0
        for massNjets, w in dSG.items():
            if massNjets-150.0 < 0.0:
                dSG[massNjets] = 1.0
            else:
                dSG[massNjets] = round(factor*dSG[massNjets], 3)
       
        for massNjets, w in dBG.items():
            dBG[massNjets] = round(w*float(countsSG)/float(countsBG), 3)

        npySWBG = np.copy(npyMassNjetsBG); npySWSG = np.copy(npyMassNjetsSG)

        if doReweight:
            for i in range(0, len(npySWBG)): npySWBG[i][0] = dBG[npySWBG[i][0]]
            for i in range(0, len(npySWSG)): npySWSG[i][0] = dSG[npySWSG[i][0]]
        else:
            for i in range(0, len(npySWBG)): npySWBG[i][0] = 1.0
            for i in range(0, len(npySWSG)): npySWSG[i][0] = 1.0

        return {"data":npyInputDataBG, "labels":npyInputAnswersBG, "domain":npyInputDomainBG, "Weight":npyInputSampleWgtsBG, "nJet":npyNJetBG, "masses":npyMassesBG, "massesReco" : npyMassesRecoBG, "model":npyModelsBG, "massNjets":npyMassNjetsBG, "sample_weight":npySWBG}, {"data":npyInputDataSG, "labels":npyInputAnswersSG, "domain":npyInputDomainSG, "Weight":npyInputSampleWgtsSG, "nJet":npyNJetSG, "masses":npyMassesSG, "massesReco" : npyMassesRecoSG, "model":npyModelsSG, "massNjets":npyMassNjetsSG, "sample_weight":npySWSG}
