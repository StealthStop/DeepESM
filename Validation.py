from DataLoader import DataLoader
from Correlation import Correlation as cor

import os
import json
import logging
import numpy as np
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')

# Little incantation to display trying to X display
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.lines as ml

import mplhep as hep
plt.style.use([hep.style.ROOT,hep.style.CMS]) # For now ROOT defaults to CMS
plt.style.use({'legend.frameon':False,'legend.fontsize':16,'legend.edgecolor':'black'})

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score

plt.rcParams['pdf.fonttype'] = 42

class Validation:

    def __init__(self, model, config, loader, valLoader, evalLoader, testLoader, result_log=None):
        self.model = model
        self.config = config
        self.result_log = result_log
        self.metric = {}
        self.doLog = False
        self.loader = loader
        self.valLoader = valLoader 
        self.evalLoader = evalLoader
        self.testLoader = testLoader

        self.sample = {"RPV" : 100, "SYY" : 101, "SHH" : 102}

    def __del__(self):
        del self.model
        del self.config
        del self.result_log
        del self.metric

    def samplesLoaded(self, trainList, xvalList):

        trainListClean = []
        for train in trainList:
            trainListClean.append(train.replace("*","").replace("-",""))

        for xval in xvalList:
            xvalClean = xval.replace("*","").replace("-","")

            if xvalClean in trainListClean:
                return True

        return False

    def getAUC(self, fpr, tpr):
        try:
            return auc(fpr, tpr)
        except:
            print("Roc curve didn't work?????")
            print(fpr)
            print(tpr)
            return -1

    def getOutput(self, model, data, Sig, Bkg):
        return model.predict(data), model.predict(Sig), model.predict(Bkg)

    def getResults(self, output, output_sg, output_bg, outputNum=0, columnNum=0):
        return output[outputNum][:,columnNum].ravel(), output_sg[outputNum][:,columnNum].ravel(), output_bg[outputNum][:,columnNum].ravel()

    # Plot a set of 1D hists together, where the hists, colors, labels, weights
    # are provided as a list argument.
    def plotDisc(self, hists, colors, labels, weights, name, xlab, ylab, bins=100, arange=(0,1), doLog=False):
        # Plot predicted mass
        fig, ax = plt.subplots(figsize=(10, 10))
        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel(xlab); ax.set_xlabel(ylab)

        for i in range(0, len(hists)): 
            try:
                mean = round(np.average(hists[i], weights=weights[i]), 2)
                plt.hist(hists[i], bins=bins, range=arange, color="xkcd:"+colors[i], alpha=0.9, histtype='step', lw=2, label=labels[i]+" mean="+str(mean), density=True, log=doLog, weights=weights[i])
            except Exception as e:
                print("\nplotDisc: Could not plot %s hist for figure %s ::"%(labels[i],name), e, "\n")
                continue

        ax.legend(loc=1, frameon=False)
        fig.savefig(self.config["outputDir"]+"/%s.pdf"%(name), dpi=fig.dpi)        
        plt.close(fig)

    # Member function to plot the discriminant variable while making a selection on another discriminant
    # The result for signal and background are shown together
    def plotDiscWithCut(self, c, b1, b2, bw, s1, s2, sw, tag1, tag2, mass, Njets=-1, bins=100, arange=(0,1)):
        maskBGT = np.ma.masked_where(b2>c, b2).mask
        maskSGT = np.ma.masked_where(s2>c, s2).mask

        bnew = b1[maskBGT]; snew = s1[maskSGT]
        bwnew = bw[maskBGT]; swnew = sw[maskSGT]
        bw2new = np.square(bwnew); sw2new = np.square(swnew)

        bwnewBinned, binEdges  = np.histogram(bnew, bins=bins, range=arange, weights=bwnew)
        swnewBinned, binEdges  = np.histogram(snew, bins=bins, range=arange, weights=swnew)
        bw2newBinned, binEdges = np.histogram(bnew, bins=bins, range=arange, weights=bw2new)
        sw2newBinned, binEdges = np.histogram(snew, bins=bins, range=arange, weights=sw2new)

        if len(bw2newBinned) == 0: bw2newBinned = np.zeros(bins) 
        if len(bwnewBinned) == 0:  bwnewBinned  = np.zeros(bins)
        if len(sw2newBinned) == 0: sw2newBinned = np.zeros(bins)
        if len(swnewBinned) == 0:  swnewBinned  = np.zeros(bins)

        if not np.any(bw2newBinned): bw2newBinned += 10e-2
        if not np.any(bwnewBinned):  bwnewBinned += 10e-2
        if not np.any(sw2newBinned): sw2newBinned += 10e-2
        if not np.any(swnewBinned):  swnewBinned += 10e-2

        fig = plt.figure()

        ax = hep.histplot(h=bwnewBinned, bins=binEdges, w2=bw2newBinned, density=True, histtype="step", label="Background", alpha=0.9, lw=2)
        ax = hep.histplot(h=swnewBinned, bins=binEdges, w2=sw2newBinned, density=True, histtype="step", label="Signal (mass = %s GeV)"%(mass), alpha=0.9, lw=2, ax=ax)

        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel('A.U.'); ax.set_xlabel('Disc. %s'%(tag1))

        plt.text(0.05, 0.85, r"$\bf{Disc. %s}$ > %.3f"%(tag2,c), transform=ax.transAxes, fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))

        # Stupid nonsense to remove duplicate entries in legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-2:], labels[-2:], loc=2, frameon=False)
 
        if Njets == -1: fig.savefig(self.config["outputDir"]+"/Disc%s_BvsS_m%s.pdf"%(tag1,mass))
        else:           fig.savefig(self.config["outputDir"]+"/Disc%s_BvsS_m%s_Njets%d.pdf"%(tag1,mass,Njets))

        plt.close(fig)

    # Member function to plot the discriminant variable while making a selection on another discriminant
    # Compare the discriminant shape on either "side" of the selection on the other disc.
    def plotDiscWithCutCompare(self, c, d1, d2, dw, tag1, tag2, tag3, mass = "", Njets=-1, bins=100, arange=(0,1)):
        maskGT = np.ma.masked_where(d2>c, d2).mask; maskLT = ~maskGT

        dgt = d1[maskGT]; dlt = d1[maskLT]
        dwgt = dw[maskGT]; dwlt = dw[maskLT]
        dw2gt = np.square(dwgt); dw2lt = np.square(dwlt)

        dwgtBinned,  binEdges = np.histogram(dgt, bins=bins, range=arange, weights=dwgt)
        dw2gtBinned, binEdges = np.histogram(dgt, bins=bins, range=arange, weights=dw2gt)
        dwltBinned,  binEdges = np.histogram(dlt, bins=bins, range=arange, weights=dwlt)
        dw2ltBinned, binEdges = np.histogram(dlt, bins=bins, range=arange, weights=dw2lt)

        if len(dw2gtBinned) == 0: dw2gtBinned = np.zeros(bins)
        if len(dwgtBinned) == 0:  dwgtBinned  = np.zeros(bins)
        if len(dw2ltBinned) == 0: dw2ltBinned = np.zeros(bins)
        if len(dwltBinned) == 0:  dwltBinned  = np.zeros(bins)

        if not np.any(dw2gtBinned): dw2gtBinned += 10e-2
        if not np.any(dwgtBinned):  dwgtBinned += 10e-2
        if not np.any(dw2ltBinned): dw2ltBinned += 10e-2
        if not np.any(dwltBinned):  dwltBinned += 10e-2

        fig = plt.figure()

        ax = hep.histplot(h=dwgtBinned, bins=binEdges, w2=dw2gtBinned, density=True, histtype="step", label="Disc. %s > %.2f"%(tag2,c), alpha=0.9, lw=2)
        ax = hep.histplot(h=dwltBinned, bins=binEdges, w2=dw2ltBinned, density=True, histtype="step", label="Disc. %s < %.2f"%(tag2,c), alpha=0.9, lw=2, ax=ax)

        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel('A.U.'); ax.set_xlabel('Disc. %s'%(tag1))

        # Stupid nonsense to remove duplicate entries in legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-2:], labels[-2:], loc=2, frameon=False)

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/%s%s_Disc%s_Compare_Shapes.pdf"%(tag3, mass, tag1))
        else:           fig.savefig(self.config["outputDir"]+"/%s%s_Njets%d_Disc%s_Compare_Shapes.pdf"%(tag3, mass, Njets, tag1))

        plt.close(fig)

    # Plot loss of training vs test
    def plotAccVsEpoch(self, h1, h2, title, name):
        fig = plt.figure()
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        plt.plot(self.result_log.history[h1])
        plt.plot(self.result_log.history[h2])
        plt.title(title, pad=45.0)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        fig.savefig(self.config["outputDir"]+"/%s.pdf"%(name), dpi=fig.dpi)
        plt.close(fig)

    def plotAccVsEpochAll(self, h, n, val, title, name):
        fig = plt.figure()
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        plt.title(title, pad=45.0)
        plt.ylabel('loss')
        plt.xlabel('epoch')

        l = []
        for H in h:
            plt.plot(self.result_log.history["%s%s_loss"%(val,H)])
            l.append(n[h.index(H)])

        plt.legend(l, loc='best')
        fig.savefig(self.config["outputDir"]+"/%s.pdf"%(name), dpi=fig.dpi)
        plt.close(fig)

    def plotDiscPerNjet(self, tag, samples, sigMask, nBins=100):
        for sample in samples:
            trainSample = samples[sample][0]
            y_train_Sp = samples[sample][1]
            weights = samples[sample][2] 
            bins = np.linspace(0, 1, nBins)
            fig, ax = plt.subplots(figsize=(10, 10))
            hep.cms.label(data=True, paper=False, year=self.config["year"])
            ax.set_ylabel('Norm Events')
            ax.set_xlabel('Discriminator')
            for key in sorted(trainSample.keys()):
                if key.find("mask_nJet") != -1:
                    mask = True
                    if sample == "Sig": mask = sigMask
                    yt = y_train_Sp[trainSample[key]&mask]                
                    wt = weights[trainSample[key]&mask]
                    if yt.size != 0 and wt.size != 0:
                        plt.hist(yt, bins, alpha=0.9, histtype='step', lw=2, label=sample+" Train "+key, density=True, log=self.doLog, weights=wt)
            plt.legend(loc='best')
            fig.savefig(self.config["outputDir"]+"/nJet_"+sample+tag+".pdf", dpi=fig.dpi)
            plt.close(fig)

    def plotROC(self, dataMaskEval=None, dataMaskVal=None, tag="", y_eval=None, y_val=None, evalData=None, valData=None, xEval=None, xVal=None, yEval=None, yVal=None, evalLab=None, valLab=None):

        extra = None
        if "disc1" in tag or "Disc1" in tag: extra = "disc1"
        else:                                extra = "disc2"

        if extra not in self.config: self.config[extra] = {"eval_auc" : {}, "val_auc" : {}}

        fig = plt.figure()
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('', pad=45.0)

        if y_eval is None:
            plt.plot(xVal, yVal, color='xkcd:red', linestyle=":", label='Val (area = {:.3f})'.format(valLab))
            plt.plot(xEval, yEval, color='xkcd:red', label='Train (area = {:.3f})'.format(evalLab))
            self.config[extra]["eval_auc"]["total"] = evalLab
            self.config[extra]["val_auc"]["total"] = valLab

        else:
            NJetsRange = range(self.config["minNJetBin"], self.config["maxNJetBin"]+1)
            for NJets in NJetsRange:
                try:        
                    njets = float(NJets)
                    if self.config["Mask"] and (int(NJets) in self.config["Mask_nJet"]): continue

                    dataNjetsMaskEval = evalData["njets"]==njets
                    labels            = evalData["label"][:,0][dataMaskEval&dataNjetsMaskEval]
                    weights           = evalData["weight"][dataMaskEval&dataNjetsMaskEval]

                    y = y_eval[dataMaskEval&dataNjetsMaskEval]
                    if len(y)==0:
                        continue

                    fpr_eval, tpr_eval, thresholds_eval = roc_curve(labels, y, sample_weight=weights)
                    auc_eval = roc_auc_score(labels, y)    
                    plt.plot(fpr_eval, tpr_eval, label="$N_{\mathregular{jets}}$ = %d (Train)"%(int(NJets)) + " (area = {:.3f})".format(auc_eval))

                    self.config[extra]["eval_auc"]["Njets%d"%(int(NJets))] = auc_eval
                except Exception as e:
                    print("\nplotROC: Could not plot ROC for Njets = %d ::"%(int(NJets)), e, "\n")
                    continue

            plt.gca().set_prop_cycle(None)
            for NJets in NJetsRange:

                try:
                    njets = float(NJets)
                    if self.config["Mask"] and (int(NJets) in self.config["Mask_nJet"]): continue

                    dataNjetsMaskVal = valData["njets"] == njets
                    valLabels        = valData["label"][:,0][dataMaskVal&dataNjetsMaskVal]
                    valWeights       = valData["weight"][dataMaskVal&dataNjetsMaskVal]

                    yVal = y_val[dataMaskVal&dataNjetsMaskVal]
                    if len(yVal)==0:
                        continue

                    fpr_val, tpr_val, thresholds_val = roc_curve(valLabels, yVal, sample_weight=valWeights)
                    auc_val   = roc_auc_score(valLabels, yVal)
                    plt.plot(fpr_val, tpr_val, linestyle=":", label="$N_{\mathregular{jets}}$ = %d (Val)"%(int(NJets)) + " (area = {:.3f})".format(auc_val))

                    self.config[extra]["val_auc"]["Njets%d"%(int(NJets))] = auc_val
                except Exception as e:
                    print("\nplotROC: Could not plot ROC for Njets = %d ::"%(int(NJets)), e, "\n")
                    continue

        newtag = tag.replace(" ", "_")
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/roc_plot"+newtag+".pdf", dpi=fig.dpi)
        plt.close(fig)

    # Plot disc1 vs disc2 for both background and signal
    def plotD1VsD2SigVsBkgd(self, b1, b2, s1, s2, mass, Njets=-1):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax1)
        ax1.scatter(b1, b2, s=10, c='b', marker="s", label='background')
        ax1.scatter(s1, s2, s=10, c='r', marker="o", label='signal (mass = %s GeV)'%(mass))
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel("Disc. 1")
        ax1.set_ylabel("Disc. 2")
        plt.legend(loc='best');
        if Njets == -1: fig.savefig(self.config["outputDir"]+"/2D_SigVsBkgd_Disc1VsDisc2_m%s.pdf"%(mass), dpi=fig.dpi)        
        else:           fig.savefig(self.config["outputDir"]+"/2D_SigVsBkgd_Disc1VsDisc2_m%s_Njets%d.pdf"%(mass,Njets), dpi=fig.dpi)  
        plt.close(fig)

    def plotPandR(self, pval, rval, ptrain, rtrain, valLab, trainLab):
        fig = plt.figure()
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.plot(pval, rval, color='xkcd:black', label='Val (AP = {:.3f})'.format(valLab))
        plt.plot(ptrain, rtrain, color='xkcd:red', label='Train (AP = {:.3f})'.format(trainLab))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision and Recall curve', pad=45.0)
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/PandR_plot.pdf", dpi=fig.dpi)        
        plt.close(fig)

    def cutAndCount(self, c1s, c2s, b1, b2, bw, s1, s2, sw):
        # First get the total counts in region "D" for all possible c1, c2
        bcounts = {"a" : {}, "b" : {}, "c" : {}, "d" : {}, "A" : {}, "B" : {}, "C" : {}, "D" : {}, "A2" : {}, "B2" : {}, "C2" : {}, "D2" : {}}  
        scounts = {"a" : {}, "b" : {}, "c" : {}, "d" : {}, "A" : {}, "B" : {}, "C" : {}, "D" : {}, "A2" : {}, "B2" : {}, "C2" : {}, "D2" : {}}  

        for c1 in c1s:
            c1k = "%.2f"%c1
            if c1k not in bcounts["A"]:
                bcounts["A"][c1k] = {}; bcounts["A2"][c1k] = {}; bcounts["a"][c1k] = {};
                bcounts["B"][c1k] = {}; bcounts["B2"][c1k] = {}; bcounts["b"][c1k] = {};
                bcounts["C"][c1k] = {}; bcounts["C2"][c1k] = {}; bcounts["c"][c1k] = {};
                bcounts["D"][c1k] = {}; bcounts["D2"][c1k] = {}; bcounts["d"][c1k] = {};
            if c1k not in scounts["A"]:
                scounts["A"][c1k] = {}; scounts["A2"][c1k] = {}; scounts["a"][c1k] = {};
                scounts["B"][c1k] = {}; scounts["B2"][c1k] = {}; scounts["b"][c1k] = {};
                scounts["C"][c1k] = {}; scounts["C2"][c1k] = {}; scounts["c"][c1k] = {};
                scounts["D"][c1k] = {}; scounts["D2"][c1k] = {}; scounts["d"][c1k] = {};

            for c2 in c2s:
            
                c2k = "%.2f"%c2
                if c2k not in bcounts["A"][c1k]:
                    bcounts["A"][c1k][c2k] = 0.0; bcounts["A2"][c1k][c2k] = 0.0; bcounts["a"][c1k][c2k] = 0.0
                    bcounts["B"][c1k][c2k] = 0.0; bcounts["B2"][c1k][c2k] = 0.0; bcounts["b"][c1k][c2k] = 0.0
                    bcounts["C"][c1k][c2k] = 0.0; bcounts["C2"][c1k][c2k] = 0.0; bcounts["c"][c1k][c2k] = 0.0
                    bcounts["D"][c1k][c2k] = 0.0; bcounts["D2"][c1k][c2k] = 0.0; bcounts["d"][c1k][c2k] = 0.0
                if c2k not in scounts["A"][c1k]:
                    scounts["A"][c1k][c2k] = 0.0; scounts["A2"][c1k][c2k] = 0.0; scounts["a"][c1k][c2k] = 0.0
                    scounts["B"][c1k][c2k] = 0.0; scounts["B2"][c1k][c2k] = 0.0; scounts["b"][c1k][c2k] = 0.0
                    scounts["C"][c1k][c2k] = 0.0; scounts["C2"][c1k][c2k] = 0.0; scounts["c"][c1k][c2k] = 0.0
                    scounts["D"][c1k][c2k] = 0.0; scounts["D2"][c1k][c2k] = 0.0; scounts["d"][c1k][c2k] = 0.0

                mask1BGT = np.ma.masked_where(b1>c1, b1).mask; mask1BLT = ~mask1BGT
                mask1SGT = np.ma.masked_where(s1>c1, s1).mask; mask1SLT = ~mask1SGT

                mask2BGT = np.ma.masked_where(b2>c2, b2).mask; mask2BLT = ~mask2BGT
                mask2SGT = np.ma.masked_where(s2>c2, s2).mask; mask2SLT = ~mask2SGT

                maskBA = mask1BGT&mask2BGT; maskSA = mask1SGT&mask2SGT
                maskBB = mask1BGT&mask2BLT; maskSB = mask1SGT&mask2SLT
                maskBC = mask1BLT&mask2BGT; maskSC = mask1SLT&mask2SGT
                maskBD = mask1BLT&mask2BLT; maskSD = mask1SLT&mask2SLT

                bA = bw[maskBA]
                bcounts["A"][c1k][c2k]  = np.sum(bA)
                bcounts["a"][c1k][c2k]  = np.count_nonzero(bA)
                bcounts["A2"][c1k][c2k] = np.sum(np.square(bA))

                bB = bw[maskBB]
                bcounts["B"][c1k][c2k]  = np.sum(bB)
                bcounts["b"][c1k][c2k]  = np.count_nonzero(bB)
                bcounts["B2"][c1k][c2k] = np.sum(np.square(bB))

                bC = bw[maskBC]
                bcounts["C"][c1k][c2k]  = np.sum(bC)
                bcounts["c"][c1k][c2k]  = np.count_nonzero(bC)
                bcounts["C2"][c1k][c2k] = np.sum(np.square(bC))

                bD = bw[maskBD]
                bcounts["D"][c1k][c2k]  = np.sum(bD)
                bcounts["d"][c1k][c2k]  = np.count_nonzero(bD)
                bcounts["D2"][c1k][c2k] = np.sum(np.square(bD))

                sA = sw[maskSA]
                scounts["A"][c1k][c2k]  = np.sum(sA)           
                scounts["a"][c1k][c2k]  = np.count_nonzero(sA)           
                scounts["A2"][c1k][c2k] = np.sum(np.square(sA))
                                                                       
                sB = sw[maskSB]
                scounts["B"][c1k][c2k]  = np.sum(sB)           
                scounts["b"][c1k][c2k]  = np.count_nonzero(sB)           
                scounts["B2"][c1k][c2k] = np.sum(np.square(sB))
                                                                       
                sC = sw[maskSC]
                scounts["C"][c1k][c2k]  = np.sum(sC)           
                scounts["c"][c1k][c2k]  = np.count_nonzero(sC)           
                scounts["C2"][c1k][c2k] = np.sum(np.square(sC))
                                                                       
                sD = sw[maskSD]
                scounts["D"][c1k][c2k]  = np.sum(sD)           
                scounts["d"][c1k][c2k]  = np.count_nonzero(sD)           
                scounts["D2"][c1k][c2k] = np.sum(np.square(sD))

        return bcounts, scounts

    def findDiscCut4SigFrac(self, bcts, scts, minBkgEvts = 5):
        # Now calculate signal fraction and significance 
        # Pick c1 and c2 that give 30% sig fraction and maximizes significance
        significance = 0.0; finalc1 = -1.0; finalc2 = -1.0; 
        closureErr = 0.0; metric = 999.0
        invSigns = []; closeErrs = []; closeErrsUncs = []; c1out = []; c2out = []
        sFracsA = []; sFracsB = []; sFracsC = []; sFracsD = []
        sTotFracsA = []; sTotFracsB = []; sTotFracsC = []; sTotFracsD = []
        bTotFracsA = []; bTotFracsB = []; bTotFracsC = []; bTotFracsD = []

        for c1k, c2s in bcts["A"].items():
            for c2k, temp in c2s.items():

                bA = bcts["A"][c1k][c2k]; bB = bcts["B"][c1k][c2k]; bC = bcts["C"][c1k][c2k]; bD = bcts["D"][c1k][c2k]
                ba = bcts["a"][c1k][c2k]; bb = bcts["b"][c1k][c2k]; bc = bcts["c"][c1k][c2k]; bd = bcts["d"][c1k][c2k]
                sA = scts["A"][c1k][c2k]; sB = scts["B"][c1k][c2k]; sC = scts["C"][c1k][c2k]; sD = scts["D"][c1k][c2k]

                bA2 = bcts["A2"][c1k][c2k]; bB2 = bcts["B2"][c1k][c2k]; bC2 = bcts["C2"][c1k][c2k]; bD2 = bcts["D2"][c1k][c2k]

                bTotal = bA + bB + bC + bD
                sTotal = sA + sB + sC + sD

                tempsbfracA = -1.0; tempsTotfracA = -1.0; tempbTotfracA = -1.0
                tempsbfracB = -1.0; tempsTotfracB = -1.0; tempbTotfracB = -1.0 
                tempsbfracC = -1.0; tempsTotfracC = -1.0; tempbTotfracC = -1.0 
                tempsbfracD = -1.0; tempsTotfracD = -1.0; tempbTotfracD = -1.0 
                if bA + sA > 0.0: tempsbfracA = sA / (sA + bA)
                if bB + sB > 0.0: tempsbfracB = sB / (sB + bB)
                if bC + sC > 0.0: tempsbfracC = sC / (sC + bC)
                if bD + sD > 0.0: tempsbfracD = sD / (sD + bD)

                tempbfracA = bA / bTotal; tempsfracA = sA / sTotal
                tempbfracB = bB / bTotal; tempsfracB = sB / sTotal
                tempbfracC = bC / bTotal; tempsfracC = sC / sTotal
                tempbfracD = bD / bTotal; tempsfracD = sD / sTotal

                tempsignificance = 0.0; tempclosureerr = -999.0; tempmetric = 999.0; tempclosureerrunc = -999.0

                if bD > 0.0 and bA > 0.0:
                    tempclosureerr    = abs(1.0 - (bB * bC) / (bA * bD))
                    tempclosureerrunc = (((bB2**0.5 * bC)/(bA * bD))**2.0 + \
                                         ((bB * bC2**0.5)/(bA * bD))**2.0 + \
                                         ((bB * bC * bA2**0.5)/(bA**2.0 * bD))**2.0 + \
                                         ((bB * bC * bD2**0.5)/(bA * bD**2.0))**2.0)**0.5

                if bA > 0.0: tempsignificance += (sA / (bA + (0.3*bA)**2.0 + (tempclosureerr*bA)**2.0)**0.5)**2.0
                #if bB > 0.0: tempsignificance += (sB / (bB + (0.3*bB)**2.0 + (tempclosureerr*bB)**2.0)**0.5)**2.0
                #if bC > 0.0: tempsignificance += (sC / (bC + (0.3*bC)**2.0 + (tempclosureerr*bC)**2.0)**0.5)**2.0
                #if bD > 0.0: tempsignificance += (sD / (bD + (0.3*bD)**2.0 + (tempclosureerr*bD)**2.0)**0.5)**2.0

                tempsignificance = tempsignificance**0.5

                if tempsignificance > 0.0 and tempclosureerr > 0.0:
                    invSigns.append(1.0 / tempsignificance)

                    closeErrs.append(abs(tempclosureerr))
                    closeErrsUncs.append(tempclosureerrunc)
                    c1out.append(float(c1k))
                    c2out.append(float(c2k))

                    sFracsA.append(float(tempsbfracA)); sTotFracsA.append(float(tempsfracA)); bTotFracsA.append(float(tempbfracA))
                    sFracsB.append(float(tempsbfracB)); sTotFracsB.append(float(tempsfracB)); bTotFracsB.append(float(tempbfracB))
                    sFracsC.append(float(tempsbfracC)); sTotFracsC.append(float(tempsfracC)); bTotFracsC.append(float(tempbfracC))
                    sFracsD.append(float(tempsbfracD)); sTotFracsD.append(float(tempsfracD)); bTotFracsD.append(float(tempbfracD))

                # Compute metric if...
                # signal fraction in B, C, and D regions is < 10%
                # total background fraction in A is greater than 5%

                if ba > minBkgEvts and \
                   bb > minBkgEvts and \
                   bc > minBkgEvts and \
                   bd > minBkgEvts:

                    tempmetric = tempclosureerr**2.0 + (1.0 / tempsignificance)**2.0

                    #tempmetric = 1.0 / tempsignificance

                if tempmetric < metric:

                    finalc1 = c1k; finalc2 = c2k
                    metric = tempmetric
                    significance = tempsignificance
                    closureErr = tempclosureerr
                
        return finalc1, finalc2, significance, closureErr, invSigns, closeErrs, closeErrsUncs, c1out, c2out, sFracsA, sFracsB, sFracsC, sFracsD, sTotFracsA, sTotFracsB, sTotFracsC, sTotFracsD, bTotFracsA, bTotFracsB, bTotFracsC, bTotFracsD

    # Define closure as how far away prediction for region D is compared to actual 
    def simpleClosureABCD(self, bNA, bNB, bNC, bND, bNAerr, bNBerr, bNCerr, bNDerr):
        # Define A: > c1, > c2        C    |    A    
        # Define B: > c1, < c2   __________|__________        
        # Define C: < c1, > c2             |        
        # Define D: < c1, < c2        D    |    B    

        num = bNC * bNB; den = bND * bNA

        bNApred = -1.0; bNApredUnc = 0.0
        if bND > 0.0:
            bNApred = num / bND
            bNApredUnc = ((bNC * bNBerr / bND)**2.0 + (bNCerr * bNB / bND)**2.0 + (bNC * bNB * bNDerr / bND**2.0)**2.0)**0.5

        if den > 0.0:
            closureErr = ((bNB * bNCerr / den)**2.0 + (bNBerr * bNC / den)**2.0 + ((num * bNAerr) / (den * bNA))**2.0 + ((num * bNDerr) / (den * bND))**2.0)**0.5
            closure = num / den
        else:
            closureErr = -999.0
            closure = -999.0

        return closure, closureErr, bNApred, bNApredUnc

    def plotDisc1vsDisc2(self, disc1, disc2, bw, c1, c2, significance, tag, mass = "", Njets = -1, nBins = 100):
        fig = plt.figure() 
        corr = 999.0
        try: corr = cor.pearson_corr(disc1, disc2)
        except: print("Correlation coefficient could not be calculated!")
        plt.hist2d(disc1, disc2, bins=[nBins, nBins], range=[[0, 1], [0, 1]], cmap=plt.cm.jet, weights=bw, cmin = bw.min())
        plt.colorbar()
        ax = plt.gca()
        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="red", linewidth=2); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="red", linewidth=2)
        ax.add_line(l1); ax.add_line(l2)
        ax.set_ylabel("Disc. 2"); ax.set_xlabel("Disc. 1")
        plt.text(0.05, 0.90, r"$\bf{CC}$ = %.3f"%(corr), fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))

        if significance > 0.0:
            plt.text(0.05, 0.95, r"$\bf{Significance}$ = %.3f"%(significance), fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))
        else:
            plt.text(0.05, 0.95, r"$\bf{Significance}$ = N/A", fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        fig.tight_layout()
        if Njets == -1: fig.savefig(self.config["outputDir"]+"/2D_%s%s_Disc1VsDisc2.pdf"%(tag,mass), dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/2D_%s%s_Disc1VsDisc2_Njets%d.pdf"%(tag,mass,Njets), dpi=fig.dpi)

        plt.close(fig)

        return corr

    def plotBkgSigFracVsBinEdges(self, sFracsA, sFracsB, sFracsC, sFracsD, sTotFracsA, sTotFracsB, sTotFracsC, sTotFracsD, bTotFracsA, bTotFracsB, bTotFracsC, bTotFracsD, d1edges, d2edges, c1, c2, minEdge, maxEdge, edgeWidth, Njets = -1):

        nBins = int((1.0 + edgeWidth)/edgeWidth)

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=sFracsA, cmin = 0.00001, cmax = 1.0, vmin = 0.0, vmax = 1.0)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/SigFracA_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/SigFracA_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=sFracsB, cmin = 0.00001, cmax = 1.0, vmin = 0.0, vmax = 1.0)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/SigFracB_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/SigFracB_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=sFracsC, cmin = 0.00001, cmax = 1.0, vmin = 0.0, vmax = 1.0)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/SigFracC_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/SigFracC_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=sFracsD, cmin = 0.00001, cmax = 1.0, vmin = 0.0, vmax = 1.0)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/SigFracD_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/SigFracD_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

        ################################################################

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=sTotFracsA, cmin = 0.00001, cmax = 1.0, vmin = 0.0, vmax = 1.0)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/SigTotFracA_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/SigTotFracA_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=sTotFracsB, cmin = 0.00001, cmax = 1.0, vmin = 0.0, vmax = 1.0)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/SigTotFracB_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/SigTotFracB_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=sTotFracsC, cmin = 0.00001, cmax = 1.0, vmin = 0.0, vmax = 1.0)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/SigTotFracC_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/SigTotFracC_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=sTotFracsD, cmin = 0.00001, cmax = 1.0, vmin = 0.0, vmax = 1.0)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/SigTotFracD_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/SigTotFracD_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=bTotFracsA, cmin = 0.00001, cmax = 1.0, vmin = 0.0, vmax = 1.0)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/BkgTotFracA_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/BkgTotFracA_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=bTotFracsB, cmin = 0.00001, cmax = 1.0, vmin = 0.0, vmax = 1.0)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/BkgTotFracB_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/BkgTotFracB_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=bTotFracsC, cmin = 0.00001, cmax = 1.0, vmin = 0.0, vmax = 1.0)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/BkgTotFracC_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/BkgTotFracC_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=bTotFracsD, cmin = 0.00001, cmax = 1.0, vmin = 0.0, vmax = 1.0)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/BkgTotFracD_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/BkgTotFracD_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

    def plotClosureVsDisc(self, closeErrs, closeErrsUncs, d1edges, d2edges, edgeWidth, disc = -1, Njets = -1):

       x25     = []; x50     = []; x75     = []
       close25 = []; close50 = []; close75 = []

       close25unc = []; close50unc = []; close75unc = []

       edges = (d1edges, d2edges)

       for i in range(0, len(closeErrs)):
           if   edges[disc-1][i] == 0.24: 
               x25.append(edges[3-disc-1][i])
               close25.append(closeErrs[i])
               close25unc.append(closeErrsUncs[i])
           elif edges[disc-1][i] == 0.50: 
               x50.append(edges[3-disc-1][i])
               close50.append(closeErrs[i])
               close50unc.append(closeErrsUncs[i])
           elif edges[disc-1][i] == 0.76: 
               x75.append(edges[3-disc-1][i])
               close75.append(closeErrs[i])
               close75unc.append(closeErrsUncs[i])

       fig, ax = plt.subplots(figsize=(10, 10))

       xWidths25 = [edgeWidth for i in range(0, len(x25))]
       xWidths50 = [edgeWidth for i in range(0, len(x50))]
       xWidths75 = [edgeWidth for i in range(0, len(x75))]

       ax.errorbar(x25, close25, yerr=close25unc, label="Disc. %d = 0.25"%(disc), xerr=xWidths25, fmt='', color="red",   lw=0, elinewidth=2, marker="o", markerfacecolor="red")
       ax.errorbar(x50, close50, yerr=close50unc, label="Disc. %d = 0.50"%(disc), xerr=xWidths50, fmt='', color="blue",  lw=0, elinewidth=2, marker="o", markerfacecolor="blue")
       ax.errorbar(x75, close75, yerr=close75unc, label="Disc. %d = 0.75"%(disc), xerr=xWidths75, fmt='', color="green", lw=0, elinewidth=2, marker="o", markerfacecolor="green")

       ax.set_ylim((0.0, 1.0))
       ax.set_ylabel("ABCD Closure Error"); ax.set_xlabel("Disc. %d Value"%(3-disc))
       plt.legend(loc='best')

       hep.cms.label(data=True, paper=False, year=self.config["year"])

       fig.tight_layout()

       if Njets == -1: fig.savefig(self.config["outputDir"]+"/Closure_Slices_Disc%d.pdf"%(disc), dpi=fig.dpi)
       else:           fig.savefig(self.config["outputDir"]+"/Closure_Slices_Disc%d_Njets%s.pdf"%(disc,Njets), dpi=fig.dpi)

       plt.close(fig)

    def plotMetricVsBinEdges(self, invSigns, closureErr, closureErrUnc, d1edges, d2edges, c1, c2, minEdge, maxEdge, edgeWidth, Njets = -1):

        nBins = int((1.0 + edgeWidth)/edgeWidth)

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=np.reciprocal(invSigns), cmin=10e-10, cmax=5.0, vmin = 0.0, vmax = 3.0)

        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/Sign_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/Sign_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=closureErr, cmin=10e-10, cmax=2.5, vmin = 0.0, vmax = 0.3)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/CloseErr_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/CloseErr_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

        fig = plt.figure() 
        plt.hist2d(d1edges, d2edges, bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.jet, weights=closureErrUnc, cmin=10e-10, cmax=2.5, vmin = 0.0, vmax = 0.5)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge")
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/CloseErrUnc_vs_Disc1Disc2.pdf", dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/CloseErrUnc_vs_Disc1Disc2_Njets%s.pdf"%(Njets), dpi=fig.dpi)

        plt.close(fig)

    def plotBinEdgeMetricComps(self, finalSign, finalClosureErr, invSign, closeErr, edges, d1edge, d2edge, Njets = -1):

        fig = plt.figure()
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        ax = plt.gca()
        plt.scatter(invSign, closeErr, color='xkcd:silver', marker="o", label="1 - Pred./Obs. vs 1 / Significance")

        if finalSign != 0.0: 
            plt.scatter([1.0/finalSign], [finalClosureErr], s=100, color='xkcd:red', marker="o", label="Chosen Solution")
        plt.xlabel('1 / Significance')
        plt.ylabel('|1 - Pred./Obs.|')
        plt.legend(loc='best')

        plt.ylim(bottom=0)
        plt.xlim(left=0)

        plt.gca().invert_yaxis()

        plt.text(0.50, 0.85, r"$%.2f < \bf{Disc.\;1\;Edge}$ = %s < %.2f"%(edges[0],d1edge,edges[-1]), transform=ax.transAxes, fontfamily='sans-serif', fontsize=16)
        plt.text(0.50, 0.80, r"$%.2f < \bf{Disc.\;2\;Edge}$ = %s < %.2f"%(edges[0],d2edge,edges[-1]), transform=ax.transAxes, fontfamily='sans-serif', fontsize=16)
        fig.tight_layout()

        if Njets == -1:
            fig.savefig(self.config["outputDir"]+"/InvSign_vs_CloseErr.pdf", dpi=fig.dpi)        
        else:
            fig.savefig(self.config["outputDir"]+"/InvSign_vs_CloseErr_Njets%d.pdf"%(Njets), dpi=fig.dpi)        

        plt.close(fig)

        return np.average(closeErr), np.std(closeErr)

    def plotNjets(self, bkgd, bkgdErr, sig, sigErr, label):

        newB = [bkgd[0]]; newB += bkgd
        newS = [sig[0]];  newS += sig 
        errX     = [i+0.5 for i in range(0, len(bkgd))]

        sign = 0.0
        for i in range(0, len(bkgd)):
            if bkgd[i] > 0.0: sign += (sig[i] / (bkgd[i] + (0.3*bkgd[i])**2.0)**0.5)**2.0
        sign = sign**0.5

        binEdges = [i for i in range(0, len(newB))]

        fig = plt.figure()
        ax = plt.gca()
        ax.set_yscale("log")
        ax.set_ylim([1,10e4])

        plt.step(binEdges, newB, label="Background", where="pre", color="black", linestyle="solid")
        plt.step(binEdges, newS, label="Signal",     where="pre", color="red",   linestyle="solid")
        plt.errorbar(errX, bkgd, yerr=bkgdErr, xerr=None, fmt='', ecolor="black", elinewidth=2, color=None, lw=0) 
        plt.errorbar(errX, sig,  yerr=sigErr,  xerr=None, fmt='', ecolor="red", elinewidth=2, color=None, lw=0) 

        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)

        plt.xlabel('Number of jets')
        plt.ylabel('Events')
        plt.legend(loc='best')
        plt.text(0.05, 0.94, r"Significance = %.2f"%(sign), transform=ax.transAxes, fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))

        fig.savefig(self.config["outputDir"]+"/Njets_Region_%s.pdf"%(label))

        plt.close(fig)

        return sign

    def plotNjetsClosure(self, bkgd, bkgdUnc, bkgdPred, bkgdPredUnc, bkgdSign):

        binCenters = []
        xErr       = []
        abcdPull   = []
        abcdError  = []
        unc        = []
        predUnc    = []
        obs        = []
        pred       = []

        totalChi2 = 0.0; wtotalChi2 = 0.0; ndof = 0; totalSig = 0.0

        Njets = list(range(self.config["minNJetBin"], self.config["maxNJetBin"]+1))
        if self.config["Mask"]:
            for i in self.config["Mask_nJet"]:
                del(Njets[Njets.index(i)])
 
        for i in range(0, len(Njets)):

            if bkgdUnc[i] != 0.0:
                binCenters.append(Njets[i])
                unc.append(bkgdUnc[i])
                predUnc.append(bkgdPredUnc[i])
                obs.append(bkgd[i])
                pred.append(bkgdPred[i])
                xErr.append(0.5)
                pull = (bkgdPred[i]-bkgd[i])/bkgdUnc[i]
                closureError = 1.0 - bkgdPred[i]/bkgd[i]
                abcdPull.append(pull)
                abcdError.append(closureError)
                totalChi2 += pull**2.0
                wtotalChi2 += bkgdSign[i] * pull**2.0
                totalSig += bkgdSign[i]
                ndof += 1

        fig = plt.figure()
        gs = fig.add_gridspec(8, 1)
        ax = fig.add_subplot(111)
        ax1 = fig.add_subplot(gs[0:6])
        ax2 = fig.add_subplot(gs[6:8], sharex=ax1)
       
        fig.subplots_adjust(hspace=0)
        
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        
        ax1.set_yscale("log")
        if ndof != 0:
            ax1.text(0.05, 0.1, "$\chi^2$ / ndof = %3.2f"%(totalChi2/float(ndof)), horizontalalignment="left", verticalalignment="center", transform=ax1.transAxes, fontsize=18)
            ax1.text(0.05, 0.20, "$\chi^2$ (weighted) / ndof = %3.2f"%(wtotalChi2/float(ndof)), horizontalalignment="left", verticalalignment="center", transform=ax1.transAxes, fontsize=18)
       
        ax1.errorbar(binCenters, pred, yerr=predUnc, label="Predicted", xerr=xErr, fmt='', color="red",   lw=0, elinewidth=2, marker="o", markerfacecolor="red")
        ax1.errorbar(binCenters, obs,  yerr=unc,     label="Observed",  xerr=xErr, fmt='', color="black", lw=0, elinewidth=2, marker="o", markerfacecolor="black")

        lowerNjets = Njets[0]

        ax1.set_xlim([lowerNjets-0.5,self.config["maxNJetBin"]+0.5])
        
        plt.xticks(Njets)

        ax2.errorbar(binCenters, abcdError, yerr=None,        xerr=xErr, fmt='', color="blue",  lw=0, elinewidth=2, marker="o", markerfacecolor="blue")
        ax2.axhline(y=0.0, color="black", linestyle="dashed", lw=1)
        ax2.grid(color="black", which="both", axis="y")

        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax1)
        
        ax2.set_xlabel('Number of jets')
        ax2.set_ylabel('1 - Pred./Obs.', fontsize="small")
        ax1.set_ylabel('Unweighted Event Counts')
        ax1.legend(loc='best')
        
        ax2.set_ylim([-1.6, 1.6])
        
        fig.savefig(self.config["outputDir"]+"/Njets_Region_A_PredVsActual.pdf")
        
        plt.close(fig)

        if totalSig == 0.0:
            wtotalChi2 = totalChi2

        return totalChi2, wtotalChi2, ndof

    def makePlots(self, doQuickVal=False, evalMass="400", evalModel="RPV_SYY_SHH"):
        NJetsRange = range(self.config["minNJetBin"], self.config["maxNJetBin"]+1)

        # For validating that the training on the samples worked well
        # Pick a safe mass point and model that was present in the training set
        # The "validation" events will be drawn based on this model/mass combination
        tRange = list(range(self.config["minStopMass"], self.config["maxStopMass"]+50, 50))
        valMass = None
        if 550 in tRange:
            valMass = 550
        else:
            valMass = tRange[0]

        valModel  = self.config["trainModel"].split("_")[0]

        # eval events are samples possibly not seen by network during training
        # val events are the 10% of the samples used for training and to verify no overtraining
        evalSig = {}; evalBkg = {}; evalData = {}

        # For vanilla validation of the network with the 10% of non-training events
        # Use TT POWHEG as that sample is present nearly all the time, a bit dubious...
        valData = self.valLoader.getFlatData()
        valSig  = self.valLoader.getFlatData(process=self.sample[valModel])
        valBkg  = self.valLoader.getFlatData(process=0)

        # If there is a xvalLoader that means we are evaluating the network
        # on events it has not seen and are not in the train+val+test sets
        if self.evalLoader != None:
            evalData = self.evalLoader.getFlatData()
            evalSig  = self.evalLoader.getFlatData(process=self.sample[evalModel]) 
            evalBkg  = self.evalLoader.getFlatData(process=self.config["evalBkg"])
        
        # Making it to the else means the events we want to evaluate
        # are contained within the train+val+test sets
        else:
            trainDataTmp = self.loader.getFlatData()
            testDataTmp  = self.testLoader.getFlatData()

            trainSigTmp = self.loader.getFlatData(process=self.sample[evalModel]) 
            trainBkgTmp = self.loader.getFlatData(process=self.config["evalBkg"])      

            valSigTmp = self.valLoader.getFlatData(process=self.sample[evalModel]) 
            valBkgTmp = self.valLoader.getFlatData(process=self.config["evalBkg"])

            testSigTmp = self.testLoader.getFlatData(process=self.sample[evalModel])
            testBkgTmp = self.testLoader.getFlatData(process=self.config["evalBkg"])

            for key in trainDataTmp.keys():
                evalData[key] = np.concatenate((trainDataTmp[key], valData[key],   testDataTmp[key]), axis=0)
                evalSig[key]  = np.concatenate((trainSigTmp[key],  valSigTmp[key], testSigTmp[key]),  axis=0)
                evalBkg[key]  = np.concatenate((trainBkgTmp[key],  valBkgTmp[key], testBkgTmp[key]),  axis=0)
        
        massMaskEval = evalSig["mass"] == float(evalMass)
        massMaskVal  = valSig["mass"]  == float(valMass)

        # Make signal model mask for signal training dataset
        rpvMaskEval = evalSig["model"]==self.sample["RPV"]
        syyMaskEval = evalSig["model"]==self.sample["SYY"]
        shhMaskEval = evalSig["model"]==self.sample["SHH"]
        bkgMaskEval = evalSig["model"]==self.config["evalBkg"]

        # Make signal model mask for mixed training dataset
        rpvMaskDataEval = evalData["model"]==self.sample["RPV"]
        syyMaskDataEval = evalData["model"]==self.sample["SYY"]
        shhMaskDataEval = evalData["model"]==self.sample["SHH"]
        bkgMaskDataEval = evalData["model"]==self.config["evalBkg"]

        # Make signal model mask for signal validation dataset
        rpvMaskVal = valSig["model"]==self.sample["RPV"]
        syyMaskVal = valSig["model"]==self.sample["SYY"]
        shhMaskVal = valSig["model"]==self.sample["SHH"]
        bkgMaskVal = valSig["model"]==0

        # Make signal model mask for mixed validation dataset
        rpvMaskDataVal = valData["model"]==self.sample["RPV"]
        syyMaskDataVal = valData["model"]==self.sample["SYY"]
        shhMaskDataVal = valData["model"]==self.sample["SHH"]
        bkgMaskDataVal = valData["model"]==0

        sigMaskEval = bkgMaskEval; sigMaskDataEval = bkgMaskDataEval; sigMaskVal = bkgMaskVal; sigMaskDataVal = bkgMaskDataVal
        if   "RPV" in evalModel:
            if sigMaskEval is None:
                sigMaskEval = rpvMaskEval
            else:
                sigMaskEval |= rpvMaskEval

            if sigMaskDataEval is None:
                sigMaskDataEval = rpvMaskDataEval
            else:
                sigMaskDataEval |= rpvMaskDataEval

        if "RPV" in valModel:
            if sigMaskVal is None:
                sigMaskVal = rpvMaskVal
            else:
                sigMaskVal |= rpvMaskVal

            if sigMaskDataVal is None:
                sigMaskDataVal = rpvMaskDataVal
            else:
                sigMaskDataVal |= rpvMaskDataVal

        if "SYY" in evalModel:
            if sigMaskEval is None:
                sigMaskEval = syyMaskEval
            else:
                sigMaskEval |= syyMaskEval

            if sigMaskDataEval is None:
                sigMaskDataEval = syyMaskDataEval
            else:
                sigMaskDataEval |= syyMaskDataEval

        if "SYY" in valModel:
            if sigMaskVal is None:
                sigMaskVal = syyMaskVal
            else:
                sigMaskVal |= syyMaskVal

            if sigMaskDataVal is None:
                sigMaskDataVal = syyMaskDataVal
            else:
                sigMaskDataVal |= syyMaskDataVal

        if "SHH" in evalModel:
            if sigMaskEval is None:
                sigMaskEval = shhMaskEval
            else:
                sigMaskEval |= shhMaskEval

            if sigMaskDataEval is None:
                sigMaskDataEval = shhMaskDataEval
            else:
                sigMaskDataEval |= shhMaskDataEval

        if "SHH" in valModel:
            if sigMaskVal is None:
                sigMaskVal = shhMaskVal
            else:
                sigMaskVal |= shhMaskVal

            if sigMaskDataVal is None:
                sigMaskDataVal = shhMaskDataVal
            else:
                sigMaskDataVal |= shhMaskDataVal

        # Part of the training samples that were not used for training
        output_val, output_val_sg, output_val_bg = self.getOutput(self.model, valData["inputs"], valSig["inputs"], valBkg["inputs"])

        y_val_disc1,  y_val_sg_disc1,  y_val_bg_disc1  = self.getResults(output_val,   output_val_sg,  output_val_bg,  outputNum=0, columnNum=0)
        y_val_disc2,  y_val_sg_disc2,  y_val_bg_disc2  = self.getResults(output_val,   output_val_sg,  output_val_bg,  outputNum=0, columnNum=2)
        y_val_mass,   y_val_mass_sg,   y_val_mass_bg   = self.getResults(output_val,   output_val_sg,  output_val_bg,  outputNum=2, columnNum=0)

        # Separately loaded samples that can have nothing to do with the what was loaded for training
        output_train, output_eval_sg, output_eval_bg = self.getOutput(self.model, evalData["inputs"], evalSig["inputs"], evalBkg["inputs"])

        y_eval_disc1, y_eval_sg_disc1, y_eval_bg_disc1 = self.getResults(output_train, output_eval_sg, output_eval_bg, outputNum=0, columnNum=0)
        y_eval_disc2, y_eval_sg_disc2, y_eval_bg_disc2 = self.getResults(output_train, output_eval_sg, output_eval_bg, outputNum=0, columnNum=2)
        y_eval_mass,  y_eval_mass_sg,  y_eval_mass_bg  = self.getResults(output_train, output_eval_sg, output_eval_bg, outputNum=2, columnNum=0)

        nBins = 20
        nBinsReg = 100
        masses = [350., 550., 850., 1150.]

        colors = ["red", "green", "blue", "magenta", "cyan"]; labels = ["Bkg Train", "Bkg Val"]

        self.plotDisc([y_eval_mass_bg, y_val_mass_bg], colors, labels, [evalBkg["weight"], valBkg["weight"]], "mass",     'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg)
        self.plotDisc([y_eval_mass_bg, y_val_mass_bg], colors, labels, [evalBkg["weight"], valBkg["weight"]], "mass_log", 'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg, doLog=True)

        tempColors = ["black"]; tempNames = ["ttbar"]; tempMass = [y_eval_mass_bg]; tempEvents = [evalBkg["weight"]]; tempMassVal = [y_val_mass_bg]; tempEventsVal = [valBkg["weight"]]
        i = 0
        for imass in masses:
            self.plotDisc([y_eval_mass_sg[(evalSig["mass"]==imass)&sigMaskEval], y_val_mass_sg[valSig["mass"]==imass]], colors, labels, [evalSig["weight"][(evalSig["mass"]==imass)&sigMaskEval], valSig["weight"][valSig["mass"]==imass]], "mass_%d"%(imass), 'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg)

            tempColors.append(colors[i])
            tempNames.append("mass %d"%(imass))
            tempMass.append(y_eval_mass_sg[(evalSig["mass"]==imass)&sigMaskEval])
            tempEvents.append(evalSig["weight"][(evalSig["mass"]==imass)&sigMaskEval])

            tempMassVal.append(y_val_mass_sg[valSig["mass"]==imass])
            tempEventsVal.append(valSig["weight"][valSig["mass"]==imass])

            i += 1

        self.plotDisc([y_eval_bg_disc1, y_val_bg_disc1], colors, labels, [evalBkg["weight"], valBkg["weight"]], "Disc1", 'Norm Events', 'Disc. 1')
        self.plotDisc([y_eval_bg_disc2, y_val_bg_disc2], colors, labels, [evalBkg["weight"], valBkg["weight"]], "Disc2", 'Norm Events', 'Disc. 2')

        self.plotDisc(tempMass, tempColors, tempNames, tempEvents, "mass_split",     'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg)
        self.plotDisc(tempMass, tempColors, tempNames, tempEvents, "mass_split_log", 'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg, doLog=True)

        self.plotDisc(tempMassVal, tempColors, tempNames, tempEventsVal, "mass_split_val",     'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg)
        self.plotDisc(tempMassVal, tempColors, tempNames, tempEventsVal, "mass_split_val_log", 'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg, doLog=True)

        for NJets in NJetsRange:
            
            njets = float(NJets)
            if self.config["Mask"] and (int(NJets) in self.config["Mask_nJet"]): continue

            bkgNjetsMaskEval = evalBkg["njets"] == njets; sigNjetsMaskEval = evalSig["njets"] == njets
            bkgNjetsMaskVal  = valBkg["njets"]  == njets; sigNjetsMaskVal  = valSig["njets"]  == njets

            tempColors = ["black"]; tempNames = ["ttbar"]; tempMass = [y_eval_mass_bg[bkgNjetsMaskEval]]; tempEvents = [evalBkg["weight"][bkgNjetsMaskEval]]; tempMassVal = [y_val_mass_bg[bkgNjetsMaskVal]]; tempEventsVal = [valBkg["weight"][bkgNjetsMaskVal]]
            i = 0
            for imass in masses:
                if imass >= self.config["minStopMass"] and imass <= self.config["maxStopMass"]:
                    mask = "mask_m%d"%(imass)

                    tempColors.append(colors[i])
                    tempNames.append("mass %d"%(imass))
                    tempMass.append(y_eval_mass_sg[(evalSig["mass"]==imass)&sigMaskEval&sigNjetsMaskEval])
                    tempEvents.append(evalSig["weight"][(evalSig["mass"]==imass)&sigMaskEval&sigNjetsMaskEval])

                    tempMassVal.append(y_val_mass_sg[(valSig["mass"]==imass)&sigMaskVal&sigNjetsMaskVal])
                    tempEventsVal.append(valSig["weight"][(valSig["mass"]==imass)&sigMaskVal&sigNjetsMaskVal])

                    i += 1

            self.plotDisc(tempMass, tempColors, tempNames, tempEvents, "mass_split_Njets%s"%(NJets),     'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg)
            self.plotDisc(tempMass, tempColors, tempNames, tempEvents, "mass_split_Njets%s_log"%(NJets), 'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg, doLog=True)

            self.plotDisc(tempMassVal, tempColors, tempNames, tempEventsVal, "mass_split_val_Njets%s"%(NJets),     'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg)

            self.plotDisc(tempMassVal, tempColors, tempNames, tempEventsVal, "mass_split_val_Njets%s_log"%(NJets), 'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg, doLog=True)

        # Plot Acc vs Epoch
        if self.result_log != None:
            self.plotAccVsEpoch('loss', 'val_loss', 'model loss', 'loss_train_val')
            for rank in ["disco", "disc", "mass_reg"]: self.plotAccVsEpoch('%s_loss'%(rank), 'val_%s_loss'%(rank), '%s loss'%(rank), '%s_loss_train_val'%(rank))        

            self.plotAccVsEpochAll(['disc', 'mass_reg' , 'disco'], ['Combined Disc Loss', 'Mass Regression Loss', 'DisCo Loss'], '',     'train output loss',      'output_loss_train')
            self.plotAccVsEpochAll(['disc', 'mass_reg' , 'disco'], ['Combined Disc Loss', 'Mass Regression Loss', 'DisCo Loss'], 'val_', 'validation output loss', 'output_loss_val')

        # Plot disc per njet
        self.plotDiscPerNjet("_Disc1", {"Bkg": [evalBkg, y_eval_bg_disc1, evalBkg["weight"]], "Sig": [evalSig, y_eval_sg_disc1, evalSig["weight"]]}, sigMaskEval, nBins=nBins)
        self.plotDiscPerNjet("_Disc2", {"Bkg": [evalBkg, y_eval_bg_disc2, evalBkg["weight"]], "Sig": [evalSig, y_eval_sg_disc2, evalSig["weight"]]}, sigMaskEval, nBins=nBins)
        
        if not doQuickVal:
            self.plotD1VsD2SigVsBkgd(y_eval_bg_disc1, y_eval_bg_disc2, y_eval_sg_disc1[massMaskEval&sigMaskEval], y_eval_sg_disc2[massMaskEval&sigMaskEval], evalMass)
            # Make arrays for possible values to cut on for both discriminant
            # starting at a minimum of 0.5 for each
            edgeWidth = 0.02; minEdge = 0.1; maxEdge = 0.90
            c1s = np.arange(minEdge, maxEdge, edgeWidth); c2s = np.arange(minEdge, maxEdge, edgeWidth)

            # Plot 2D of the discriminants
            self.plotDisc1vsDisc2(y_eval_bg_disc1, y_eval_bg_disc2, evalBkg["weight"], -1.0, -1.0, -1.0, "BG")
            self.plotDisc1vsDisc2(y_eval_sg_disc1[massMaskEval&sigMaskEval], y_eval_sg_disc2[massMaskEval&sigMaskEval], evalSig["weight"][massMaskEval&sigMaskEval], -1.0, -1.0, -1.0, "SG", mass=evalMass)

            self.plotDisc1vsDisc2(y_val_bg_disc1, y_val_bg_disc2, valBkg["weight"], -1.0, -1.0, -1.0, "valBG")
            self.plotDisc1vsDisc2(y_val_sg_disc1[massMaskVal&sigMaskVal], y_val_sg_disc2[massMaskVal&sigMaskVal], valSig["weight"][massMaskVal&sigMaskVal], -1.0, -1.0, -1.0, "valSG", mass=valMass)

            bkgdNjets    = {"A" : [], "B" : [], "C" : [], "D" : [], "a" : [], "b" : [], "c" : [], "d" : []}; sigNjets    = {"A" : [], "B" : [], "C" : [], "D" : [], "a" : [], "b" : [], "c" : [], "d" : []}
            bkgdNjetsErr = {"A" : [], "B" : [], "C" : [], "D" : [], "a" : [], "b" : [], "c" : [], "d" : []}; sigNjetsErr = {"A" : [], "B" : [], "C" : [], "D" : [], "a" : [], "b" : [], "c" : [], "d" : []}
            bkgdNjetsAPred = {"val" : [], "err" : []}
            bkgdNjetsSign = []
            
            for NJets in NJetsRange:
           
                njets = float(NJets)
                if self.config["Mask"] and (int(NJets) in self.config["Mask_nJet"]): continue 

                dataNjetsMaskEval = evalData["njets"] == njets
                bkgNjetsMaskEval  = evalBkg["njets"]   == njets; sigNjetsMaskEval = evalSig["njets"] == njets
                bkgFullMaskEval   = bkgNjetsMaskEval; sigFullMaskEval  = sigMaskEval & massMaskEval & sigNjetsMaskEval

                dataNjetsMaskVal = valData["njets"] == njets
                bkgNjetsMaskVal  = valBkg["njets"]   == njets; sigNjetsMaskVal = valSig["njets"]==njets
                bkgFullMaskVal   = bkgNjetsMaskVal; sigFullMaskVal  = sigMaskVal & massMaskVal & sigNjetsMaskVal

                # Get number of background and signal counts for each A, B, C, D region for every possible combination of cuts on disc 1 and disc 2
                bc, sc = self.cutAndCount(c1s, c2s, y_eval_bg_disc1[bkgFullMaskEval], y_eval_bg_disc2[bkgFullMaskEval], evalBkg["weight"][bkgFullMaskEval], y_eval_sg_disc1[sigFullMaskEval], y_eval_sg_disc2[sigFullMaskEval], evalSig["weight"][sigFullMaskEval])
                c1, c2, significance, closureErr, invSigns, closeErrs, closeErrsUncs, d1edges, d2edges, sFracsA, sFracsB, sFracsC, sFracsD, sTotFracsA, sTotFracsB, sTotFracsC, sTotFracsD, bTotFracsA, bTotFracsB, bTotFracsC, bTotFracsD = self.findDiscCut4SigFrac(bc, sc)

                self.plotMetricVsBinEdges(invSigns, closeErrs, closeErrsUncs, d1edges, d2edges, float(c1), float(c2), minEdge, maxEdge, edgeWidth, int(NJets))

                self.plotClosureVsDisc(closeErrs, closeErrsUncs, d1edges, d2edges, edgeWidth/2.0, 1, int(NJets))
                self.plotClosureVsDisc(closeErrs, closeErrsUncs, d1edges, d2edges, edgeWidth/2.0, 2, int(NJets))

                self.plotBkgSigFracVsBinEdges(sFracsA, sFracsB, sFracsC, sFracsD, sTotFracsA, sTotFracsB, sTotFracsC, sTotFracsD, bTotFracsA, bTotFracsB, bTotFracsC, bTotFracsD, d1edges, d2edges, float(c1), float(c2), minEdge, maxEdge, edgeWidth, int(NJets))

                tempAveClose, tempStdClose = self.plotBinEdgeMetricComps(significance, closureErr, invSigns, closeErrs, c1s, c1, c2, int(NJets))

                bkgdNjetsSign.append(significance)

                if c1 == -1.0 or c2 == -1.0:
                    bkgdNjets["A"].append(0.0); sigNjets["A"].append(0.0)
                    bkgdNjets["B"].append(0.0); sigNjets["B"].append(0.0)
                    bkgdNjets["C"].append(0.0); sigNjets["C"].append(0.0)
                    bkgdNjets["D"].append(0.0); sigNjets["D"].append(0.0)

                    bkgdNjets["a"].append(0.0); sigNjets["a"].append(0.0)
                    bkgdNjets["b"].append(0.0); sigNjets["b"].append(0.0)
                    bkgdNjets["c"].append(0.0); sigNjets["c"].append(0.0)
                    bkgdNjets["d"].append(0.0); sigNjets["d"].append(0.0)

                    bkgdNjetsErr["A"].append(0.0); sigNjetsErr["A"].append(0.0)
                    bkgdNjetsErr["B"].append(0.0); sigNjetsErr["B"].append(0.0)
                    bkgdNjetsErr["C"].append(0.0); sigNjetsErr["C"].append(0.0)
                    bkgdNjetsErr["D"].append(0.0); sigNjetsErr["D"].append(0.0)

                    bkgdNjetsErr["a"].append(0.0); sigNjetsErr["a"].append(0.0)
                    bkgdNjetsErr["b"].append(0.0); sigNjetsErr["b"].append(0.0)
                    bkgdNjetsErr["c"].append(0.0); sigNjetsErr["c"].append(0.0)
                    bkgdNjetsErr["d"].append(0.0); sigNjetsErr["d"].append(0.0)

                    bkgdNjetsAPred["val"].append(0.0); bkgdNjetsAPred["err"].append(0.0)

                else:
                    closure, closureUnc, Apred, ApredUnc = self.simpleClosureABCD(bc["a"][c1][c2], bc["b"][c1][c2], bc["c"][c1][c2], bc["d"][c1][c2], bc["a"][c1][c2]**0.5, bc["b"][c1][c2]**0.5, bc["c"][c1][c2]**0.5, bc["d"][c1][c2]**0.5)
                    bkgdNjets["a"].append(bc["a"][c1][c2]); sigNjets["a"].append(sc["a"][c1][c2])
                    bkgdNjets["b"].append(bc["b"][c1][c2]); sigNjets["b"].append(sc["b"][c1][c2])
                    bkgdNjets["c"].append(bc["c"][c1][c2]); sigNjets["c"].append(sc["c"][c1][c2])
                    bkgdNjets["d"].append(bc["d"][c1][c2]); sigNjets["d"].append(sc["d"][c1][c2])

                    bkgdNjets["A"].append(bc["A"][c1][c2]); sigNjets["A"].append(sc["A"][c1][c2])
                    bkgdNjets["B"].append(bc["B"][c1][c2]); sigNjets["B"].append(sc["B"][c1][c2])
                    bkgdNjets["C"].append(bc["C"][c1][c2]); sigNjets["C"].append(sc["C"][c1][c2])
                    bkgdNjets["D"].append(bc["D"][c1][c2]); sigNjets["D"].append(sc["D"][c1][c2])

                    bkgdNjetsErr["a"].append(bc["a"][c1][c2]**0.5); sigNjetsErr["a"].append(sc["a"][c1][c2]**0.5)
                    bkgdNjetsErr["b"].append(bc["b"][c1][c2]**0.5); sigNjetsErr["b"].append(sc["b"][c1][c2]**0.5)
                    bkgdNjetsErr["c"].append(bc["c"][c1][c2]**0.5); sigNjetsErr["c"].append(sc["c"][c1][c2]**0.5)
                    bkgdNjetsErr["d"].append(bc["d"][c1][c2]**0.5); sigNjetsErr["d"].append(sc["d"][c1][c2]**0.5)

                    bkgdNjetsErr["A"].append(bc["A"][c1][c2]**0.5); sigNjetsErr["A"].append(sc["A"][c1][c2]**0.5)
                    bkgdNjetsErr["B"].append(bc["B"][c1][c2]**0.5); sigNjetsErr["B"].append(sc["B"][c1][c2]**0.5)
                    bkgdNjetsErr["C"].append(bc["C"][c1][c2]**0.5); sigNjetsErr["C"].append(sc["C"][c1][c2]**0.5)
                    bkgdNjetsErr["D"].append(bc["D"][c1][c2]**0.5); sigNjetsErr["D"].append(sc["D"][c1][c2]**0.5)

                    bkgdNjetsAPred["val"].append(Apred); bkgdNjetsAPred["err"].append(ApredUnc)

                self.config["c1_nJet_%s"%(("%s"%(NJets)).zfill(2))] = c1
                self.config["c2_nJet_%s"%(("%s"%(NJets)).zfill(2))] = c2

                # Avoid completely masked Njets bins that makes below plotting
                # highly unsafe
                if not any(bkgNjetsMaskEval) or not any(sigNjetsMaskEval): continue

                self.plotD1VsD2SigVsBkgd(y_eval_bg_disc1[bkgFullMaskEval], y_eval_bg_disc2[bkgFullMaskEval], y_eval_sg_disc1[sigFullMaskEval], y_eval_sg_disc2[sigFullMaskEval], evalMass, NJets)

                # Plot each discriminant for sig and background while making cut on other disc
                self.plotDiscWithCut(float(c2), y_eval_bg_disc1[bkgFullMaskEval], y_eval_bg_disc2[bkgFullMaskEval], evalBkg["weight"][bkgFullMaskEval], y_eval_sg_disc1[sigFullMaskEval], y_eval_sg_disc2[sigFullMaskEval], evalSig["weight"][sigFullMaskEval], "1", "2", mass=evalMass, Njets=NJets, bins=nBins)
                self.plotDiscWithCut(float(c1), y_eval_bg_disc2[bkgFullMaskEval], y_eval_bg_disc1[bkgFullMaskEval], evalBkg["weight"][bkgFullMaskEval], y_eval_sg_disc2[sigFullMaskEval], y_eval_sg_disc1[sigFullMaskEval], evalSig["weight"][sigFullMaskEval], "2", "1", mass=evalMass, Njets=NJets, bins=nBins)
            
                self.plotDiscWithCutCompare(float(c2), y_eval_bg_disc1[bkgFullMaskEval], y_eval_bg_disc2[bkgFullMaskEval], evalBkg["weight"][bkgFullMaskEval], "1", "2", "BG", mass="", Njets=-1, bins=10)
                self.plotDiscWithCutCompare(float(c2), y_eval_sg_disc1[sigFullMaskEval], y_eval_sg_disc2[sigFullMaskEval], evalSig["weight"][sigFullMaskEval], "1", "2", "SG", mass=evalMass, Njets=NJets, bins=10)
            
                self.plotDiscWithCutCompare(float(c1), y_eval_bg_disc2[bkgFullMaskEval], y_eval_bg_disc1[bkgFullMaskEval], evalBkg["weight"][bkgFullMaskEval], "2", "1", "BG", mass="", Njets=-1, bins=10)
                self.plotDiscWithCutCompare(float(c1), y_eval_sg_disc2[sigFullMaskEval], y_eval_sg_disc1[sigFullMaskEval], evalSig["weight"][sigFullMaskEval], "2", "1", "SG", mass=evalMass, Njets=NJets, bins=10)
            
                # Plot 2D of the discriminants
                bkgdCorr = self.plotDisc1vsDisc2(y_eval_bg_disc1[bkgFullMaskEval], y_eval_bg_disc2[bkgFullMaskEval], evalBkg["weight"][bkgFullMaskEval], float(c1), float(c2), significance, "BG", mass="",   Njets=NJets)
                self.plotDisc1vsDisc2(y_eval_sg_disc1[sigFullMaskEval], y_eval_sg_disc2[sigFullMaskEval], evalSig["weight"][sigFullMaskEval], float(c1), float(c2), significance, "SG", mass=evalMass, Njets=NJets)
                self.metric["bkgdCorr_nJet_%s"%(NJets)] = abs(bkgdCorr) 

                self.plotDisc1vsDisc2(y_val_bg_disc1[bkgFullMaskVal], y_val_bg_disc2[bkgFullMaskVal], valBkg["weight"][bkgFullMaskVal], float(c1), float(c2), significance, "valBG", mass="",   Njets=NJets)
                self.plotDisc1vsDisc2(y_val_sg_disc1[sigFullMaskVal], y_val_sg_disc2[sigFullMaskVal], valSig["weight"][sigFullMaskVal], float(c1), float(c2), significance, "valSG", mass=valMass, Njets=NJets)

            signA = self.plotNjets(bkgdNjets["A"], bkgdNjetsErr["A"], sigNjets["A"], sigNjetsErr["A"], "A")
            signB = self.plotNjets(bkgdNjets["B"], bkgdNjetsErr["B"], sigNjets["B"], sigNjetsErr["B"], "B")
            signC = self.plotNjets(bkgdNjets["C"], bkgdNjetsErr["C"], sigNjets["C"], sigNjetsErr["C"], "C")
            signD = self.plotNjets(bkgdNjets["D"], bkgdNjetsErr["D"], sigNjets["D"], sigNjetsErr["D"], "D")

            totalChi2, wtotalChi2, ndof = self.plotNjetsClosure(bkgdNjets["a"], bkgdNjetsErr["a"], bkgdNjetsAPred["val"], bkgdNjetsAPred["err"], bkgdNjetsSign)

            self.config["Achi2"] = totalChi2
            if ndof != 0:
                self.config["Achi2ndof"] = float(totalChi2/ndof)
                self.config["Awchi2ndof"] = float(wtotalChi2/ndof)
            else:
                self.config["Achi2ndof"] = 9999.0
            self.config["Asignificance"] = float(signA)
            self.config["Bsignificance"] = float(signB)
            self.config["Csignificance"] = float(signC)
            self.config["Dsignificance"] = float(signD)
            self.config["TotalSignificance"] = (signA**2.0 + signB**2.0 + signC**2.0 + signD**2.0)**0.5

            print("A SIGNIFICANCE: %3.2f"%(signA))
            print("B SIGNIFICANCE: %3.2f"%(signB))
            print("C SIGNIFICANCE: %3.2f"%(signC))
            print("D SIGNIFICANCE: %3.2f"%(signD))
            print("TOTAL SIGNIFICANCE: %3.2f"%((signA**2.0 + signB**2.0 + signC**2.0 + signD**2.0)**0.5))
  
            if    self.config["TotalSignificance"] > 0.0: self.metric["InvTotalSignificance"] = 1.0/self.config["TotalSignificance"]
            else: self.metric["InvTotalSignificance"] = 999.0

        # Plot validation roc curve
        fpr_val_disc1, tpr_val_disc1, thresholds_val_disc1    = roc_curve(valData["label"][:,0][sigMaskDataVal],   y_val_disc1[sigMaskDataVal],   sample_weight=valData["weight"][sigMaskDataVal])
        fpr_val_disc2, tpr_val_disc2, thresholds_val_disc2    = roc_curve(valData["label"][:,0][sigMaskDataVal],   y_val_disc2[sigMaskDataVal],   sample_weight=valData["weight"][sigMaskDataVal])
        fpr_eval_disc1, tpr_eval_disc1, thresholds_eval_disc1 = roc_curve(evalData["label"][:,0][sigMaskDataEval], y_eval_disc1[sigMaskDataEval], sample_weight=evalData["weight"][sigMaskDataEval])
        fpr_eval_disc2, tpr_eval_disc2, thresholds_eval_disc2 = roc_curve(evalData["label"][:,0][sigMaskDataEval], y_eval_disc2[sigMaskDataEval], sample_weight=evalData["weight"][sigMaskDataEval])
        auc_val_disc1  = roc_auc_score(valData["label"][:,0][sigMaskDataVal],   y_val_disc1[sigMaskDataVal])
        auc_val_disc2  = roc_auc_score(valData["label"][:,0][sigMaskDataVal],   y_val_disc2[sigMaskDataVal])
        auc_eval_disc1 = roc_auc_score(evalData["label"][:,0][sigMaskDataEval], y_eval_disc1[sigMaskDataEval])
        auc_eval_disc2 = roc_auc_score(evalData["label"][:,0][sigMaskDataEval], y_eval_disc2[sigMaskDataEval])

        # Define metrics for the training
        self.metric["OverTrain_Disc1"]   = abs(auc_val_disc1 - auc_eval_disc1)
        self.metric["OverTrain_Disc2"]   = abs(auc_val_disc2 - auc_eval_disc2)
        self.metric["Performance_Disc1"] = abs(1 - auc_eval_disc1)
        self.metric["Performance_Disc2"] = abs(1 - auc_eval_disc2)
       
        # Plot some ROC curves
        self.plotROC(None, None, "_Disc1", None, None, None, None, fpr_eval_disc1, fpr_val_disc1, tpr_eval_disc1, tpr_val_disc1, auc_eval_disc1, auc_val_disc1)
        self.plotROC(None, None, "_Disc2", None, None, None, None, fpr_eval_disc2, fpr_val_disc2, tpr_eval_disc2, tpr_val_disc2, auc_eval_disc2, auc_val_disc2)
        self.plotROC(sigMaskDataEval, sigMaskDataVal, "_"+self.config["bkgd"][0]+"_nJet_disc1", y_eval_disc1, y_val_disc1, evalData, valData)
        self.plotROC(sigMaskDataEval, sigMaskDataVal, "_"+self.config["bkgd"][0]+"_nJet_disc2", y_eval_disc2, y_val_disc2, evalData, valData)
        
        # Plot validation precision recall
        precision_val_disc1,  recall_val_disc1,  _ = precision_recall_curve(valData["label"][:,0][sigMaskDataVal],   y_val_disc1[sigMaskDataVal],   sample_weight=valData["weight"][sigMaskDataVal])
        precision_eval_disc1, recall_eval_disc1, _ = precision_recall_curve(evalData["label"][:,0][sigMaskDataEval], y_eval_disc1[sigMaskDataEval], sample_weight=evalData["weight"][sigMaskDataEval])
        ap_val_disc1  = average_precision_score(valData["label"][:,0],  y_val_disc1,  sample_weight=valData["weight"])
        ap_eval_disc1 = average_precision_score(evalData["label"][:,0], y_eval_disc1, sample_weight=evalData["weight"])
        
        self.plotPandR(precision_val_disc1, recall_val_disc1, precision_eval_disc1, recall_eval_disc1, ap_val_disc1, ap_eval_disc1)
        
        for key in self.metric:
            print(key, self.metric[key])

        self.config["metric"] = self.metric
        with open(self.config["outputDir"]+"/config.json",'w') as configFile:
            json.dump(self.config, configFile, indent=4, sort_keys=True)

        return self.metric
