from DataGetter import get_data, getSamplesToRun
import numpy as np
import os

# Little incantation to display trying to X display
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.ROOT,hep.style.CMS]) # For now ROOT defaults to CMS
plt.style.use({'legend.frameon':False,'legend.fontsize':16,'legend.edgecolor':'black'})
from matplotlib.colors import LogNorm
import matplotlib.lines as ml
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
import json
from Correlation import Correlation as cor

plt.rcParams['pdf.fonttype'] = 42

class Validation:

    def __init__(self, model, config, trainData, trainSg, trainBg, result_log=None):
        self.model = model
        self.config = config
        self.Data = trainData
        self.Sg = trainSg
        self.Bg = trainBg
        self.result_log = result_log
        self.metric = {}
        self.doLog = False

    def __del__(self):
        del self.model
        del self.config
        del self.Data
        del self.Sg
        del self.Bg
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
        
    def plot2DVar(self, name, binxl, binxh, numbin, xIn, yIn, nbiny):
        fig = plt.figure()
        h, xedges, yedges, image = plt.hist2d(xIn, yIn, bins=[numbin, nbiny], range=[[binxl, binxh], [0, 1]], cmap=plt.cm.jet, cmin = 1)
        plt.colorbar()
        hep.cms.label(data=True, paper=False, year=self.config["year"])
    
        bin_centersx = 0.5 * (xedges[:-1] + xedges[1:])
        bin_centersy = 0.5 * (yedges[:-1] + yedges[1:])
        y = []
        ye = []
        for i in range(h.shape[0]):
            ynum = 0
            ynum2 = 0
            ydom = 0
            for j in range(len(h[i])):
                ynum += h[i][j] * bin_centersy[j]
                ynum2 += h[i][j] * (bin_centersy[j]**2)
                ydom += h[i][j]        
            yavg = ynum / ydom if ydom != 0 else -1
            yavg2 = ynum2 / ydom if ydom != 0 else -1
            sigma = np.sqrt(yavg2 - (yavg**2)) if ydom != 0 else 0
            y.append(yavg)
            ye.append(sigma)
            
        xerr = 0.5*(xedges[1]-xedges[0])
        plt.errorbar(bin_centersx, y, xerr=xerr, yerr=ye, fmt='o', color='xkcd:red')
        fig.savefig(self.config["outputDir"]+"/"+name+"_discriminator.pdf", dpi=fig.dpi) 

        plt.close(fig)

    def getAUC(self, fpr, tpr):
        try:
            return auc(fpr, tpr)
        except:
            print("Roc curve didn't work?????")
            print(fpr)
            print(tpr)
            return -1

    def getOutput(self, model, data, Sg, Bg):
        return model.predict(data), model.predict(Sg), model.predict(Bg)

    def getResults(self, output, output_Sg, output_Bg, outputNum=0, columnNum=0):
        return output[outputNum][:,columnNum].ravel(), output_Sg[outputNum][:,columnNum].ravel(), output_Bg[outputNum][:,columnNum].ravel()
        #return output[:,columnNum].ravel(), output_Sg[:,columnNum].ravel(), output_Bg[:,columnNum].ravel()

    # Plot a set of 1D hists together, where the hists, colors, labels, weights
    # are provided as a list argument.
    def plotDisc(self, hists, colors, labels, weights, name, xlab, ylab, bins=100, arange=(0,1), doLog=False):
        # Plot predicted mass
        fig, ax = plt.subplots(figsize=(10, 10))
        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel(xlab); ax.set_xlabel(ylab)

        for i in range(0, len(hists)): 
            mean = round(np.average(hists[i], weights=weights[i][:,0]), 2)
            plt.hist(hists[i], bins=bins, range=arange, color="xkcd:"+colors[i], alpha=0.9, histtype='step', lw=2, label=labels[i]+" mean="+str(mean), density=True, log=doLog, weights=weights[i])

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

    def plotAccVsEpochAll(self, h2, h3, h4, h5, n2, n3, n4, n5, val, title, name):
        fig = plt.figure()
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        plt.plot(self.result_log.history["%s%s_loss"%(val,h2)])
        plt.plot(self.result_log.history["%s%s_output_loss"%(val,h3)])
        plt.plot(self.result_log.history["%s%s_output_loss"%(val,h4)])
        plt.plot(self.result_log.history["%s%s_loss"%(val,h5)])
        plt.title(title, pad=45.0)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend([n2, n3, n4, n5], loc='best')
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
                    if sample == "Sg": mask = sigMask
                    yt = y_train_Sp[trainSample[key]&mask]                
                    wt = weights[trainSample[key]&mask]
                    if yt.size != 0 and wt.size != 0:
                        plt.hist(yt, bins, alpha=0.9, histtype='step', lw=2, label=sample+" Train "+key, density=True, log=self.doLog, weights=wt)
            plt.legend(loc='best')
            fig.savefig(self.config["outputDir"]+"/nJet_"+sample+tag+".pdf", dpi=fig.dpi)
            plt.close(fig)

    def plotROC(self, tag="", y_Train=None, y_Val=None, trainData=None, valData=None, xVal=None, yVal=None, xTrain=None, yTrain=None, valLab=None, trainLab=None):

        extra = None
        if "disc1" in tag or "Disc1" in tag: extra = "disc1"
        else:                                extra = "disc2"

        if extra not in self.config: self.config[extra] = {"train_auc" : {}, "val_auc" : {}}

        fig = plt.figure()
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve', pad=45.0)

        if y_Train is None:
            plt.plot(xVal, yVal, color='xkcd:black', label='Val (area = {:.3f})'.format(valLab))
            plt.plot(xTrain, yTrain, color='xkcd:red', label='Train (area = {:.3f})'.format(trainLab))
            self.config[extra]["train_auc"]["total"] = trainLab
            self.config[extra]["val_auc"]["total"] = valLab

        else:

            for key in sorted(trainData.keys()):
                if key.find("mask_nJet") != -1:
                    labels = trainData["labels"][trainData[key]]
                    valLabels = valData["labels"][valData[key]]
                    weights = trainData["Weight"][trainData[key]][:,0]
                    y = y_Train[trainData[key]]
                    yVal = y_Val[valData[key]]
                    if len(y)==0 or len(yVal)==0:
                        continue
                    fpr_Train, tpr_Train, thresholds_Train = roc_curve(labels[:,0], y, sample_weight=weights)
                    auc_Train = roc_auc_score(labels[:,0], y)    
                    auc_Val   = roc_auc_score(valLabels[:,0], yVal)
                    plt.plot(fpr_Train, tpr_Train, label="Train "+key+" (area = {:.3f})".format(auc_Train))
                    self.config[extra]["train_auc"][key.replace("mask_","")] = auc_Train
                    self.config[extra]["val_auc"][key.replace("mask_","")] = auc_Val

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

    # Just plot the 2D for either background or signal
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
        plt.text(0.05, 0.95, r"$\bf{Significance}$ = %.3f"%(significance), fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/2D_%s%s_Disc1VsDisc2.pdf"%(tag,mass), dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/2D_%s%s_Njets%d_Disc1VsDisc2.pdf"%(tag,mass,Njets), dpi=fig.dpi)

        plt.close(fig)

        return corr

    def cutAndCount(self, c1s, c2s, b1, b2, bw, s1, s2, sw, cdiff = 0.2):
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
            
                # Keep scan of c1,c2 close to the diagnonal
                # Default c1 and c2 to be within 20% of one another
                if abs(1.0 - c1/c2) > cdiff: continue

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

    def findDiscCut4SigFrac(self, bcts, scts, minFracB = 0.05, minFracS = 0.1):
        # Now calculate signal fraction and significance 
        # Pick c1 and c2 that give 30% sig fraction and maximizes significance
        significance = 0.0; finalc1 = -1.0; finalc2 = -1.0; 
        closureErr = 0.0; metric = 999.0
        sFracA = 0.0; sbFracA = 0.0
        sFracB = 0.0; sbFracB = 0.0
        sFracC = 0.0; sbFracC = 0.0
        sFracD = 0.0; sbFracD = 0.0
        invSigns = []; closeErrs = []
        for c1k, c2s in bcts["A"].items():
            for c2k, temp in c2s.items():

                bA = bcts["A"][c1k][c2k]; bB = bcts["B"][c1k][c2k]; bC = bcts["C"][c1k][c2k]; bD = bcts["D"][c1k][c2k]
                sA = scts["A"][c1k][c2k]; sB = scts["B"][c1k][c2k]; sC = scts["C"][c1k][c2k]; sD = scts["D"][c1k][c2k]

                sTotal = sA + sB + sC + sD
                bTotal = bA + bB + bC + bD

                tempsfracA = -1.0; tempsbfracA = -1.0
                tempsfracB = -1.0; tempsbfracB = -1.0
                tempsfracC = -1.0; tempsbfracC = -1.0
                tempsfracD = -1.0; tempsbfracD = -1.0
                if bA + sA > 0.0: tempsbfracA = sA / (sA + bA)
                if bB + sB > 0.0: tempsbfracB = sB / (sB + bB)
                if bC + sC > 0.0: tempsbfracC = sC / (sC + bC)
                if bD + sD > 0.0: tempsbfracD = sD / (sD + bD)

                tempbfracA = bA / bTotal
                tempsfracA = sA / sTotal

                tempbfracB = bB / bTotal
                tempsfracB = sB / sTotal

                tempbfracC = bC / bTotal
                tempsfracC = sC / sTotal

                tempbfracD = bD / bTotal
                tempsfracD = sD / sTotal

                # Minimum signal fraction requirement
                if tempbfracA > minFracB and \
                   tempbfracB > minFracB and \
                   tempbfracC > minFracB and \
                   tempbfracD > minFracB:

                    tempsignificance = 0.0; tempclosureerr = 999.0; tempmetric = 999.0
                    if bA > 0.0: tempsignificance += (sA / (bA + (0.3*bA)**2.0)**0.5)**2.0
                    if bB > 0.0: tempsignificance += (sB / (bB + (0.3*bB)**2.0)**0.5)**2.0
                    if bC > 0.0: tempsignificance += (sC / (bC + (0.3*bC)**2.0)**0.5)**2.0
                    if bD > 0.0: tempsignificance += (sD / (bD + (0.3*bD)**2.0)**0.5)**2.0

                    if bD > 0.0 and bA > 0.0: tempclosureerr = abs(1.0 - (bB * bC) / (bA * bD))

                    tempsignificance = tempsignificance**0.5

                    if tempsignificance > 0.0:
                        tempmetric = tempclosureerr**2.0 + (1.0 / tempsignificance)**2.0

                        invSigns.append(1.0 / tempsignificance)
                        closeErrs.append(abs(tempclosureerr))
 
                    if tempmetric < metric:
                        sFracA = tempsfracA
                        sbFracA = tempsbfracA
                        sFracB = tempsfracB
                        sbFracB = tempsbfracB
                        sFracC = tempsfracC
                        sbFracC = tempsbfracC
                        sFracD = tempsfracD
                        sbFracD = tempsbfracD

                        finalc1 = c1k; finalc2 = c2k
                        metric = tempmetric
                        significance = tempsignificance
                        closureErr = tempclosureerr
                
        return finalc1, finalc2, significance, closureErr, invSigns, closeErrs, sFracA, sbFracA, sFracB, sbFracB, sFracC, sbFracC, sFracD, sbFracD

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

    def plotBinEdgeMetricComps(self, finalSign, finalClosureErr, invSign, closeErr, edges, Njets = -1):

        fig = plt.figure()
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        ax = plt.gca()
        plt.scatter(invSign, closeErr, color='xkcd:black', marker="o", label="1 - Pred./Obs. vs 1 / Significance")

        if finalSign != 0.0: 
            plt.scatter([1.0/finalSign], [finalClosureErr], color='xkcd:red', marker="o", label="Chosen Solution")
        plt.xlabel('1 / Significance')
        plt.ylabel('1 - Pred./Obs.')
        plt.legend(loc='best')
        #plt.text(0.05, 0.94, r"$\bf{ABCD\;Closure\;Error}$ = %.2f"%(finalClosureErr), transform=ax.transAxes, fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))
        #plt.text(0.05, 0.84, r"$\bf{Significance}$ = %.2f"%(finalSign), transform=ax.transAxes, fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))

        plt.ylim(bottom=0)
        plt.xlim(left=0)

        plt.gca().invert_yaxis()

        plt.text(0.65, 0.84, r"$%.2f < \bf{Disc. 1}$ < %.2f"%(edges[0],edges[-1]), transform=ax.transAxes, fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))
        plt.text(0.65, 0.77, r"$%.2f < \bf{Disc. 2}$ < %.2f"%(edges[0],edges[-1]), transform=ax.transAxes, fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))

        if Njets == -1:
            fig.savefig(self.config["outputDir"]+"/InvSign_vs_CloseErr.pdf", dpi=fig.dpi)        
        else:
            fig.savefig(self.config["outputDir"]+"/InvSign_vs_CloseErr_Njets%d.pdf"%(Njets), dpi=fig.dpi)        

        plt.close(fig)

        return np.average(closeErr), np.std(closeErr)

    # Member function to plot input variable for events in specific region 
    def plotInputVarMassCut(self, m, dm, sm, d, s, sigMask, massMask, var, varLabel, process, mass = "", Njets=-1, bins=100, arange=(0,1)):
    
        iVar = self.config["allVars"].index(var)
        maskBkgRegGT = np.ma.masked_where(dm>m, dm).mask; maskBkgRegLT = ~maskBkgRegGT
        maskSigRegGT = np.ma.masked_where(sm>m, sm).mask; maskSigRegLT = ~maskSigRegGT

        dw = d["Weight"][:,0]; sw = s["Weight"][:,0]
        v  = d["data"][:,iVar]; vS = s["data"][:,iVar]
        vGT = v[maskBkgRegGT]; vSGT = vS[sigMask&massMask&maskSigRegGT]
        vLT = v[maskBkgRegLT]; vSLT = vS[sigMask&massMask&maskSigRegLT]

        binxl = np.amin(v)
        binxh = np.amax(v)

        dwLT = dw[maskBkgRegLT]; dwGT = dw[maskBkgRegGT]
        swLT = sw[sigMask&massMask&maskSigRegLT]; swGT = sw[sigMask&massMask&maskSigRegGT]

        dw2LT = np.square(dwLT); dw2GT = np.square(dwGT)
        sw2LT = np.square(swLT); sw2GT = np.square(swGT)

        dwLTBinned,  binEdges = np.histogram(vLT, bins=bins, range=(binxl,binxh), weights=dwLT)
        dwGTBinned,  binEdges = np.histogram(vGT, bins=bins, range=(binxl,binxh), weights=dwGT)
        swLTBinned,  binEdges = np.histogram(vSLT, bins=bins, range=(binxl,binxh), weights=swLT)
        swGTBinned,  binEdges = np.histogram(vSGT, bins=bins, range=(binxl,binxh), weights=swGT)

        dw2LTBinned,  binEdges = np.histogram(vLT, bins=bins, range=(binxl,binxh), weights=dw2LT)
        dw2GTBinned,  binEdges = np.histogram(vGT, bins=bins, range=(binxl,binxh), weights=dw2GT)
        sw2LTBinned,  binEdges = np.histogram(vSLT, bins=bins, range=(binxl,binxh), weights=sw2LT)
        sw2GTBinned,  binEdges = np.histogram(vSGT, bins=bins, range=(binxl,binxh), weights=sw2GT)

        if len(dw2GTBinned) == 0: dw2GTBinned = np.zeros(bins)
        if len(sw2GTBinned) == 0: sw2GTBinned = np.zeros(bins)
        if len(dw2LTBinned) == 0: dw2LTBinned = np.zeros(bins)
        if len(sw2LTBinned) == 0: sw2LTBinned = np.zeros(bins)

        if not np.any(dw2GTBinned): dw2GTBinned += 10e-2
        if not np.any(sw2GTBinned): sw2GTBinned += 10e-2
        if not np.any(dw2LTBinned): dw2LTBinned += 10e-2
        if not np.any(sw2LTBinned): sw2LTBinned += 10e-2

        if len(dwGTBinned) == 0: dwGTBinned = np.zeros(bins)
        if len(swGTBinned) == 0: swGTBinned = np.zeros(bins)
        if len(dwLTBinned) == 0: dwLTBinned = np.zeros(bins)
        if len(swLTBinned) == 0: swLTBinned = np.zeros(bins)

        if not np.any(dwGTBinned): dwGTBinned += 10e-2
        if not np.any(swGTBinned): swGTBinned += 10e-2
        if not np.any(dwLTBinned): dwLTBinned += 10e-2
        if not np.any(swLTBinned): swLTBinned += 10e-2

        fig = plt.figure()
        ax = plt.gca()
        ax = hep.histplot(h=dwGTBinned, bins=binEdges, density=True, histtype="step", label="Background (mass > %d)"%(m), alpha=0.9, lw=2, ax=ax)
        ax = hep.histplot(h=swGTBinned, bins=binEdges, density=True, histtype="step", label="Signal (mass > %d)"%(m), alpha=0.9, lw=2, ax=ax)
        ax = hep.histplot(h=dwLTBinned, bins=binEdges, density=True, histtype="step", label="Background (mass < %d)"%(m), alpha=0.9, lw=2, ax=ax)
        ax = hep.histplot(h=swLTBinned, bins=binEdges, density=True, histtype="step", label="Signal (mass < %d)"%(m), alpha=0.9, lw=2, ax=ax)

        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel('A.U.'); ax.set_xlabel(varLabel)

        # Stupid nonsense to remove duplicate entries in legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[0:5], labels[0:5], loc=1, frameon=False)

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/%s_%s%s_MassRegCut.pdf"%(var, process, mass))
        else:           fig.savefig(self.config["outputDir"]+"/%s_%s%s_Njets%d_MassRegCut.pdf"%(var, process, mass, Njets))

        plt.close(fig)

    # Member function to plot input variable for events in specific region 
    def plotInputVar(self, c1, c2, d1, d2, d, s, sigMask, massMask, var, varLabel, process, mass = "", Njets=-1, bins=100, arange=(0,1)):
    
        iVar = self.config["allVars"].index(var)
        mask1GT = np.ma.masked_where(d1>c1, d1).mask; mask1LT = ~mask1GT
        mask2GT = np.ma.masked_where(d2>c2, d2).mask; mask2LT = ~mask2GT

        maskA = mask1GT&mask2GT
        maskB = mask1LT&mask2GT
        maskC = mask1LT&mask2LT
        maskD = mask1GT&mask2LT

        dw = d["Weight"][:,0]; sw = s["Weight"][:,0][sigMask&massMask]
        v  = d["data"][:,iVar]; vS = s["data"][:,iVar][sigMask&massMask]
        vA = d["data"][:,iVar][maskA]; vB = d["data"][:,iVar][maskB]
        vC = d["data"][:,iVar][maskC]; vD = d["data"][:,iVar][maskD]

        binxl = np.amin(v)
        binxh = np.amax(v)

        dwA = dw[maskA]; dwB = dw[maskB]
        dwC = dw[maskC]; dwD = dw[maskD]

        dwA2 = np.square(dwA); dwB2 = np.square(dwB)
        dwC2 = np.square(dwC); dwD2 = np.square(dwD)
        sw2  = np.square(sw)

        dwABinned, binEdges = np.histogram(vA, bins=bins, range=(binxl,binxh), weights=dwA)
        dwBBinned, binEdges = np.histogram(vB, bins=bins, range=(binxl,binxh), weights=dwB)
        dwCBinned, binEdges = np.histogram(vC, bins=bins, range=(binxl,binxh), weights=dwC)
        dwDBinned, binEdges = np.histogram(vD, bins=bins, range=(binxl,binxh), weights=dwD)
        swBinned,  binEdges = np.histogram(vS, bins=bins, range=(binxl,binxh), weights=sw)

        dwA2Binned, binEdges = np.histogram(vA, bins=bins, range=(binxl,binxh), weights=dwA2)
        dwB2Binned, binEdges = np.histogram(vB, bins=bins, range=(binxl,binxh), weights=dwB2)
        dwC2Binned, binEdges = np.histogram(vC, bins=bins, range=(binxl,binxh), weights=dwC2)
        dwD2Binned, binEdges = np.histogram(vD, bins=bins, range=(binxl,binxh), weights=dwD2)
        sw2Binned,  binEdges = np.histogram(vS, bins=bins, range=(binxl,binxh), weights=sw2)

        if len(dwA2Binned) == 0: dwA2Binned = np.zeros(bins)
        if len(dwB2Binned) == 0: dwB2Binned = np.zeros(bins)
        if len(dwC2Binned) == 0: dwC2Binned = np.zeros(bins)
        if len(dwD2Binned) == 0: dwD2Binned = np.zeros(bins)
        if len(sw2Binned)  == 0: sw2Binned  = np.zeros(bins)

        if not np.any(dwA2Binned): dwA2Binned += 10e-2
        if not np.any(dwB2Binned): dwB2Binned += 10e-2
        if not np.any(dwC2Binned): dwC2Binned += 10e-2
        if not np.any(dwD2Binned): dwD2Binned += 10e-2
        if not np.any(sw2Binned):  dwD2Binned += 10e-2

        if len(dwABinned) == 0: dwABinned = np.zeros(bins)
        if len(dwBBinned) == 0: dwBBinned = np.zeros(bins)
        if len(dwCBinned) == 0: dwCBinned = np.zeros(bins)
        if len(dwDBinned) == 0: dwDBinned = np.zeros(bins)
        if len(swBinned)  == 0: swBinned  = np.zeros(bins)

        if not np.any(dwABinned): dwABinned += 10e-2
        if not np.any(dwBBinned): dwBBinned += 10e-2
        if not np.any(dwCBinned): dwCBinned += 10e-2
        if not np.any(dwDBinned): dwDBinned += 10e-2
        if not np.any(swBinned):  dwDBinned += 10e-2

        fig = plt.figure()
        ax = plt.gca()
        ax = hep.histplot(h=dwABinned, bins=binEdges, density=True, histtype="step", label="Disc. 1 > %3.2f, Disc. 2 > %3.2f"%(c1, c2), alpha=0.9, lw=2, ax=ax)
        ax = hep.histplot(h=dwBBinned, bins=binEdges, density=True, histtype="step", label="Disc. 1 < %3.2f, Disc. 2 > %3.2f"%(c1, c2), alpha=0.9, lw=2, ax=ax)
        ax = hep.histplot(h=dwCBinned, bins=binEdges, density=True, histtype="step", label="Disc. 1 < %3.2f, Disc. 2 < %3.2f"%(c1, c2), alpha=0.9, lw=2, ax=ax)
        ax = hep.histplot(h=dwDBinned, bins=binEdges, density=True, histtype="step", label="Disc. 1 > %3.2f, Disc. 2 < %3.2f"%(c1, c2), alpha=0.9, lw=2, ax=ax)
        ax = hep.histplot(h=swBinned,  bins=binEdges, density=True, histtype="step", label="Signal", alpha=0.9, lw=2, ax=ax)

        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel('A.U.'); ax.set_xlabel(varLabel)

        # Stupid nonsense to remove duplicate entries in legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[0:5], labels[0:5], loc=1, frameon=False)

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/%s_%s%s_Compare_Shapes.pdf"%(var, process, mass))
        else:           fig.savefig(self.config["outputDir"]+"/%s_%s%s_Njets%d_Compare_Shapes.pdf"%(var, process, mass, Njets))

        plt.close(fig)

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

    def plotAveClosureNjets(self, aves, stds):

        binCenters = []
        xErr       = []

        for Njets in range(0, len(aves)):
        
            binCenters.append(Njets+self.config["minNJetBin"])
            xErr.append(0.5)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
       
        ax.errorbar(binCenters, aves, yerr=stds, label="Fractional Closure",  xerr=xErr, fmt='', color="red",   lw=0, elinewidth=2, marker="o", markerfacecolor="red")
        
        lowerNjets = self.config["minNJetBin"] 
        if self.config["minNJetBin"] == int(self.config["Mask_nJet"]) and self.config["Mask"]:
            lowerNjets = self.config["minNJetBin"] + 1

        ax.set_xlim([lowerNjets-0.5,self.config["maxNJetBin"]+0.5])
        
        ax.axhline(y=0.0, color="black", linestyle="dashed", lw=2)
        ax.grid(color="black", which="both", axis="y")

        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        
        ax.set_xlabel('Number of jets')
        ax.set_ylabel('|1 - Pred./Obs.|', fontsize="small")
        ax.legend(loc='best')
        
        ax.set_ylim([-0.4, 1.6])
        
        fig.savefig(self.config["outputDir"]+"/Njets_Closure_Robustness.pdf")
        
        plt.close(fig)

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
        i = 0; Njets = self.config["minNJetBin"]
        while i < len(bkgd):

            Njets += i

            if Njets == int(self.config["Mask_nJet"]) and self.config["Mask"]:
                Njets += 1
        
            if bkgdUnc[i] != 0.0:
                binCenters.append(Njets)
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

            i += 1
        
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
        ax1.text(0.05, 0.1, "$\chi^2$ / ndof = %3.2f"%(totalChi2/float(ndof)), horizontalalignment="left", verticalalignment="center", transform=ax1.transAxes, fontsize=18)
        ax1.text(0.05, 0.20, "$\chi^2$ (weighted) / ndof = %3.2f"%(wtotalChi2/float(ndof)), horizontalalignment="left", verticalalignment="center", transform=ax1.transAxes, fontsize=18)
       
        ax1.errorbar(binCenters, pred, yerr=predUnc, label="Observed",  xerr=xErr, fmt='', color="red",   lw=0, elinewidth=2, marker="o", markerfacecolor="red")
        ax1.errorbar(binCenters, obs,  yerr=unc,     label="Predicted", xerr=xErr, fmt='', color="black", lw=0, elinewidth=2, marker="o", markerfacecolor="black")

        lowerNjets = self.config["minNJetBin"] 
        if self.config["minNJetBin"] == int(self.config["Mask_nJet"]) and self.config["Mask"]:
            lowerNjets = self.config["minNJetBin"] + 1

        ax1.set_xlim([lowerNjets-0.4,self.config["maxNJetBin"]+0.4])
        
        #ax2.errorbar(binCenters, abcdPull, yerr=None,        xerr=xErr, fmt='', color="blue",  lw=0, elinewidth=2, marker="o", markerfacecolor="blue")
        #ax2.axhline(y=0.0, color="black", linestyle="dashed", lw=1)
        #ax2.grid(color="black", which="major", axis="y")
        
        ax2.errorbar(binCenters, abcdError, yerr=None,        xerr=xErr, fmt='', color="blue",  lw=0, elinewidth=2, marker="o", markerfacecolor="blue")
        ax2.axhline(y=0.0, color="black", linestyle="dashed", lw=1)
        ax2.grid(color="black", which="both", axis="y")

        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax1)
        
        ax2.set_xlabel('Number of jets')
        ax2.set_ylabel('1 - Pred./Obs.', fontsize="small")
        #ax2.set_ylabel('(Pred. - Obs.) / $\delta$')
        ax1.set_ylabel('Unweighted Event Counts')
        ax1.legend(loc='best')
        
        #ax2.set_ylim([-5.9, 5.9])
        ax2.set_ylim([-1.6, 1.6])
        
        fig.savefig(self.config["outputDir"]+"/Njets_Region_A_PredVsActual.pdf")
        
        plt.close(fig)

        return totalChi2, wtotalChi2/totalSig, ndof

    def makePlots(self, doQuickVal=False, xvalMass="400", xvalModel="RPV_SYY_SHH"):
        NJetsRange = range(self.config["minNJetBin"], self.config["maxNJetBin"]+1)

        # Validation set used events not a part of training samples
        sgValSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+"*Val.root") for mass in self.config["signal"]) , [])
        bgValSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+bkgd+"*Val.root") for bkgd in self.config["bkgd"][1]), [])
        valData, valSg, valBg = get_data(sgValSet, bgValSet, self.config)

        # Xvalidation set could be from samples that were not even trained on and the network has never seen
        loadXvalSignal = not self.samplesLoaded(self.config["signal"], self.config["signalVal"])
        loadXvalBkgd   = not self.samplesLoaded(self.config["bkgd"][1], self.config["bkgdVal"][1])

        bgXvalSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+bkgd+"*.root") for bkgd in self.config["bkgdVal"][1]), [])
        sgXvalSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+"*.root") for mass in self.config["signalVal"]), [])

        #if loadXvalBkgd or loadXvalSignal:
        self.Data, self.Sg, self.Bg = get_data(sgXvalSet, bgXvalSet, self.config)

        # For validating that the training on the samples worked well
        # Pick a safe mass point and model that was present in the training set
        # The "validation" events will be drawn based on this model/mass combination
        tRange = list(range(self.config["minStopMass"], self.config["maxStopMass"], 50))
        print(tRange)
        valMass = None
        if 550 in tRange:
            valMass = 550
        else:
            valMass = tRange[0]

        valModel = None
        if "RPV" in self.config["trainModel"]:
            valModel = "RPV"
        elif "SYY" in self.config["trainModel"]:
            valModel = "SYY"
        elif "SHH" in self.config["trainModel"]:
            valModel = "SHH"

        massMask = self.Sg["mask_m%s"%(xvalMass)]
        massMaskVal = valSg["mask_m%s"%(valMass)]

        # Make signal model mask for signal training dataset
        rpvMask = self.Sg["mask_RPV"]
        syyMask = self.Sg["mask_SYY"]
        shhMask = self.Sg["mask_SHH"]

        # Make signal model mask for mixed training dataset
        rpvMaskData = self.Data["mask_RPV"]
        syyMaskData = self.Data["mask_SYY"]
        shhMaskData = self.Data["mask_SHH"]

        # Make signal model mask for signal validation dataset
        rpvMaskVal = valSg["mask_RPV"]
        syyMaskVal = valSg["mask_SYY"]
        shhMaskVal = valSg["mask_SHH"]

        # Make signal model mask for mixed validation dataset
        rpvMaskValData = valData["mask_RPV"]
        syyMaskValData = valData["mask_SYY"]
        shhMaskValData = valData["mask_SHH"]

        sigMask = None; sigMaskData = None; sigMaskVal = None; sigMaskValData = None
        if   "RPV" in xvalModel:
            if sigMask is None:
                sigMask = rpvMask
            else:
                sigMask |= rpvMask

            if sigMaskData is None:
                sigMaskData = rpvMaskData
            else:
                sigMaskData |= rpvMaskData

        if "RPV" in valModel:
            if sigMaskVal is None:
                sigMaskVal = rpvMaskVal
            else:
                sigMaskVal |= rpvMaskVal

            if sigMaskValData is None:
                sigMaskValData = rpvMaskValData
            else:
                sigMaskValData |= rpvMaskValData

        if "SYY" in xvalModel:
            if sigMask is None:
                sigMask = syyMask
            else:
                sigMask |= syyMask

            if sigMaskData is None:
                sigMaskData = syyMaskData
            else:
                sigMaskData |= syyMaskData

        if "SYY" in valModel:
            if sigMaskVal is None:
                sigMaskVal = syyMaskVal
            else:
                sigMaskVal |= syyMaskVal

            if sigMaskValData is None:
                sigMaskValData = syyMaskValData
            else:
                sigMaskValData |= syyMaskValData

        if "SHH" in xvalModel:
            if sigMask is None:
                sigMask = shhMask
            else:
                sigMask |= shhMask

            if sigMaskData is None:
                sigMaskData = shhMaskData
            else:
                sigMaskData |= shhMaskData

        if "SHH" in valModel:
            if sigMaskVal is None:
                sigMaskVal = shhMaskVal
            else:
                sigMaskVal |= shhMaskVal

            if sigMaskValData is None:
                sigMaskValData = shhMaskValData
            else:
                sigMaskValData |= shhMaskValData

        # Part of the training samples that were not used for training
        output_Val, output_Val_Sg, output_Val_Bg = self.getOutput(self.model, valData["data"], valSg["data"], valBg["data"])

        # Separately loaded samples that can have nothing to do with the what was loaded for training
        output_Train, output_Xval_Sg, output_Xval_Bg = self.getOutput(self.model, self.Data["data"], self.Sg["data"], self.Bg["data"])

        y_Val_disc1, y_Val_Sg_disc1, y_Val_Bg_disc1 = self.getResults(output_Val, output_Val_Sg, output_Val_Bg, outputNum=0, columnNum=0)
        y_Val_disc2, y_Val_Sg_disc2, y_Val_Bg_disc2 = self.getResults(output_Val, output_Val_Sg, output_Val_Bg, outputNum=0, columnNum=2)
        y_Val_mass, y_Val_mass_Sg, y_Val_mass_Bg = self.getResults(output_Val, output_Val_Sg, output_Val_Bg, outputNum=3, columnNum=0)
        y_Xval_disc1, y_Xval_Sg_disc1, y_Xval_Bg_disc1 = self.getResults(output_Train, output_Xval_Sg, output_Xval_Bg, outputNum=0, columnNum=0)
        y_Xval_disc2, y_Xval_Sg_disc2, y_Xval_Bg_disc2 = self.getResults(output_Train, output_Xval_Sg, output_Xval_Bg, outputNum=0, columnNum=2)
        y_Xval_mass, y_Xval_mass_Sg, y_Xval_mass_Bg = self.getResults(output_Train, output_Xval_Sg, output_Xval_Bg, outputNum=3, columnNum=0)

        nBins = 20
        nBinsReg = 100
        masses = [350, 550, 850, 1150]

        colors = ["red", "green", "blue", "magenta", "cyan"]; labels = ["Bg Train", "Bg Val"]

        self.plotDisc([y_Xval_mass_Bg, y_Val_mass_Bg], colors, labels, [self.Bg["Weight"], valBg["Weight"]], "mass", 'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg)
        self.plotDisc([y_Xval_mass_Bg, y_Val_mass_Bg], colors, labels, [self.Bg["Weight"], valBg["Weight"]], "mass_log", 'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg, doLog=True)

        tempColors = ["black"]; tempNames = ["ttbar"]; tempMass = [y_Xval_mass_Bg]; tempEvents = [self.Bg["Weight"]]; tempMassVal = [y_Val_mass_Bg]; tempEventsVal = [valBg["Weight"]]
        i = 0
        for imass in masses:
            mask = "mask_m%d"%(imass)
            self.plotDisc([y_Xval_mass_Sg[self.Sg[mask]&sigMask], y_Val_mass_Sg[valSg[mask]]], colors, labels, [self.Sg["Weight"][self.Sg[mask]&sigMask], valSg["Weight"][valSg[mask]]], "mass_%d"%(imass), 'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg)

            tempColors.append(colors[i])
            tempNames.append("mass %d"%(imass))
            tempMass.append(y_Xval_mass_Sg[self.Sg[mask]&sigMask])
            tempEvents.append(self.Sg["Weight"][self.Sg[mask]&sigMask])

            tempMassVal.append(y_Val_mass_Sg[valSg[mask]])
            tempEventsVal.append(valSg["Weight"][valSg[mask]])

            i += 1

        self.plotDisc([y_Xval_Bg_disc1, y_Val_Bg_disc1], colors, labels, [self.Bg["Weight"], valBg["Weight"]], "Disc1", 'Norm Events', 'Disc. 1')
        self.plotDisc([y_Xval_Bg_disc2, y_Val_Bg_disc2], colors, labels, [self.Bg["Weight"], valBg["Weight"]], "Disc2", 'Norm Events', 'Disc. 2')

        self.plotDisc(tempMass, tempColors, tempNames, tempEvents, "mass_split", 'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg)
        self.plotDisc(tempMass, tempColors, tempNames, tempEvents, "mass_split_log", 'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg, doLog=True)

        self.plotDisc(tempMassVal, tempColors, tempNames, tempEventsVal, "mass_split_val", 'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg)
        self.plotDisc(tempMassVal, tempColors, tempNames, tempEventsVal, "mass_split_val_log", 'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg, doLog=True)

        for NJets in NJetsRange:

            if self.config["Mask"] and self.config["Mask_nJet"] == int(NJets): continue

            njetsStr = "mask_nJet_%s"%(("%s"%(NJets)).zfill(2))
            bkgNjetsMask = self.Bg[njetsStr]; sigNjetsMask = self.Sg[njetsStr]
            bkgNjetsMaskVal = valBg[njetsStr]; sigNjetsMaskVal = valSg[njetsStr]

            tempColors = ["black"]; tempNames = ["ttbar"]; tempMass = [y_Xval_mass_Bg[bkgNjetsMask]]; tempEvents = [self.Bg["Weight"][bkgNjetsMask]]; tempMassVal = [y_Val_mass_Bg[bkgNjetsMaskVal]]; tempEventsVal = [valBg["Weight"][bkgNjetsMaskVal]]
            i = 0
            for imass in masses:
                if imass >= self.config["minStopMass"] and imass <= self.config["maxStopMass"]:
                    mask = "mask_m%d"%(imass)

                    tempColors.append(colors[i])
                    tempNames.append("mass %d"%(imass))
                    tempMass.append(y_Xval_mass_Sg[self.Sg[mask]&sigMask&sigNjetsMask])
                    tempEvents.append(self.Sg["Weight"][self.Sg[mask]&sigMask&sigNjetsMask])

                    tempMassVal.append(y_Val_mass_Sg[valSg[mask]&sigMaskVal&sigNjetsMaskVal])
                    tempEventsVal.append(valSg["Weight"][valSg[mask]&sigMaskVal&sigNjetsMaskVal])

                    i += 1

            self.plotDisc(tempMass, tempColors, tempNames, tempEvents, "mass_split_Njets%s"%(NJets), 'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg)
            self.plotDisc(tempMass, tempColors, tempNames, tempEvents, "mass_split_Njets%s_log"%(NJets), 'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg, doLog=True)

            self.plotDisc(tempMassVal, tempColors, tempNames, tempEventsVal, "mass_split_val_Njets%s"%(NJets), 'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg)
            self.plotDisc(tempMassVal, tempColors, tempNames, tempEventsVal, "mass_split_val_Njets%s_log"%(NJets), 'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg, doLog=True)

        # Plot Acc vs Epoch
        if self.result_log != None:
            self.plotAccVsEpoch('loss', 'val_loss', 'model loss', 'loss_train_val')
            for rank in ["third", "fourth"]: self.plotAccVsEpoch('%s_output_loss'%(rank), 'val_%s_output_loss'%(rank), '%s output loss'%(rank), '%s_output_loss_train_val'%(rank))        

            self.plotAccVsEpochAll('disc_comb_layer', 'third', 'fourth' , 'correlation_layer', 'Combined Disc Loss', 'GR Loss', 'Regression Loss', 'Correlation Loss', '', 'train output loss', 'output_loss_train')
            self.plotAccVsEpochAll('disc_comb_layer', 'third', 'fourth' , 'correlation_layer', 'Combined Disc Loss', 'GR Loss', 'Regression Loss', 'Correlation Loss', 'val_', 'validation output loss', 'output_loss_val')

            #for rank in ["first", "second", "third", "fourth"]: self.plotAccVsEpoch('%s_output_loss'%(rank), 'val_%s_output_loss'%(rank), '%s output loss'%(rank), '%s_output_loss_train_val'%(rank))        
            self.plotAccVsEpoch('correlation_layer_loss', 'val_correlation_layer_loss', 'correlation_layer output loss', 'correlation_layer_loss_train_val')
            self.plotAccVsEpoch('disc_comb_layer_loss', 'val_correlation_layer_loss', 'correlation_layer output loss', 'correlation_layer_loss_train_val')
       
        # Plot disc per njet
        self.plotDiscPerNjet("_Disc1", {"Bg": [self.Bg, y_Xval_Bg_disc1, self.Bg["Weight"]], "Sg": [self.Sg, y_Xval_Sg_disc1, self.Sg["Weight"]]}, sigMask, nBins=nBins)
        self.plotDiscPerNjet("_Disc2", {"Bg": [self.Bg, y_Xval_Bg_disc2, self.Bg["Weight"]], "Sg": [self.Sg, y_Xval_Sg_disc2, self.Sg["Weight"]]}, sigMask, nBins=nBins)
        
        if not doQuickVal:
            self.plotD1VsD2SigVsBkgd(y_Xval_Bg_disc1, y_Xval_Bg_disc2, y_Xval_Sg_disc1[massMask&sigMask], y_Xval_Sg_disc2[massMask&sigMask], xvalMass)
            # Make arrays for possible values to cut on for both discriminant
            # starting at a minimum of 0.5 for each
            c1s = np.arange(0.10, 0.95, 0.02); c2s = np.arange(0.10, 0.95, 0.02)
        
            # Get number of background and signal counts for each A, B, C, D region for every possible combination of cuts on disc 1 and disc 2
            bc, sc = self.cutAndCount(c1s, c2s, y_Xval_Bg_disc1, y_Xval_Bg_disc2, self.Bg["Weight"][:,0], y_Xval_Sg_disc1[massMask&sigMask], y_Xval_Sg_disc2[massMask&sigMask], self.Sg["Weight"][:,0][massMask&sigMask])
        
            # For a given signal fraction, figure out the cut on disc 1 and disc 2 that maximizes the significance
            c1, c2, significance, closureErr, invSigns, closeErrs, sfracA, sbfracA, sfracB, sbfracB, sfracC, sbfracC, sfracD, sbfracD = self.findDiscCut4SigFrac(bc, sc)
        
            _, _ = self.plotBinEdgeMetricComps(significance, closureErr, invSigns, closeErrs, c1s)

            self.config["Disc1"] = c1
            self.config["Disc2"] = c2
            self.config["Significance"] = significance
            self.config["sFracA"] = sfracA
            self.config["sFracB"] = sfracB
            self.config["sFracC"] = sfracC
            self.config["sFracD"] = sfracD
            self.config["sbFracA"] = sbfracA
            self.config["sbFracB"] = sbfracB
            self.config["sbFracC"] = sbfracC
            self.config["sbFracD"] = sbfracD
 
            #for i in range(len(self.config["allVars"])):
            #    theVar = self.config["allVars"][i]
            #    self.plotInputVarMassCut(100, y_Xval_mass_Bg, y_Xval_mass_Sg, self.Bg, self.Sg, sigMask, massMask, theVar, theVar, "BGvSG", mass = "", Njets=-1, bins=100)
            #    self.plotInputVar(float(c1), float(c2), y_Xval_Bg_disc1, y_Xval_Bg_disc2, self.Bg, self.Sg, sigMask, massMask, theVar, theVar, "BG", mass = "", Njets = -1, bins = 64)

            # Plot each discriminant for sig and background while making cut on other disc
            self.plotDiscWithCut(float(c2), y_Xval_Bg_disc1, y_Xval_Bg_disc2, self.Bg["Weight"][:,0], y_Xval_Sg_disc1[massMask&sigMask], y_Xval_Sg_disc2[massMask&sigMask], self.Sg["Weight"][:,0][massMask&sigMask], "1", "2", mass=xvalMass, Njets=-1, bins=nBins)
            self.plotDiscWithCut(float(c1), y_Xval_Bg_disc2, y_Xval_Bg_disc1, self.Bg["Weight"][:,0], y_Xval_Sg_disc2[massMask&sigMask], y_Xval_Sg_disc1[massMask&sigMask], self.Sg["Weight"][:,0][massMask&sigMask], "2", "1", mass=xvalMass, Njets=-1, bins=nBins)
        
            self.plotDiscWithCutCompare(float(c2), y_Xval_Bg_disc1, y_Xval_Bg_disc2, self.Bg["Weight"][:,0], "1", "2", "BG", mass="", Njets=-1, bins=10)
            self.plotDiscWithCutCompare(float(c2), y_Xval_Sg_disc1[massMask&sigMask], y_Xval_Sg_disc2[massMask&sigMask], self.Sg["Weight"][:,0][massMask&sigMask], "1", "2", "SG", mass=xvalMass, Njets=-1, bins=10)
        
            self.plotDiscWithCutCompare(float(c1), y_Xval_Bg_disc2, y_Xval_Bg_disc1, self.Bg["Weight"][:,0], "2", "1", "BG", bins=10)
            self.plotDiscWithCutCompare(float(c1), y_Xval_Sg_disc2[massMask&sigMask], y_Xval_Sg_disc1[massMask&sigMask], self.Sg["Weight"][:,0][massMask&sigMask], "2", "1", "SG", mass=xvalMass, Njets=-1, bins=10)


            #bg1s = []; bg2s = []; wbg = []
            #sg1s = []; sg2s = []; wsg = []
            #for i in range(0, 1000000):
            #    bg1 = np.random.exponential(1.0)
            #    bg2 = np.random.exponential(1.0)

            #    sg1 = np.random.normal(0.95, 0.1)
            #    sg2 = np.random.normal(0.95, 0.1)

            #    if bg1 >= 0.0 and bg1 <= 1.0 and bg2 >= 0.0 and bg2 <= 1.0:
            #        bg1s.append(bg1); bg2s.append(bg2); wbg.append(1.0)
            #
            #    if sg1 >= 0.0 and sg1 <= 1.0 and sg2 >= 0.0 and sg2 <= 1.0:
            #        sg1s.append(sg1); sg2s.append(sg2); wsg.append(1.0)

            #self.plotDisc1vsDisc2(np.array(bg1s), np.array(bg2s), np.array(wbg), -1.0, -1.0, -999.0, "BGDREAM")
            #self.plotDisc1vsDisc2(np.array(sg1s), np.array(sg2s), np.array(wsg), -1.0, -1.0, -999.0, "SGDREAM")
           
        
            # Plot 2D of the discriminants
            self.plotDisc1vsDisc2(y_Xval_Bg_disc1, y_Xval_Bg_disc2, self.Bg["Weight"][:,0], float(c1), float(c2), significance, "BG")
            self.plotDisc1vsDisc2(y_Xval_Sg_disc1[massMask&sigMask], y_Xval_Sg_disc2[massMask&sigMask], self.Sg["Weight"][:,0][massMask&sigMask], float(c1), float(c2), significance, "SG", mass=xvalMass)

            self.plotDisc1vsDisc2(y_Val_Bg_disc1, y_Val_Bg_disc2, valBg["Weight"][:,0], float(c1), float(c2), significance, "valBG")
            self.plotDisc1vsDisc2(y_Val_Sg_disc1[massMaskVal&sigMaskVal], y_Val_Sg_disc2[massMaskVal&sigMaskVal], valSg["Weight"][:,0][massMaskVal&sigMaskVal], float(c1), float(c2), significance, "valSG", mass=valMass)

            bkgdNjets    = {"A" : [], "B" : [], "C" : [], "D" : [], "a" : [], "b" : [], "c" : [], "d" : []}; sigNjets    = {"A" : [], "B" : [], "C" : [], "D" : [], "a" : [], "b" : [], "c" : [], "d" : []}
            bkgdNjetsErr = {"A" : [], "B" : [], "C" : [], "D" : [], "a" : [], "b" : [], "c" : [], "d" : []}; sigNjetsErr = {"A" : [], "B" : [], "C" : [], "D" : [], "a" : [], "b" : [], "c" : [], "d" : []}
            bkgdNjetsAPred = {"val" : [], "err" : []}
            bkgdNjetsSign = []
            bkgdCorrs = []
            
            # Training with systs
            #c1vals = {7 : "0.50", 8 : "0.62", 9 : "0.70", 10 : "0.78", 11 : "0.82"}
            #c2vals = {7 : "0.58", 8 : "0.64", 9 : "0.68", 10 : "0.68", 11 : "0.70"}

            # Training just with POWHEG
            #c1vals = {7 : "0.62", 8 : "0.78", 9 : "0.82", 10 : "0.88", 11 : "0.88"}
            #c2vals = {7 : "0.72", 8 : "0.70", 9 : "0.76", 10 : "0.76", 11 : "0.82"}

            aveClosure = []; stdClosure = []

            for NJets in NJetsRange:
            
                if self.config["Mask"] and self.config["Mask_nJet"] == int(NJets): continue

                njetsStr = "mask_nJet_%s"%(("%s"%(NJets)).zfill(2))
                bkgNjetsMask = self.Bg[njetsStr]; sigNjetsMask = self.Sg[njetsStr]
                bkgFullMask  = bkgNjetsMask;           sigFullMask  = sigMask & massMask & sigNjetsMask

                bkgNjetsMaskVal = valBg[njetsStr]; sigNjetsMask = valSg[njetsStr]
                bkgFullMaskVal  = bkgNjetsMaskVal; sigFullMaskVal  = sigMaskVal & massMaskVal & sigNjetsMask

                # Get number of background and signal counts for each A, B, C, D region for every possible combination of cuts on disc 1 and disc 2
                bc, sc = self.cutAndCount(c1s, c2s, y_Xval_Bg_disc1[bkgFullMask], y_Xval_Bg_disc2[bkgFullMask], self.Bg["Weight"][:,0][bkgFullMask], y_Xval_Sg_disc1[sigFullMask], y_Xval_Sg_disc2[sigFullMask], self.Sg["Weight"][:,0][sigFullMask])
                c1, c2, significance, closureErr, invSigns, closeErrs, _, _, _, _, _, _, _, _ = self.findDiscCut4SigFrac(bc, sc)

                tempAveClose, tempStdClose = self.plotBinEdgeMetricComps(significance, closureErr, invSigns, closeErrs, c1s, int(NJets))

                aveClosure.append(tempAveClose)
                stdClosure.append(tempStdClose)

                self.config["aveClosureNjets%s"%(NJets)] = tempAveClose
                self.config["stdClosureNjets%s"%(NJets)] = tempStdClose

                bkgdNjetsSign.append(significance)

                #c1 = c1vals[NJets]
                #c2 = c2vals[NJets]

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

                self.plotAveClosureNjets(aveClosure, stdClosure)

                self.config["c1_nJet_%s"%(("%s"%(NJets)).zfill(2))] = c1
                self.config["c2_nJet_%s"%(("%s"%(NJets)).zfill(2))] = c2

                # Avoid completely masked Njets bins that makes below plotting
                # highly unsafe
                if not any(bkgNjetsMask) or not any(sigNjetsMask): continue

                self.plotD1VsD2SigVsBkgd(y_Xval_Bg_disc1[bkgFullMask], y_Xval_Bg_disc2[bkgFullMask], y_Xval_Sg_disc1[sigFullMask], y_Xval_Sg_disc2[sigFullMask], xvalMass, NJets)

                # Plot each discriminant for sig and background while making cut on other disc
                self.plotDiscWithCut(float(c2), y_Xval_Bg_disc1[bkgFullMask], y_Xval_Bg_disc2[bkgFullMask], self.Bg["Weight"][:,0][bkgFullMask], y_Xval_Sg_disc1[sigFullMask], y_Xval_Sg_disc2[sigFullMask], self.Sg["Weight"][:,0][sigFullMask], "1", "2", mass=xvalMass, Njets=NJets, bins=nBins)
                self.plotDiscWithCut(float(c1), y_Xval_Bg_disc2[bkgFullMask], y_Xval_Bg_disc1[bkgFullMask], self.Bg["Weight"][:,0][bkgFullMask], y_Xval_Sg_disc2[sigFullMask], y_Xval_Sg_disc1[sigFullMask], self.Sg["Weight"][:,0][sigFullMask], "2", "1", mass=xvalMass, Njets=NJets, bins=nBins)
            
                self.plotDiscWithCutCompare(float(c2), y_Xval_Bg_disc1[bkgFullMask], y_Xval_Bg_disc2[bkgFullMask], self.Bg["Weight"][:,0][bkgFullMask], "1", "2", "BG", mass="", Njets=-1, bins=10)
                self.plotDiscWithCutCompare(float(c2), y_Xval_Sg_disc1[sigFullMask], y_Xval_Sg_disc2[sigFullMask], self.Sg["Weight"][:,0][sigFullMask], "1", "2", "SG", mass=xvalMass, Njets=NJets, bins=10)
            
                self.plotDiscWithCutCompare(float(c1), y_Xval_Bg_disc2[bkgFullMask], y_Xval_Bg_disc1[bkgFullMask], self.Bg["Weight"][:,0][bkgFullMask], "2", "1", "BG", mass="", Njets=-1, bins=10)
                self.plotDiscWithCutCompare(float(c1), y_Xval_Sg_disc2[sigFullMask], y_Xval_Sg_disc1[sigFullMask], self.Sg["Weight"][:,0][sigFullMask], "2", "1", "SG", mass=xvalMass, Njets=NJets, bins=10)
            
                # Plot 2D of the discriminants
                bkgdCorr = self.plotDisc1vsDisc2(y_Xval_Bg_disc1[bkgFullMask], y_Xval_Bg_disc2[bkgFullMask], self.Bg["Weight"][:,0][bkgFullMask], float(c1), float(c2), significance, "BG", mass="",   Njets=NJets)
                self.plotDisc1vsDisc2(y_Xval_Sg_disc1[sigFullMask], y_Xval_Sg_disc2[sigFullMask], self.Sg["Weight"][:,0][sigFullMask], float(c1), float(c2), significance, "SG", mass=xvalMass, Njets=NJets)
                self.metric["bkgdCorr_nJet_%s"%(NJets)] = abs(bkgdCorr) 
                bkgdCorrs.append(bkgdCorr)

                self.plotDisc1vsDisc2(y_Val_Bg_disc1[bkgFullMaskVal], y_Val_Bg_disc2[bkgFullMaskVal], valBg["Weight"][:,0][bkgFullMaskVal], float(c1), float(c2), significance, "valBG", mass="",   Njets=NJets)
                self.plotDisc1vsDisc2(y_Val_Sg_disc1[sigFullMaskVal], y_Val_Sg_disc2[sigFullMaskVal], valSg["Weight"][:,0][sigFullMaskVal], float(c1), float(c2), significance, "valSG", mass=valMass, Njets=NJets)

            self.config["bkgdCorrAve"] = np.average(np.abs(bkgdCorrs))
            self.config["bkgdCorrStd"] = np.std(bkgdCorrs)

            signA = self.plotNjets(bkgdNjets["A"], bkgdNjetsErr["A"], sigNjets["A"], sigNjetsErr["A"], "A")
            signB = self.plotNjets(bkgdNjets["B"], bkgdNjetsErr["B"], sigNjets["B"], sigNjetsErr["B"], "B")
            signC = self.plotNjets(bkgdNjets["C"], bkgdNjetsErr["C"], sigNjets["C"], sigNjetsErr["C"], "C")
            signD = self.plotNjets(bkgdNjets["D"], bkgdNjetsErr["D"], sigNjets["D"], sigNjetsErr["D"], "D")

            totalChi2, wtotalChi2, ndof = self.plotNjetsClosure(bkgdNjets["a"], bkgdNjetsErr["a"], bkgdNjetsAPred["val"], bkgdNjetsAPred["err"], bkgdNjetsSign)

            self.config["Achi2"] = totalChi2
            if ndof != 0:
                self.config["Achi2ndof"] = totalChi2/ndof
            else:
                self.config["Achi2ndof"] = 9999.0
            self.config["Asignificance"] = signA
            self.config["Bsignificance"] = signB
            self.config["Csignificance"] = signC
            self.config["Dsignificance"] = signD
            self.config["TotalSignificance"] = (signA**2.0 + signB**2.0 + signC**2.0 + signD**2.0)**0.5

            print("A SIGNIFICANCE: %3.2f"%(signA))
            print("B SIGNIFICANCE: %3.2f"%(signB))
            print("C SIGNIFICANCE: %3.2f"%(signC))
            print("D SIGNIFICANCE: %3.2f"%(signD))
            print("TOTAL SIGNIFICANCE: %3.2f"%((signA**2.0 + signB**2.0 + signC**2.0 + signD**2.0)**0.5))
  
            if    self.config["TotalSignificance"] > 0.0: self.metric["InvTotalSignificance"] = 1.0/self.config["TotalSignificance"]
            else: self.metric["InvTotalSignificance"] = 999.0

        # Plot validation roc curve
        fpr_Val_disc1, tpr_Val_disc1, thresholds_Val_disc1 = roc_curve(valData["labels"][:,0][sigMaskValData], y_Val_disc1[sigMaskValData], sample_weight=valData["Weight"][:,0][sigMaskValData])
        fpr_Val_disc2, tpr_Val_disc2, thresholds_Val_disc2 = roc_curve(valData["labels"][:,0][sigMaskValData], y_Val_disc2[sigMaskValData], sample_weight=valData["Weight"][:,0][sigMaskValData])
        fpr_Xval_disc1, tpr_Xval_disc1, thresholds_Xval_disc1 = roc_curve(self.Data["labels"][:,0][sigMaskData], y_Xval_disc1[sigMaskData], sample_weight=self.Data["Weight"][:,0][sigMaskData])
        fpr_Xval_disc2, tpr_Xval_disc2, thresholds_Xval_disc2 = roc_curve(self.Data["labels"][:,0][sigMaskData], y_Xval_disc2[sigMaskData], sample_weight=self.Data["Weight"][:,0][sigMaskData])
        auc_Val_disc1 = roc_auc_score(valData["labels"][:,0][sigMaskValData], y_Val_disc1[sigMaskValData])
        auc_Val_disc2 = roc_auc_score(valData["labels"][:,0][sigMaskValData], y_Val_disc2[sigMaskValData])
        auc_Xval_disc1 = roc_auc_score(self.Data["labels"][:,0][sigMaskData], y_Xval_disc1[sigMaskData])
        auc_Xval_disc2 = roc_auc_score(self.Data["labels"][:,0][sigMaskData], y_Xval_disc2[sigMaskData])
        
        # Define metrics for the training
        self.metric["OverTrain_Disc1"] = abs(auc_Val_disc1 - auc_Xval_disc1)
        self.metric["OverTrain_Disc2"] = abs(auc_Val_disc2 - auc_Xval_disc2)
        self.metric["Performance_Disc1"] = abs(1 - auc_Xval_disc1)
        self.metric["Performance_Disc2"] = abs(1 - auc_Xval_disc2)
       
        # Plot some ROC curves
        self.plotROC("_Disc1", None, None, None, None, fpr_Val_disc1, tpr_Val_disc1, fpr_Xval_disc1, tpr_Xval_disc1, auc_Val_disc1, auc_Xval_disc1)
        self.plotROC("_Disc2", None, None, None, None, fpr_Val_disc2, tpr_Val_disc2, fpr_Xval_disc2, tpr_Xval_disc2, auc_Val_disc2, auc_Xval_disc2)
        self.plotROC("_"+self.config["bkgd"][0]+"_nJet_disc1", y_Xval_disc1, y_Val_disc1, self.Data, valData)
        self.plotROC("_"+self.config["bkgd"][0]+"_nJet_disc2", y_Xval_disc2, y_Val_disc2, self.Data, valData)
        
        # Plot validation precision recall
        precision_Val_disc1, recall_Val_disc1, _ = precision_recall_curve(valData["labels"][:,0][sigMaskValData], y_Val_disc1[sigMaskValData], sample_weight=valData["Weight"][:,0][sigMaskValData])
        precision_Xval_disc1, recall_Xval_disc1, _ = precision_recall_curve(self.Data["labels"][:,0][sigMaskData], y_Xval_disc1[sigMaskData], sample_weight=self.Data["Weight"][:,0][sigMaskData])
        ap_Val_disc1 = average_precision_score(valData["labels"][:,0], y_Val_disc1, sample_weight=valData["Weight"][:,0])
        ap_Xval_disc1 = average_precision_score(self.Data["labels"][:,0], y_Xval_disc1, sample_weight=self.Data["Weight"][:,0])
        
        self.plotPandR(precision_Val_disc1, recall_Val_disc1, precision_Xval_disc1, recall_Xval_disc1, ap_Val_disc1, ap_Xval_disc1)
        
        # Plot NJet dependance
        binxl = self.config["minNJetBin"]
        binxh = self.config["maxNJetBin"] + 1
        numbin = binxh - binxl        
        self.plot2DVar(name="nJet", binxl=binxl, binxh=binxh, numbin=numbin, xIn=self.Bg["nJet"][:,0], yIn=y_Xval_Bg_disc1, nbiny=50)
        
        # Save useful stuff
        self.Data["y"] = y_Xval_disc1
        #np.save(self.config["outputDir"]+"/deepESMbin_dis_nJet.npy", self.Data)
        
        for key in self.metric:
            print(key, self.metric[key])
        
        self.config["metric"] = self.metric
        with open(self.config["outputDir"]+"/config.json",'w') as configFile:
            json.dump(self.config, configFile, indent=4, sort_keys=True)

        return self.metric

