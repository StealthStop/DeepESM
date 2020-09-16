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

class Validation:

    def __init__(self, model, config, sgTrainSet, trainData, trainSg, trainBg, result_log):
        self.model = model
        self.config = config
        self.sgTrainSet = sgTrainSet
        self.trainData = trainData
        self.trainSg = trainSg
        self.trainBg = trainBg
        self.result_log = result_log
        self.metric = {}
        self.doLog = False

    def __del__(self):
        del self.model
        del self.config
        del self.sgTrainSet
        del self.trainData
        del self.trainSg
        del self.trainBg
        del self.result_log
        del self.metric
        
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
        fig.savefig(self.config["outputDir"]+"/"+name+"_discriminator.png", dpi=fig.dpi)        

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

    # Plot a set of 1D hists together, where the hists, colors, labels, weights
    # are provided as a list argument.
    def plotDisc(self, hists, colors, labels, weights, bins, name, xlab, ylab, doLog=False):
        # Plot predicted mass
        fig, ax = plt.subplots(figsize=(10, 10))
        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel(xlab); ax.set_xlabel(ylab)

        for i in range(0, len(hists)): plt.hist(hists[i], bins, color="xkcd:"+colors[i], alpha=0.9, histtype='step', lw=2, label=labels[i], density=True, log=doLog, weights=weights[i])

        ax.legend(loc=2, frameon=False)
        fig.savefig(self.config["outputDir"]+"/%s.png"%(name), dpi=fig.dpi)        
        plt.close(fig)

    # Member function to plot the discriminant variable while making a selection on another discriminant
    # The result for signal and background are shown together
    def plotDiscWithCut(self, c, b1, b2, bw, s1, s2, sw, tag1, tag2, mass, arange=(0,1), bins=100):
        bnew = np.zeros(len(b1)); bwnew = np.zeros(len(b1)); bw2new = np.zeros(len(b1))
        snew = np.zeros(len(s1)); swnew = np.zeros(len(s1)); sw2new = np.zeros(len(s1))
        blong = len(b1); slong = len(s1)
        total = blong if blong > slong else slong 
        j = 0; k = 0
        for i in range(0, total):

            if i < blong:        
                if b2[i] > c:
                    bnew[j] = b1[i]; bwnew[j] = bw[i]; bw2new[j] = bw[i]**2.0
                    j += 1

            if i < slong:
                if s2[i] > c:
                    snew[k] = s1[i]; swnew[k] = sw[i]; sw2new[j] = sw[i]**2.0
                    k += 1

        bwnewBinned, binEdges  = np.histogram(bnew, bins=bins, range=arange, weights=bwnew)
        swnewBinned, binEdges  = np.histogram(snew, bins=bins, range=arange, weights=swnew)
        bw2newBinned, binEdges = np.histogram(bnew, bins=bins, range=arange, weights=bw2new)
        sw2newBinned, binEdges = np.histogram(snew, bins=bins, range=arange, weights=sw2new)

        fig = plt.figure()

        ax = hep.histplot(h=bwnewBinned, bins=binEdges, w2=bw2newBinned, histtype="step", label="Background", alpha=0.9, lw=2)
        ax = hep.histplot(h=swnewBinned, bins=binEdges, w2=sw2newBinned, histtype="step", label="Signal (mass = %s GeV)"%(mass), alpha=0.9, lw=2, ax=ax)

        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel('A.U.'); ax.set_xlabel('Disc. %s'%(tag1))

        plt.text(0.05, 0.85, r"$\bf{Disc. %s}$ > %.3f"%(tag2,c), transform=ax.transAxes, fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))

        # Stupid nonsense to remove duplicate entries in legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-2:], labels[-2:], loc=2, frameon=False)
 
        fig.savefig(self.config["outputDirSpec"]+"/Disc%s_BvsS_m%s.png"%(tag1,mass))
        plt.close(fig)

    # Member function to plot the discriminant variable while making a selection on another discriminant
    # Compare the discriminant shape on either "side" of the selection on the other disc.
    def plotDiscWithCutCompare(self, c, d1, d2, dw, tag1, tag2, tag3, mass = "", arange=(0,1), bins=100):
        dgt = np.zeros(len(d1)); dwgt = np.zeros(len(d1)); dw2gt = np.zeros(len(d1))
        dlt = np.zeros(len(d1)); dwlt = np.zeros(len(d1)); dw2lt = np.zeros(len(d1))
        j = 0
        for i in range(0, len(d2)):
        
            if d2[i] > c:
                dgt[j] = d1[i]; dwgt[j] = dw[i]; dw2gt[j] = dw[i]**2.0
                j += 1
            else:
                dlt[j] = d1[i]; dwlt[j] = dw[i]; dw2lt[j] = dw[i]**2.0
                j += 1

        dwgtBinned,  binEdges = np.histogram(dgt, bins=bins, range=arange, weights=dwgt)
        dw2gtBinned, binEdges = np.histogram(dgt, bins=bins, range=arange, weights=dw2gt)
        dwltBinned,  binEdges = np.histogram(dlt, bins=bins, range=arange, weights=dwlt)
        dw2ltBinned, binEdges = np.histogram(dlt, bins=bins, range=arange, weights=dw2lt)

        fig = plt.figure()

        ax = hep.histplot(h=dwgtBinned, bins=binEdges, w2=dw2gtBinned, histtype="step", label="Disc. %s > %.2f"%(tag2,c), alpha=0.9, lw=2)
        ax = hep.histplot(h=dwltBinned, bins=binEdges, w2=dw2ltBinned, histtype="step", label="Disc. %s < %.2f"%(tag2,c), alpha=0.9, lw=2, ax=ax)

        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel('A.U.'); ax.set_xlabel('Disc. %s'%(tag1))

        # Stupid nonsense to remove duplicate entries in legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-2:], labels[-2:], loc=2, frameon=False)

        fig.savefig(self.config["outputDirSpec"]+"/%s%s_Disc%s_Compare_Shapes.png"%(tag3, mass, tag1))
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
        fig.savefig(self.config["outputDir"]+"/%s.png"%(name), dpi=fig.dpi)
        plt.close(fig)

    def plotDiscPerNjet(self, tag, samples, nBins=100):
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
                    yt = y_train_Sp[trainSample[key]]                
                    wt = weights[trainSample[key]]
                    if yt.size != 0 and wt.size != 0:
                        plt.hist(yt, bins, alpha=0.9, histtype='step', lw=2, label=sample+" Train "+key, density=True, log=self.doLog, weights=wt)
            plt.legend(loc='best')
            fig.savefig(self.config["outputDir"]+"/nJet_"+sample+tag+".png", dpi=fig.dpi)
            plt.close(fig)

    def plotROC(self, tag="", y_Train=None, trainData=None, xVal=None, yVal=None, xTrain=None, yTrain=None, valLab=None, trainLab=None):
        fig = plt.figure()
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve', pad=45.0)

        if y_Train is None:
            plt.plot(xVal, yVal, color='xkcd:black', label='Val (area = {:.3f})'.format(valLab))
            plt.plot(xTrain, yTrain, color='xkcd:red', label='Train (area = {:.3f})'.format(trainLab))

        else:
            for key in sorted(trainData.keys()):
                if key.find("mask_nJet") != -1:
                    labels = trainData["labels"][trainData[key]]
                    weights = trainData["Weight"][trainData[key]][:,0]
                    y = y_Train[trainData[key]]
                    if len(y)==0:
                        continue
                    fpr_Train, tpr_Train, thresholds_Train = roc_curve(labels[:,0], y, sample_weight=weights)
                    auc_Train = roc_auc_score(labels[:,0], y)    
                    plt.plot(fpr_Train, tpr_Train, label="Train "+key+" (area = {:.3f})".format(auc_Train))

        newtag = tag.replace(" ", "_")
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/roc_plot"+newtag+".png", dpi=fig.dpi)
        plt.close(fig)

    # Plot disc1 vs disc2 for both background and signal
    def plotD1VsD2SigVsBkgd(self, b1, b2, s1, s2, mass):
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
        fig.savefig(self.config["outputDir"]+"/2D_SigVsBkgd_Disc1VsDisc2_m%s.png"%(mass), dpi=fig.dpi)        
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
        fig.savefig(self.config["outputDir"]+"/PandR_plot.png", dpi=fig.dpi)        
        plt.close(fig)

    # Just plot the 2D for either background or signal
    def plotDisc1vsDisc2(self, disc1, disc2, bw, sw, c1, c2, significance, tag, nBins=100, mass = ""):
        fig = plt.figure() 
        corr = cor.pearson_corr(disc1, disc2)
        plt.hist2d(disc1, disc2, bins=[nBins, nBins], range=[[0, 1], [0, 1]], cmap=plt.cm.jet, weights=bw, cmin = sw.min())
        plt.colorbar()
        ax = plt.gca()
        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="red", linewidth=2); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="red", linewidth=2)
        ax.add_line(l1); ax.add_line(l2)
        ax.set_ylabel("Disc. 2"); ax.set_xlabel("Disc. 1")
        plt.text(0.05, 0.90, r"$\bf{CC}$ = %.3f"%(corr), fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))
        plt.text(0.05, 0.95, r"$\bf{Significance}$ = %.3f"%(significance), fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        fig.savefig(self.config["outputDirSpec"]+"/2D_%s%s_Disc1VsDisc2.png"%(tag,mass), dpi=fig.dpi)

    def cutAndCount(self, c1s, c2s, b1, b2, bw, s1, s2, sw, cdiff = 0.2):
        # First get the total counts in region "D" for all possible c1, c2
        bcounts = {"A" : {}, "B" : {}, "C" : {}, "D" : {}, "A2" : {}, "B2" : {}, "C2" : {}, "D2" : {}}  
        scounts = {"A" : {}, "B" : {}, "C" : {}, "D" : {}, "A2" : {}, "B2" : {}, "C2" : {}, "D2" : {}}  
        blong = len(b1); slong = len(s1)
        total = blong if blong > slong else slong 
        for i in range(0, total):
            for c1 in c1s:
                c1k = "%.2f"%c1
                if c1k not in bcounts["A"]:
                    bcounts["A"][c1k] = {}; bcounts["A2"][c1k] = {}
                    bcounts["B"][c1k] = {}; bcounts["B2"][c1k] = {}
                    bcounts["C"][c1k] = {}; bcounts["C2"][c1k] = {}
                    bcounts["D"][c1k] = {}; bcounts["D2"][c1k] = {}
                if c1k not in scounts["A"]:
                    scounts["A"][c1k] = {}; scounts["A2"][c1k] = {} 
                    scounts["B"][c1k] = {}; scounts["B2"][c1k] = {} 
                    scounts["C"][c1k] = {}; scounts["C2"][c1k] = {} 
                    scounts["D"][c1k] = {}; scounts["D2"][c1k] = {} 

                for c2 in c2s:
                
                    # Keep scan of c1,c2 close to the diagnonal
                    # Default c1 and c2 to be within 20% of one another
                    if abs(1.0 - c1/c2) > cdiff: continue

                    c2k = "%.2f"%c2
                    if c2k not in bcounts["A"][c1k]:
                        bcounts["A"][c1k][c2k] = 0.0; bcounts["A2"][c1k][c2k] = 0.0
                        bcounts["B"][c1k][c2k] = 0.0; bcounts["B2"][c1k][c2k] = 0.0
                        bcounts["C"][c1k][c2k] = 0.0; bcounts["C2"][c1k][c2k] = 0.0
                        bcounts["D"][c1k][c2k] = 0.0; bcounts["D2"][c1k][c2k] = 0.0
                    if c2k not in scounts["A"][c1k]:
                        scounts["A"][c1k][c2k] = 0.0; scounts["A2"][c1k][c2k] = 0.0
                        scounts["B"][c1k][c2k] = 0.0; scounts["B2"][c1k][c2k] = 0.0
                        scounts["C"][c1k][c2k] = 0.0; scounts["C2"][c1k][c2k] = 0.0
                        scounts["D"][c1k][c2k] = 0.0; scounts["D2"][c1k][c2k] = 0.0

                    if i < blong:
                        if b1[i] > c1 and b2[i] > c2:
                            bcounts["A"][c1k][c2k]  += bw[i]
                            bcounts["A2"][c1k][c2k] += bw[i]**2.0
                        if b1[i] > c1 and b2[i] < c2:
                            bcounts["B"][c1k][c2k]  += bw[i]
                            bcounts["B2"][c1k][c2k] += bw[i]**2.0
                        if b1[i] < c1 and b2[i] > c2:
                            bcounts["C"][c1k][c2k]  += bw[i]
                            bcounts["C2"][c1k][c2k] += bw[i]**2.0
                        if b1[i] < c1 and b2[i] < c2:
                            bcounts["D"][c1k][c2k]  += bw[i]
                            bcounts["D2"][c1k][c2k] += bw[i]**2.0
                    if i < slong:
                        if s1[i] > c1 and s2[i] > c2:
                            scounts["A"][c1k][c2k]  += sw[i]
                            scounts["A2"][c1k][c2k] += sw[i]**2.0
                        if s1[i] > c1 and s2[i] < c2:
                            scounts["B"][c1k][c2k]  += sw[i]
                            scounts["B2"][c1k][c2k] += sw[i]**2.0
                        if s1[i] < c1 and s2[i] > c2:
                            scounts["C"][c1k][c2k]  += sw[i]
                            scounts["C2"][c1k][c2k] += sw[i]**2.0
                        if s1[i] < c1 and s2[i] < c2:
                            scounts["D"][c1k][c2k]  += sw[i]
                            scounts["D2"][c1k][c2k] += sw[i]**2.0

        return bcounts, scounts

    def findDiscCut4SigFrac(self, bcts, scts, minNB = 5.0, minNS = 5.0):
        # Now calculate signal fraction and significance 
        # Pick c1 and c2 that give 30% sig fraction and maximizes significance
        significance = 0.0; sigFrac = 0.0; finalc1 = -1.0; finalc2 = 0.0; 
        for c1k, c2s in bcts["A"].items():
            for c2k, temp in c2s.items():
                tempsigfrac = -1.0 
                if bcts["A"][c1k][c2k] + scts["A"][c1k][c2k] > 0.0: tempsigfrac = scts["A"][c1k][c2k] / (scts["A"][c1k][c2k] + bcts["A"][c1k][c2k])
                else: tempsigfrac = -1.0

                # Minimum signal fraction requirement
                if bcts["A"][c1k][c2k] > minNB and scts["A"][c1k][c2k] > minNS:

                    sigFrac = tempsigfrac

                    tempsignificance = -1.0
                    if bcts["A"][c1k][c2k]: tempsignificance = scts["A"][c1k][c2k] / (bcts["A"][c1k][c2k] + 0.3*bcts["A"][c1k][c2k]**2.0)**0.5
                    else: tempsignificance = -1.0
                    
                    # Save significance if we found a better one
                    if tempsignificance > significance:
                        finalc1 = c1k; finalc2 = c2k
                        significance = tempsignificance
                
        return finalc1, finalc2, significance, sigFrac

    # Define closure as how far away prediction for region D is compared to actual 
    def simpleClosureABCD(self, bNA, bNB, bNC, bND, bNAerr, bNBerr, bNCerr, bNDerr):
        # Define A: > c1, > c2        C    |    A    
        # Define B: > c1, < c2   __________|__________        
        # Define C: < c1, > c2             |        
        # Define D: < c1, < c2        D    |    B    

        num = bNC * bNB; den = bND * bNA

        if den > 0.0:
            closureErr = ((bNB * bNAerr / den)**2.0 + (bNBerr * bNA / den)**2.0 + ((num * bNAerr) / (den * bNA))**2.0 + ((num * bNDerr) / (den * bND))**2.0)**0.5
            closure = num / den
        else:
            closureErr = -999.0
            closure = -999.0

        return closure, closureErr

    # Calculate Eq. 2.8 from https://arxiv.org/pdf/2007.14400.pdf
    def normSignalContamination(self, bNA, bNB, bNC, bND, sNA, sNB, sNC, sND):
        if not bNA: bNA = 10e-20
        if not bNB: bNB = 10e-20
        if not bNC: bNC = 10e-20
        if not bND: bND = 10e-20

        Ar = sNA / bNA; Br = sNB / bNB; Cr = sNC / bNC; Dr = sND / bND
        
        return (Br + Cr - Dr) / Ar if Ar else -1.0

    # Simple calculation of background rejection
    def backgroundRejection(self, bNA, bNB, bNC, bND):
        bN = bNA + bNB + bNC + bND
        reject = bNB + bNC + bND

        return reject / bN if bN else -1.0

    def plotBkgdRejVsSigCont(self, bc, sc, mass, closureLimit = 0.1):

        # Lame way of figuring out how long the array should be
        nVals = 0
        for c1, c2s in bc["A"].items():
            for c2, temp in c2s.items():
                nVals += 1

        bkgdRej = np.zeros(nVals); sigCont = np.zeros(nVals)

        i = 0
        for c1, c2s in bc["A"].items():
            for c2, temp in c2s.items():

                closure, closureErr = self.simpleClosureABCD(bc["A"][c1][c2], bc["B"][c1][c2], bc["C"][c1][c2], bc["D"][c1][c2], bc["A2"][c1][c2]**0.5, bc["B2"][c1][c2]**0.5, bc["C2"][c1][c2]**0.5, bc["D2"][c1][c2]**0.5) 
                if abs(1.0 - closure) < closureLimit: continue

                sigContamination = self.normSignalContamination(bc["A"][c1][c2], bc["B"][c1][c2], bc["C"][c1][c2], bc["D"][c1][c2], sc["A"][c1][c2], sc["B"][c1][c2], sc["C"][c1][c2], sc["D"][c1][c2])
                backgroundReject = self.backgroundRejection(bc["A"][c1][c2], bc["B"][c1][c2], bc["C"][c1][c2], bc["D"][c1][c2])

                bkgdRej[i] = backgroundReject; sigCont[i] = sigContamination

                i += 1

        fig = plt.figure()
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        plt.ylim(0,1)
        plt.xlim(0,1)
        ax = plt.gca()
        plt.scatter(sigCont, bkgdRej, color='xkcd:red', marker="o", label="30% Signal Fraction")
        plt.xlabel('Normalized Signal Contamination')
        plt.ylabel('Background Rejection')
        plt.legend(loc='best')
        plt.text(0.05, 0.94, r"$\bf{ABCD\;Closure}$ > %.1f"%(closureLimit), transform=ax.transAxes, fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))
        fig.savefig(self.config["outputDir"]+"/SigContamination_vs_BkgdRejection_m%s.png"%(mass), dpi=fig.dpi)        
        plt.close(fig)

    def makePlots(self, doFullVal=False, mass="550"):
        sgValSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+"*Val.root") for mass in self.config["massModels"]) , [])
        bgValSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+ttbar+"*Val.root") for ttbar in self.config["ttbarMC"][1]), [])
        sgOTrainSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+"*Val.root") for mass in self.config["othermassModels"]) , [])
        bgOTrainSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+ttbar+"*Val.root") for ttbar in self.config["otherttbarMC"][1]), [])

        valData, valSg, valBg = get_data(sgValSet, bgValSet, self.config)
        trainOData, trainOSg, trainOBg = get_data(sgOTrainSet, bgOTrainSet, self.config)

        output_Val, output_Val_Sg, output_Val_Bg = self.getOutput(self.model, valData["data"], valSg["data"], valBg["data"])
        output_Train, output_Train_Sg, output_Train_Bg = self.getOutput(self.model, self.trainData["data"], self.trainSg["data"], self.trainBg["data"])
        output_OTrain, output_OTrain_Sg, output_OTrain_Bg = self.getOutput(self.model, trainOData["data"], trainOSg["data"], trainOBg["data"])

        y_Val_disc1, y_Val_Sg_disc1, y_Val_Bg_disc1 = self.getResults(output_Val, output_Val_Sg, output_Val_Bg, 0, 0)
        y_Val_disc2, y_Val_Sg_disc2, y_Val_Bg_disc2 = self.getResults(output_Val, output_Val_Sg, output_Val_Bg, 1, 0)
        #y_Val_mass, y_Val_mass_Sg, y_Val_mass_Bg = self.getResults(output_Val, output_Val_Sg, output_Val_Bg, 4, 0)
        y_Train_disc1, y_Train_Sg_disc1, y_Train_Bg_disc1 = self.getResults(output_Train, output_Train_Sg, output_Train_Bg, 0, 0)
        y_Train_disc2, y_Train_Sg_disc2, y_Train_Bg_disc2 = self.getResults(output_Train, output_Train_Sg, output_Train_Bg, 1, 0)
        #y_Train_mass, y_Train_mass_Sg, y_Train_mass_Bg = self.getResults(output_Train, output_Train_Sg, output_Train_Bg, 4, 0)
        y_OTrain, y_OTrain_Sg, y_OTrain_Bg = self.getResults(output_OTrain, output_OTrain_Sg, output_OTrain_Bg, 0, 0)

        nBins = 20
        colors = ["red", "green", "blue", "magenta"]; labels = ["Sg Train", "Sg Val", "Bg Train", "Bg Val"]
        #self.plotDisc([y_Train_mass_Sg, y_Val_mass_Sg, y_Train_mass_Bg, y_Val_mass_Bg], colors, labels, [self.trainSg["Weight"], valSg["Weight"], self.trainBg["Weight"], valBg["Weight"]], np.linspace(0, 1500, 150), "mass", 'Norm Events', 'predicted mass')
        #self.plotDisc([y_Train_mass_Sg, y_Val_mass_Sg, y_Train_mass_Bg, y_Val_mass_Bg], colors, labels, [self.trainSg["Weight"], valSg["Weight"], self.trainBg["Weight"], valBg["Weight"]], np.linspace(0, 1500, 150), "mass_log", 'Norm Events', 'predicted mass', doLog=True)

        self.plotDisc([y_Train_Sg_disc1, y_Val_Sg_disc1, y_Train_Bg_disc1, y_Val_Bg_disc1], colors, labels, [self.trainSg["Weight"], valSg["Weight"], self.trainBg["Weight"], valBg["Weight"]], np.linspace(0, 1, nBins), "Disc1", 'Norm Events', 'Disc. 1')
        self.plotDisc([y_Train_Sg_disc2, y_Val_Sg_disc2, y_Train_Bg_disc2, y_Val_Bg_disc2], colors, labels, [self.trainSg["Weight"], valSg["Weight"], self.trainBg["Weight"], valBg["Weight"]], np.linspace(0, 1, nBins), "Disc2", 'Norm Events', 'Disc. 2')

        #self.plotDisc([y_Train_mass_Sg[self.trainSg["mask_m550"]], y_Train_mass_Sg[self.trainSg["mask_m850"]], y_Train_mass_Sg[self.trainSg["mask_m1200"]], y_Train_mass_Bg], colors, ["mass 550", "mass 850", "mass 1200", "ttbar"], [self.trainSg["Weight"][self.trainSg["mask_m550"]], self.trainSg["Weight"][self.trainSg["mask_m850"]], self.trainSg["Weight"][self.trainSg["mask_m1200"]], self.trainBg["Weight"]], np.linspace(0, 1500, 150), "mass_split", 'Norm Events', 'predicted mass')
        #self.plotDisc([y_Train_mass_Sg[self.trainSg["mask_m550"]], y_Train_mass_Sg[self.trainSg["mask_m850"]], y_Train_mass_Sg[self.trainSg["mask_m1200"]], y_Train_mass_Bg], colors, ["mass 550", "mass 850", "mass 1200", "ttbar"], [self.trainSg["Weight"][self.trainSg["mask_m550"]], self.trainSg["Weight"][self.trainSg["mask_m850"]], self.trainSg["Weight"][self.trainSg["mask_m1200"]], self.trainBg["Weight"]], np.linspace(0, 1500, 150), "mass_split_log", 'Norm Events', 'predicted mass', doLog=True)

        self.plotD1VsD2SigVsBkgd(y_Train_Bg_disc1, y_Train_Bg_disc2, y_Train_Sg_disc1[self.trainSg["mask_m%s"%(mass)]], y_Train_Sg_disc2[self.trainSg["mask_m%s"%(mass)]], mass)

        # Plot Acc vs Epoch
        self.plotAccVsEpoch('loss', 'val_loss', 'model loss', 'loss_train_val')
        for rank in ["first", "second", "third"]: self.plotAccVsEpoch('%s_output_loss'%(rank), 'val_%s_output_loss'%(rank), '%s output loss'%(rank), '%s_output_loss_train_val'%(rank))        
        #for rank in ["first", "second", "third", "fourth"]: self.plotAccVsEpoch('%s_output_loss'%(rank), 'val_%s_output_loss'%(rank), '%s output loss'%(rank), '%s_output_loss_train_val'%(rank))        
        self.plotAccVsEpoch('correlation_layer_loss', 'val_correlation_layer_loss', 'correlation_layer output loss', 'correlation_layer_loss_train_val')

        # Plot disc per njet
        self.plotDiscPerNjet("_Disc1", {"Bg": [self.trainBg, y_Train_Bg_disc1, self.trainBg["Weight"]], "Sg": [self.trainSg, y_Train_Sg_disc1, self.trainSg["Weight"]]}, nBins=nBins)
        self.plotDiscPerNjet("_Disc2", {"Bg": [self.trainBg, y_Train_Bg_disc2, self.trainBg["Weight"]], "Sg": [self.trainSg, y_Train_Sg_disc2, self.trainSg["Weight"]]}, nBins=nBins)

        if doFullVal:
            # Make arrays for possible values to cut on for both discriminant
            # starting at a minimum of 0.5 for each
            c1s = np.arange(0.50, 0.95, 0.45); c2s = np.arange(0.50, 0.95, 0.45)

            # Get number of background and signal counts for each A, B, C, D region for every possible combination of cuts on disc 1 and disc 2
            bc, sc = self.cutAndCount(c1s, c2s, y_Train_Bg_disc1, y_Train_Bg_disc2, self.trainBg["Weight"][:,0], y_Train_Sg_disc1[self.trainSg["mask_m%s"%(mass)]], y_Train_Sg_disc2[self.trainSg["mask_m%s"%(mass)]], self.trainSg["Weight"][:,0][self.trainSg["mask_m%s"%(mass)]])

            # Plot signal contamination versus background rejection
            self.plotBkgdRejVsSigCont(bc, sc, mass)

            # For a given signal fraction, figure out the cut on disc 1 and disc 2 that maximizes the significance
            c1, c2, significance, sigfrac = self.findDiscCut4SigFrac(bc, sc)
            if c1 != -1.0 and c2 != -1.0:
                closure, closureUnc = self.simpleClosureABCD(bc["A"][c1][c2], bc["B"][c1][c2], bc["C"][c1][c2], bc["D"][c1][c2], bc["A2"][c1][c2], bc["B2"][c1][c2], bc["C2"][c1][c2], bc["D2"][c1][c2])
                self.metric["ABCDclosure"] = abs(1.0 - closure)
                self.metric["ABCDclosureUnc"] = closureUnc

                self.config["Disc1"] = c1
                self.config["Disc2"] = c2
                self.config["Significance"] = significance
                self.config["SignalFrac"] = sigfrac

                self.config["outputDirSpec"] = self.config["outputDir"] + "/c1_%s_c2_%s_closure_%.2f_significance_%.2f"%(c1, c2, closure, significance)
                os.makedirs(self.config["outputDirSpec"])

                # Plot each discriminant for sig and background while making cut on other disc
                self.plotDiscWithCut(float(c2), y_Train_Bg_disc1, y_Train_Bg_disc2, self.trainBg["Weight"][:,0], y_Train_Sg_disc1[self.trainSg["mask_m%s"%(mass)]], y_Train_Sg_disc2[self.trainSg["mask_m%s"%(mass)]], self.trainSg["Weight"][:,0][self.trainSg["mask_m%s"%(mass)]], "1", "2", mass=mass, bins=nBins)
                self.plotDiscWithCut(float(c1), y_Train_Bg_disc2, y_Train_Bg_disc1, self.trainBg["Weight"][:,0], y_Train_Sg_disc2[self.trainSg["mask_m%s"%(mass)]], y_Train_Sg_disc1[self.trainSg["mask_m%s"%(mass)]], self.trainSg["Weight"][:,0][self.trainSg["mask_m%s"%(mass)]], "2", "1", mass=mass, bins=nBins)

                self.plotDiscWithCutCompare(float(c2), y_Train_Bg_disc1, y_Train_Bg_disc2, self.trainBg["Weight"][:,0], "1", "2", "BG", bins=10)
                self.plotDiscWithCutCompare(float(c2), y_Train_Sg_disc1[self.trainSg["mask_m%s"%(mass)]], y_Train_Sg_disc2[self.trainSg["mask_m%s"%(mass)]], self.trainSg["Weight"][:,0][self.trainSg["mask_m%s"%(mass)]], "1", "2", "SG", mass=mass, bins=10)

                self.plotDiscWithCutCompare(float(c1), y_Train_Bg_disc2, y_Train_Bg_disc1, self.trainBg["Weight"][:,0], "2", "1", "BG", bins=10)
                self.plotDiscWithCutCompare(float(c1), y_Train_Sg_disc2[self.trainSg["mask_m%s"%(mass)]], y_Train_Sg_disc1[self.trainSg["mask_m%s"%(mass)]], self.trainSg["Weight"][:,0][self.trainSg["mask_m%s"%(mass)]], "2", "1", "SG", mass=mass, bins=10)

                # Plot 2D of the discriminants
                self.plotDisc1vsDisc2(y_Train_Bg_disc1, y_Train_Bg_disc2, self.trainBg["Weight"][:,0], self.trainSg["Weight"][:,0], float(c1), float(c2), significance, "BG", nBins=nBins)
                self.plotDisc1vsDisc2(y_Train_Sg_disc1[self.trainSg["mask_m%s"%(mass)]], y_Train_Sg_disc2[self.trainSg["mask_m%s"%(mass)]], self.trainSg["Weight"][:,0][self.trainSg["mask_m%s"%(mass)]], self.trainSg["Weight"][:,0][self.trainSg["mask_m%s"%(mass)]], float(c1), float(c2), significance, "SG", nBins=nBins, mass=mass)

        # Plot validation roc curve
        fpr_Val_disc1, tpr_Val_disc1, thresholds_Val_disc1 = roc_curve(valData["labels"][:,0], y_Val_disc1, sample_weight=valData["Weight"][:,0])
        fpr_Val_disc2, tpr_Val_disc2, thresholds_Val_disc2 = roc_curve(valData["labels"][:,0], y_Val_disc2, sample_weight=valData["Weight"][:,0])
        fpr_Train_disc1, tpr_Train_disc1, thresholds_Train_disc1 = roc_curve(self.trainData["labels"][:,0], y_Train_disc1, sample_weight=self.trainData["Weight"][:,0])
        fpr_Train_disc2, tpr_Train_disc2, thresholds_Train_disc2 = roc_curve(self.trainData["labels"][:,0], y_Train_disc2, sample_weight=self.trainData["Weight"][:,0])
        fpr_OTrain, tpr_OTrain, thresholds_OTrain = roc_curve(trainOData["labels"][:,0], y_OTrain, sample_weight=trainOData["Weight"][:,0])
        auc_Val_disc1 = roc_auc_score(valData["labels"][:,0], y_Val_disc1)
        auc_Val_disc2 = roc_auc_score(valData["labels"][:,0], y_Val_disc2)
        auc_Train_disc1 = roc_auc_score(self.trainData["labels"][:,0], y_Train_disc1)
        auc_Train_disc2 = roc_auc_score(self.trainData["labels"][:,0], y_Train_disc2)
        auc_OTrain = roc_auc_score(trainOData["labels"][:,0], y_OTrain)
        
        # Define metrics for the training
        self.metric["OverTrain_Disc1"] = abs(auc_Val_disc1 - auc_Train_disc1)
        self.metric["OverTrain_Disc2"] = abs(auc_Val_disc2 - auc_Train_disc2)
        self.metric["Performance_Disc1"] = abs(1 - auc_Train_disc1)
        self.metric["Performance_Disc2"] = abs(1 - auc_Train_disc2)

        # Plot some ROC curves
        self.plotROC("_Disc1", None, None, fpr_Val_disc1, tpr_Val_disc1, fpr_Train_disc1, tpr_Train_disc1, auc_Val_disc1, auc_Train_disc1)
        self.plotROC("_Disc2", None, None, fpr_Val_disc2, tpr_Val_disc2, fpr_Train_disc2, tpr_Train_disc2, auc_Val_disc2, auc_Train_disc2)
        self.plotROC("_TT_TTJets", None, None, fpr_OTrain, tpr_OTrain, fpr_Train_disc1, tpr_Train_disc1, auc_OTrain, auc_Train_disc1)
        self.plotROC("_"+self.config["ttbarMC"][0]+"_nJet_disc1", y_Train_disc1, self.trainData)
        self.plotROC("_"+self.config["ttbarMC"][0]+"_nJet_disc2", y_Train_disc2, self.trainData)
        self.plotROC("_"+self.config["otherttbarMC"][0]+"_nJet", y_OTrain, trainOData) 

        # Plot validation precision recall
        precision_Val_disc1, recall_Val_disc1, _ = precision_recall_curve(valData["labels"][:,0], y_Val_disc1, sample_weight=valData["Weight"][:,0])
        precision_Train_disc1, recall_Train_disc1, _ = precision_recall_curve(self.trainData["labels"][:,0], y_Train_disc1, sample_weight=self.trainData["Weight"][:,0])
        ap_Val_disc1 = average_precision_score(valData["labels"][:,0], y_Val_disc1, sample_weight=valData["Weight"][:,0])
        ap_Train_disc1 = average_precision_score(self.trainData["labels"][:,0], y_Train_disc1, sample_weight=self.trainData["Weight"][:,0])
        
        self.plotPandR(precision_Val_disc1, recall_Val_disc1, precision_Train_disc1, recall_Train_disc1, ap_Val_disc1, ap_Train_disc1)
        
        # Plot NJet dependance
        binxl = self.config["minNJetBin"]
        binxh = self.config["maxNJetBin"] + 1
        numbin = binxh - binxl        
        self.plot2DVar(name="nJet", binxl=binxl, binxh=binxh, numbin=numbin, xIn=self.trainBg["nJet"][:,0], yIn=y_Train_Bg_disc1, nbiny=50)
        
        # Save useful stuff
        self.trainData["y"] = y_Train_disc1
        np.save(self.config["outputDir"]+"/deepESMbin_dis_nJet.npy", self.trainData)
        
        for key in self.metric:
            print(key, self.metric[key])
        
        self.config["metric"] = self.metric
        with open(self.config["outputDir"]+"/config.json",'w') as configFile:
            json.dump(self.config, configFile, indent=4, sort_keys=True)

        return self.metric
