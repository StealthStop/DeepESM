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

    def __init__(self, model, config, sgTrainSet, trainData, trainSg, trainBg, result_log=None):
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
        fig.savefig(self.config["outputDir"]+"/"+name+"_discriminator.pdf", dpi=fig.dpi) 

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
        fig.savefig(self.config["outputDir"]+"/%s.png"%(name), dpi=fig.dpi)        
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
 
        if Njets == -1: fig.savefig(self.config["outputDir"]+"/Disc%s_BvsS_m%s.png"%(tag1,mass))
        else:           fig.savefig(self.config["outputDir"]+"/Disc%s_BvsS_m%s_Njets%d.png"%(tag1,mass,Njets))

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

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/%s%s_Disc%s_Compare_Shapes.png"%(tag3, mass, tag1))
        else:           fig.savefig(self.config["outputDir"]+"/%s%s_Njets%d_Disc%s_Compare_Shapes.png"%(tag3, mass, Njets, tag1))

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
        if Njets == -1: fig.savefig(self.config["outputDir"]+"/2D_SigVsBkgd_Disc1VsDisc2_m%s.png"%(mass), dpi=fig.dpi)        
        else:           fig.savefig(self.config["outputDir"]+"/2D_SigVsBkgd_Disc1VsDisc2_m%s_Njets%d.png"%(mass,Njets), dpi=fig.dpi)  
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
    def plotDisc1vsDisc2(self, disc1, disc2, bw, sw, c1, c2, significance, tag, mass = "", Njets = -1, nBins = 100):
        fig = plt.figure() 
        corr = 999.0
        try: corr = cor.pearson_corr(disc1, disc2)
        except: print("Correlation coefficient could not be calculated!")
        plt.hist2d(disc1, disc2, bins=[nBins, nBins], range=[[0, 1], [0, 1]], cmap=plt.cm.jet, weights=bw, cmin = sw.min())
        plt.colorbar()
        ax = plt.gca()
        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="red", linewidth=2); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="red", linewidth=2)
        ax.add_line(l1); ax.add_line(l2)
        ax.set_ylabel("Disc. 2"); ax.set_xlabel("Disc. 1")
        plt.text(0.05, 0.90, r"$\bf{CC}$ = %.3f"%(corr), fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))
        plt.text(0.05, 0.95, r"$\bf{Significance}$ = %.3f"%(significance), fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))
        hep.cms.label(data=True, paper=False, year=self.config["year"])

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/2D_%s%s_Disc1VsDisc2.png"%(tag,mass), dpi=fig.dpi)
        else:           fig.savefig(self.config["outputDir"]+"/2D_%s%s_Njets%d_Disc1VsDisc2.png"%(tag,mass,Njets), dpi=fig.dpi)

        return corr

    def cutAndCount(self, c1s, c2s, b1, b2, bw, s1, s2, sw, cdiff = 0.2):
        # First get the total counts in region "D" for all possible c1, c2
        bcounts = {"A" : {}, "B" : {}, "C" : {}, "D" : {}, "A2" : {}, "B2" : {}, "C2" : {}, "D2" : {}}  
        scounts = {"A" : {}, "B" : {}, "C" : {}, "D" : {}, "A2" : {}, "B2" : {}, "C2" : {}, "D2" : {}}  

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

                mask1BGT = np.ma.masked_where(b1>c1, b1).mask; mask1BLT = ~mask1BGT
                mask1SGT = np.ma.masked_where(s1>c1, s1).mask; mask1SLT = ~mask1SGT

                mask2BGT = np.ma.masked_where(b2>c2, b2).mask; mask2BLT = ~mask2BGT
                mask2SGT = np.ma.masked_where(s2>c2, s2).mask; mask2SLT = ~mask2SGT

                maskBA = mask1BGT&mask2BGT; maskSA = mask1SGT&mask2SGT
                maskBB = mask1BGT&mask2BLT; maskSB = mask1SGT&mask2SLT
                maskBC = mask1BLT&mask2BGT; maskSC = mask1SLT&mask2SGT
                maskBD = mask1BLT&mask2BLT; maskSD = mask1SLT&mask2SLT

                bcounts["A"][c1k][c2k]  = np.sum(bw[maskBA])
                bcounts["A2"][c1k][c2k] = np.sum(np.square(bw[maskBA]))

                bcounts["B"][c1k][c2k]  = np.sum(bw[maskBB])
                bcounts["B2"][c1k][c2k] = np.sum(np.square(bw[maskBB]))

                bcounts["C"][c1k][c2k]  = np.sum(bw[maskBC])
                bcounts["C2"][c1k][c2k] = np.sum(np.square(bw[maskBC]))

                bcounts["D"][c1k][c2k]  = np.sum(bw[maskBD])
                bcounts["D2"][c1k][c2k] = np.sum(np.square(bw[maskBD]))

                scounts["A"][c1k][c2k]  = np.sum(sw[maskSA])           
                scounts["A2"][c1k][c2k] = np.sum(np.square(sw[maskSA]))
                                                                       
                scounts["B"][c1k][c2k]  = np.sum(sw[maskSB])           
                scounts["B2"][c1k][c2k] = np.sum(np.square(sw[maskSB]))
                                                                       
                scounts["C"][c1k][c2k]  = np.sum(sw[maskSC])           
                scounts["C2"][c1k][c2k] = np.sum(np.square(sw[maskSC]))
                                                                       
                scounts["D"][c1k][c2k]  = np.sum(sw[maskSD])           
                scounts["D2"][c1k][c2k] = np.sum(np.square(sw[maskSD]))

        return bcounts, scounts

    def findDiscCut4SigFrac(self, bcts, scts, minFracB = 0.1, minFracS = 0.1):
        # Now calculate signal fraction and significance 
        # Pick c1 and c2 that give 30% sig fraction and maximizes significance
        significance = 0.0; sigFrac = 0.0; finalc1 = -1.0; finalc2 = -1.0; 
        closureErr = 0.0; metric = 999.0
        for c1k, c2s in bcts["A"].items():
            for c2k, temp in c2s.items():

                bA = bcts["A"][c1k][c2k]; bB = bcts["B"][c1k][c2k]; bC = bcts["C"][c1k][c2k]; bD = bcts["D"][c1k][c2k]
                sA = scts["A"][c1k][c2k]; sB = scts["B"][c1k][c2k]; sC = scts["C"][c1k][c2k]; sD = scts["D"][c1k][c2k]

                tempsigfrac = -1.0 
                if bA + sA > 0.0: tempsigfrac = sA / (sA + bA)

                bfracA = bA / (bA + bB + bC + bD)
                sfracA = sA / (sA + sB + sC + sD)

                bfracB = bB / (bA + bB + bC + bD)
                sfracB = sB / (sA + sB + sC + sD)

                bfracC = bC / (bA + bB + bC + bD)
                sfracC = sC / (sA + sB + sC + sD)

                bfracD = bD / (bA + bB + bC + bD)
                sfracD = sD / (sA + sB + sC + sD)

                # Minimum signal fraction requirement
                if bfracA > minFracB and sfracA > minFracS and \
                   bfracB > minFracB and sfracB > minFracS and \
                   bfracC > minFracB and sfracC > minFracS and \
                   bfracD > minFracB and sfracD > minFracS:

                    sigFrac = tempsigfrac

                    tempsignificance = 0.0; tempclosureerr = 999.0; tempmetric = 999.0
                    if bA > 0.0: tempsignificance += (sA / (bA + (0.3*bA)**2.0)**0.5)**2.0
                    if bB > 0.0: tempsignificance += (sB / (bB + (0.3*bB)**2.0)**0.5)**2.0
                    if bC > 0.0: tempsignificance += (sC / (bC + (0.3*bC)**2.0)**0.5)**2.0
                    if bD > 0.0: tempsignificance += (sD / (bD + (0.3*bD)**2.0)**0.5)**2.0

                    if bD > 0.0 and bA > 0.0: tempclosureerr = abs(1.0 - (bB * bC) / (bA * bD))

                    tempsignificance = tempsignificance**0.5

                    if tempsignificance > 0.0:
                        tempmetric = tempclosureerr / tempsignificance
 
                    # Save significance if we found a better one
                    #if tempsignificance > significance:
                    #    finalc1 = c1k; finalc2 = c2k
                    #    significance = tempsignificance

                    if tempmetric < metric:
                        finalc1 = c1k; finalc2 = c2k
                        metric = tempmetric
                        significance = tempsignificance
                        closureErr = tempclosureerr
                
        return finalc1, finalc2, significance, sigFrac

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

                closure, closureErr, _, _ = self.simpleClosureABCD(bc["A"][c1][c2], bc["B"][c1][c2], bc["C"][c1][c2], bc["D"][c1][c2], bc["A2"][c1][c2]**0.5, bc["B2"][c1][c2]**0.5, bc["C2"][c1][c2]**0.5, bc["D2"][c1][c2]**0.5) 
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

    # Member function to plot input variable for events in specific region 
    def plotInputVar(self, c1, c2, d1, d2, d, s, massMask, var, varLabel, process, mass = "", Njets=-1, bins=100, arange=(0,1)):
    
        iVar = self.config["allVars"].index(var)
        mask1GT = np.ma.masked_where(d1>c1, d1).mask; mask1LT = ~mask1GT
        mask2GT = np.ma.masked_where(d2>c2, d2).mask; mask2LT = ~mask2GT

        maskA = mask1GT&mask2GT
        maskB = mask1LT&mask2GT
        maskC = mask1LT&mask2LT
        maskD = mask1GT&mask2LT

        dw = d["Weight"][:,0]; sw = s["Weight"][:,0][massMask]
        v  = d["data"][:,iVar]; vS = s["data"][:,iVar][massMask]
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

        if Njets == -1: fig.savefig(self.config["outputDir"]+"/%s_%s%s_Compare_Shapes.png"%(var, process, mass))
        else:           fig.savefig(self.config["outputDir"]+"/%s_%s%s_Njets%d_Compare_Shapes.png"%(var, process, mass, Njets))

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

        fig.savefig(self.config["outputDir"]+"/Njets_Region_%s"%(label))

        plt.close(fig)

        return sign

    def plotNjetsClosure(self, bkgd, bkgdUnc, bkgdPred, bkgdPredUnc):

        newB        = [bkgd[0]];        newB        += bkgd
        newBPred    = [bkgdPred[0]];    newBPred    += bkgdPred

        binEdges = [i for i in range(0, len(newB))]
        errX     = [i+0.5 for i in range(0, len(bkgd))]

        fig = plt.figure()
        ax = plt.gca()
        ax.set_yscale("log")

        plt.step(binEdges, newB,     label="Background",           where="pre", color="black", linestyle="dashed")
        plt.step(binEdges, newBPred, label="Predicted Background", where="pre", color="red",   linestyle="solid")
        plt.errorbar(errX, bkgdPred, yerr=bkgdPredUnc, xerr=None, fmt='', ecolor="red", elinewidth=2, color=None, lw=0) 
        plt.errorbar(errX, bkgd,     yerr=bkgdUnc,    xerr=None, fmt='', ecolor="black", elinewidth=2, color=None, lw=0) 

        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)

        plt.xlabel('Number of jets')
        plt.ylabel('Events')
        plt.legend(loc='best')
        #plt.text(0.05, 0.94, r"Closure = %.2f"%(closure), transform=ax.transAxes, fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))

        fig.savefig(self.config["outputDir"]+"/Njets_Region_A_PredVsActual")

        plt.close(fig)

    def makePlots(self, doQuickVal=False, valMass="400"):
        NJetsRange = range(self.config["minNJetBin"], self.config["maxNJetBin"]+1)
        massMask = self.trainSg["mask_m%s"%(valMass)]

        sgValSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+"*Val.root") for mass in self.config["massModels"]) , [])
        bgValSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+ttbar+"*Val.root") for ttbar in self.config["ttbarMC"][1]), [])

        valData, valSg, valBg = get_data(sgValSet, bgValSet, self.config)

        output_Val, output_Val_Sg, output_Val_Bg = self.getOutput(self.model, valData["data"], valSg["data"], valBg["data"])
        output_Train, output_Train_Sg, output_Train_Bg = self.getOutput(self.model, self.trainData["data"], self.trainSg["data"], self.trainBg["data"])

        y_Val_disc1, y_Val_Sg_disc1, y_Val_Bg_disc1 = self.getResults(output_Val, output_Val_Sg, output_Val_Bg, outputNum=0, columnNum=0)
        y_Val_disc2, y_Val_Sg_disc2, y_Val_Bg_disc2 = self.getResults(output_Val, output_Val_Sg, output_Val_Bg, outputNum=0, columnNum=2)
        y_Val_mass, y_Val_mass_Sg, y_Val_mass_Bg = self.getResults(output_Val, output_Val_Sg, output_Val_Bg, outputNum=3, columnNum=0)
        y_Train_disc1, y_Train_Sg_disc1, y_Train_Bg_disc1 = self.getResults(output_Train, output_Train_Sg, output_Train_Bg, outputNum=0, columnNum=0)
        y_Train_disc2, y_Train_Sg_disc2, y_Train_Bg_disc2 = self.getResults(output_Train, output_Train_Sg, output_Train_Bg, outputNum=0, columnNum=2)
        y_Train_mass, y_Train_mass_Sg, y_Train_mass_Bg = self.getResults(output_Train, output_Train_Sg, output_Train_Bg, outputNum=3, columnNum=0)

        nBins = 20
        nBinsReg = 50
        masses = [300, 400, 550, 850, 1200]
        colors = ["red", "green", "blue", "magenta", "orange", "black"]; labels = ["Sg Train", "Sg Val", "Bg Train", "Bg Val"]

        self.plotDisc([y_Train_mass_Sg, y_Val_mass_Sg, y_Train_mass_Bg, y_Val_mass_Bg], colors, labels, [self.trainSg["Weight"], valSg["Weight"], self.trainBg["Weight"], valBg["Weight"]], "mass", 'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg)
        self.plotDisc([y_Train_mass_Sg, y_Val_mass_Sg, y_Train_mass_Bg, y_Val_mass_Bg], colors, labels, [self.trainSg["Weight"], valSg["Weight"], self.trainBg["Weight"], valBg["Weight"]], "mass_log", 'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg, doLog=True)

        tempColors = ["black"]; tempNames = ["ttbar"]; tempMass = [y_Train_mass_Bg]; tempEvents = [self.trainBg["Weight"]]
        i = 0
        for imass in masses:
            if imass >= self.config["minStopMass"] and imass <= self.config["maxStopMass"]:
                mask = "mask_m%d"%(imass)
                self.plotDisc([y_Train_mass_Sg[self.trainSg[mask]], y_Val_mass_Sg[valSg[mask]]], colors, labels, [self.trainSg["Weight"][self.trainSg[mask]], valSg["Weight"][valSg[mask]]], "mass_%d"%(imass), 'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg)

                tempColors.append(colors[i])
                tempNames.append("mass %d"%(imass))
                tempMass.append(y_Train_mass_Sg[self.trainSg[mask]])
                tempEvents.append(self.trainSg["Weight"][self.trainSg[mask]])
                i += 1

        self.plotDisc([y_Train_Sg_disc1, y_Val_Sg_disc1, y_Train_Bg_disc1, y_Val_Bg_disc1], colors, labels, [self.trainSg["Weight"], valSg["Weight"], self.trainBg["Weight"], valBg["Weight"]], "Disc1", 'Norm Events', 'Disc. 1')
        self.plotDisc([y_Train_Sg_disc2, y_Val_Sg_disc2, y_Train_Bg_disc2, y_Val_Bg_disc2], colors, labels, [self.trainSg["Weight"], valSg["Weight"], self.trainBg["Weight"], valBg["Weight"]], "Disc2", 'Norm Events', 'Disc. 2')

        self.plotDisc(tempMass, tempColors, tempNames, tempEvents, "mass_split", 'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg)

        self.plotDisc(tempMass, tempColors, tempNames, tempEvents, "mass_split", 'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg, doLog=True)

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
        self.plotDiscPerNjet("_Disc1", {"Bg": [self.trainBg, y_Train_Bg_disc1, self.trainBg["Weight"]], "Sg": [self.trainSg, y_Train_Sg_disc1, self.trainSg["Weight"]]}, nBins=nBins)
        self.plotDiscPerNjet("_Disc2", {"Bg": [self.trainBg, y_Train_Bg_disc2, self.trainBg["Weight"]], "Sg": [self.trainSg, y_Train_Sg_disc2, self.trainSg["Weight"]]}, nBins=nBins)
        
        if not doQuickVal:
            self.plotD1VsD2SigVsBkgd(y_Train_Bg_disc1, y_Train_Bg_disc2, y_Train_Sg_disc1[massMask], y_Train_Sg_disc2[massMask], valMass)
            # Make arrays for possible values to cut on for both discriminant
            # starting at a minimum of 0.5 for each
            c1s = np.arange(0.15, 0.85, 0.05); c2s = np.arange(0.15, 0.85, 0.05)
        
            # Get number of background and signal counts for each A, B, C, D region for every possible combination of cuts on disc 1 and disc 2
            bc, sc = self.cutAndCount(c1s, c2s, y_Train_Bg_disc1, y_Train_Bg_disc2, self.trainBg["Weight"][:,0], y_Train_Sg_disc1[massMask], y_Train_Sg_disc2[massMask], self.trainSg["Weight"][:,0][massMask])
        
            # Plot signal contamination versus background rejection
            self.plotBkgdRejVsSigCont(bc, sc, valMass)
        
            # For a given signal fraction, figure out the cut on disc 1 and disc 2 that maximizes the significance
            c1, c2, significance, sigfrac = self.findDiscCut4SigFrac(bc, sc)
        
            self.config["Disc1"] = c1
            self.config["Disc2"] = c2
            self.config["Significance"] = significance
            self.config["SignalFrac"] = sigfrac
        
            for i in range(len(self.config["allVars"])):
                theVar = self.config["allVars"][i]
                self.plotInputVar(float(c1), float(c2), y_Train_Bg_disc1, y_Train_Bg_disc2, self.trainBg, self.trainSg, massMask, theVar, theVar, "BG", mass = "", Njets = -1, bins = 64)

            # Plot each discriminant for sig and background while making cut on other disc
            self.plotDiscWithCut(float(c2), y_Train_Bg_disc1, y_Train_Bg_disc2, self.trainBg["Weight"][:,0], y_Train_Sg_disc1[massMask], y_Train_Sg_disc2[massMask], self.trainSg["Weight"][:,0][massMask], "1", "2", mass=valMass, Njets=-1, bins=nBins)
            self.plotDiscWithCut(float(c1), y_Train_Bg_disc2, y_Train_Bg_disc1, self.trainBg["Weight"][:,0], y_Train_Sg_disc2[massMask], y_Train_Sg_disc1[massMask], self.trainSg["Weight"][:,0][massMask], "2", "1", mass=valMass, Njets=-1, bins=nBins)
        
            self.plotDiscWithCutCompare(float(c2), y_Train_Bg_disc1, y_Train_Bg_disc2, self.trainBg["Weight"][:,0], "1", "2", "BG", mass="", Njets=-1, bins=10)
            self.plotDiscWithCutCompare(float(c2), y_Train_Sg_disc1[massMask], y_Train_Sg_disc2[massMask], self.trainSg["Weight"][:,0][massMask], "1", "2", "SG", mass=valMass, Njets=-1, bins=10)
        
            self.plotDiscWithCutCompare(float(c1), y_Train_Bg_disc2, y_Train_Bg_disc1, self.trainBg["Weight"][:,0], "2", "1", "BG", bins=10)
            self.plotDiscWithCutCompare(float(c1), y_Train_Sg_disc2[massMask], y_Train_Sg_disc1[massMask], self.trainSg["Weight"][:,0][massMask], "2", "1", "SG", mass=valMass, Njets=-1, bins=10)
        
            # Plot 2D of the discriminants
            self.plotDisc1vsDisc2(y_Train_Bg_disc1, y_Train_Bg_disc2, self.trainBg["Weight"][:,0], self.trainSg["Weight"][:,0], float(c1), float(c2), significance, "BG")
            self.plotDisc1vsDisc2(y_Train_Sg_disc1[massMask], y_Train_Sg_disc2[massMask], self.trainSg["Weight"][:,0][massMask], self.trainSg["Weight"][:,0][massMask], float(c1), float(c2), significance, "SG", mass=valMass)

            bkgdNjets = {"A" : [], "B" : [], "C" : [], "D" : []}; sigNjets = {"A" : [], "B" : [], "C" : [], "D" : []}
            bkgdNjetsErr = {"A" : [], "B" : [], "C" : [], "D" : []}; sigNjetsErr = {"A" : [], "B" : [], "C" : [], "D" : []}
            bkgdNjetsAPred = {"val" : [], "err" : []}
            for NJets in NJetsRange:
            
                njetsStr = "mask_nJet_%s"%(("%s"%(NJets)).zfill(2))
                bkgNjetsMask = self.trainBg[njetsStr]; sigNjetsMask = self.trainSg[njetsStr]
                bkgFullMask  = bkgNjetsMask;           sigFullMask  = massMask & sigNjetsMask

                # Get number of background and signal counts for each A, B, C, D region for every possible combination of cuts on disc 1 and disc 2
                bc, sc = self.cutAndCount(c1s, c2s, y_Train_Bg_disc1[bkgFullMask], y_Train_Bg_disc2[bkgFullMask], self.trainBg["Weight"][:,0][bkgFullMask], y_Train_Sg_disc1[sigFullMask], y_Train_Sg_disc2[sigFullMask], self.trainSg["Weight"][:,0][sigFullMask])
                c1, c2, significance, sigfrac = self.findDiscCut4SigFrac(bc, sc)
                if c1 == -1.0 or c2 == -1.0:
                    bkgdNjets["A"].append(0.0); sigNjets["A"].append(0.0)
                    bkgdNjets["B"].append(0.0); sigNjets["B"].append(0.0)
                    bkgdNjets["C"].append(0.0); sigNjets["C"].append(0.0)
                    bkgdNjets["D"].append(0.0); sigNjets["D"].append(0.0)

                    bkgdNjetsErr["A"].append(0.0); sigNjetsErr["A"].append(0.0)
                    bkgdNjetsErr["B"].append(0.0); sigNjetsErr["B"].append(0.0)
                    bkgdNjetsErr["C"].append(0.0); sigNjetsErr["C"].append(0.0)
                    bkgdNjetsErr["D"].append(0.0); sigNjetsErr["D"].append(0.0)

                    bkgdNjetsAPred["val"].append(0.0); bkgdNjetsAPred["err"].append(0.0)

                else:
                    closure, closureUnc, Apred, ApredUnc = self.simpleClosureABCD(bc["A"][c1][c2], bc["B"][c1][c2], bc["C"][c1][c2], bc["D"][c1][c2], bc["A2"][c1][c2]**0.5, bc["B2"][c1][c2]**0.5, bc["C2"][c1][c2]**0.5, bc["D2"][c1][c2]**0.5)
                    bkgdNjets["A"].append(bc["A"][c1][c2]); sigNjets["A"].append(sc["A"][c1][c2])
                    bkgdNjets["B"].append(bc["B"][c1][c2]); sigNjets["B"].append(sc["B"][c1][c2])
                    bkgdNjets["C"].append(bc["C"][c1][c2]); sigNjets["C"].append(sc["C"][c1][c2])
                    bkgdNjets["D"].append(bc["D"][c1][c2]); sigNjets["D"].append(sc["D"][c1][c2])

                    bkgdNjetsErr["A"].append(bc["A2"][c1][c2]**0.5); sigNjetsErr["A"].append(sc["A2"][c1][c2]**0.5)
                    bkgdNjetsErr["B"].append(bc["B2"][c1][c2]**0.5); sigNjetsErr["B"].append(sc["B2"][c1][c2]**0.5)
                    bkgdNjetsErr["C"].append(bc["C2"][c1][c2]**0.5); sigNjetsErr["C"].append(sc["C2"][c1][c2]**0.5)
                    bkgdNjetsErr["D"].append(bc["D2"][c1][c2]**0.5); sigNjetsErr["D"].append(sc["D2"][c1][c2]**0.5)

                    bkgdNjetsAPred["val"].append(Apred); bkgdNjetsAPred["err"].append(ApredUnc)

                self.config["c1_nJet_%s"%(("%s"%(NJets)).zfill(2))] = c1
                self.config["c2_nJet_%s"%(("%s"%(NJets)).zfill(2))] = c2

                # Avoid completely masked Njets bins that makes below plotting
                # highly unsafe
                if not any(bkgNjetsMask) or not any(sigNjetsMask): continue

                self.plotD1VsD2SigVsBkgd(y_Train_Bg_disc1[bkgFullMask], y_Train_Bg_disc2[bkgFullMask], y_Train_Sg_disc1[sigFullMask], y_Train_Sg_disc2[sigFullMask], valMass, NJets)

                # Plot each discriminant for sig and background while making cut on other disc
                self.plotDiscWithCut(float(c2), y_Train_Bg_disc1[bkgFullMask], y_Train_Bg_disc2[bkgFullMask], self.trainBg["Weight"][:,0][bkgFullMask], y_Train_Sg_disc1[sigFullMask], y_Train_Sg_disc2[sigFullMask], self.trainSg["Weight"][:,0][sigFullMask], "1", "2", mass=valMass, Njets=NJets, bins=nBins)
                self.plotDiscWithCut(float(c1), y_Train_Bg_disc2[bkgFullMask], y_Train_Bg_disc1[bkgFullMask], self.trainBg["Weight"][:,0][bkgFullMask], y_Train_Sg_disc2[sigFullMask], y_Train_Sg_disc1[sigFullMask], self.trainSg["Weight"][:,0][sigFullMask], "2", "1", mass=valMass, Njets=NJets, bins=nBins)
            
                self.plotDiscWithCutCompare(float(c2), y_Train_Bg_disc1[bkgFullMask], y_Train_Bg_disc2[bkgFullMask], self.trainBg["Weight"][:,0][bkgFullMask], "1", "2", "BG", mass="", Njets=-1, bins=10)
                self.plotDiscWithCutCompare(float(c2), y_Train_Sg_disc1[sigFullMask], y_Train_Sg_disc2[sigFullMask], self.trainSg["Weight"][:,0][sigFullMask], "1", "2", "SG", mass=valMass, Njets=NJets, bins=10)
            
                self.plotDiscWithCutCompare(float(c1), y_Train_Bg_disc2[bkgFullMask], y_Train_Bg_disc1[bkgFullMask], self.trainBg["Weight"][:,0][bkgFullMask], "2", "1", "BG", mass="", Njets=-1, bins=10)
                self.plotDiscWithCutCompare(float(c1), y_Train_Sg_disc2[sigFullMask], y_Train_Sg_disc1[sigFullMask], self.trainSg["Weight"][:,0][sigFullMask], "2", "1", "SG", mass=valMass, Njets=NJets, bins=10)
            
                # Plot 2D of the discriminants
                bkgdCorr = self.plotDisc1vsDisc2(y_Train_Bg_disc1[bkgFullMask], y_Train_Bg_disc2[bkgFullMask], self.trainBg["Weight"][:,0][bkgFullMask], self.trainSg["Weight"][:,0][sigFullMask], float(c1), float(c2), significance, "BG", mass="",   Njets=NJets)
                self.plotDisc1vsDisc2(y_Train_Sg_disc1[sigFullMask], y_Train_Sg_disc2[sigFullMask], self.trainSg["Weight"][:,0][sigFullMask], self.trainSg["Weight"][:,0][sigFullMask], float(c1), float(c2), significance, "SG", mass=valMass, Njets=NJets)
                self.metric["bkgdCorr_nJet_%s"%(NJets)] = abs(bkgdCorr) 

            signA = self.plotNjets(bkgdNjets["A"], bkgdNjetsErr["A"], sigNjets["A"], sigNjetsErr["A"], "A")
            signB = self.plotNjets(bkgdNjets["B"], bkgdNjetsErr["B"], sigNjets["B"], sigNjetsErr["B"], "B")
            signC = self.plotNjets(bkgdNjets["C"], bkgdNjetsErr["C"], sigNjets["C"], sigNjetsErr["C"], "C")
            signD = self.plotNjets(bkgdNjets["D"], bkgdNjetsErr["D"], sigNjets["D"], sigNjetsErr["D"], "D")

            self.plotNjetsClosure(bkgdNjets["A"], bkgdNjetsErr["A"], bkgdNjetsAPred["val"], bkgdNjetsAPred["err"])

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
  
        # Plot validation roc curve
        fpr_Val_disc1, tpr_Val_disc1, thresholds_Val_disc1 = roc_curve(valData["labels"][:,0], y_Val_disc1, sample_weight=valData["Weight"][:,0])
        fpr_Val_disc2, tpr_Val_disc2, thresholds_Val_disc2 = roc_curve(valData["labels"][:,0], y_Val_disc2, sample_weight=valData["Weight"][:,0])
        fpr_Train_disc1, tpr_Train_disc1, thresholds_Train_disc1 = roc_curve(self.trainData["labels"][:,0], y_Train_disc1, sample_weight=self.trainData["Weight"][:,0])
        fpr_Train_disc2, tpr_Train_disc2, thresholds_Train_disc2 = roc_curve(self.trainData["labels"][:,0], y_Train_disc2, sample_weight=self.trainData["Weight"][:,0])
        auc_Val_disc1 = roc_auc_score(valData["labels"][:,0], y_Val_disc1)
        auc_Val_disc2 = roc_auc_score(valData["labels"][:,0], y_Val_disc2)
        auc_Train_disc1 = roc_auc_score(self.trainData["labels"][:,0], y_Train_disc1)
        auc_Train_disc2 = roc_auc_score(self.trainData["labels"][:,0], y_Train_disc2)
        
        # Define metrics for the training
        self.metric["OverTrain_Disc1"] = abs(auc_Val_disc1 - auc_Train_disc1)
        self.metric["OverTrain_Disc2"] = abs(auc_Val_disc2 - auc_Train_disc2)
        self.metric["Performance_Disc1"] = abs(1 - auc_Train_disc1)
        self.metric["Performance_Disc2"] = abs(1 - auc_Train_disc2)
        if    self.config["TotalSignificance"] > 0.0: self.metric["InvTotalSignificance"] = 1.0/self.config["TotalSignificance"]
        else: self.metric["InvTotalSignificance"] = 999.0
       
        # Plot some ROC curves
        self.plotROC("_Disc1", None, None, fpr_Val_disc1, tpr_Val_disc1, fpr_Train_disc1, tpr_Train_disc1, auc_Val_disc1, auc_Train_disc1)
        self.plotROC("_Disc2", None, None, fpr_Val_disc2, tpr_Val_disc2, fpr_Train_disc2, tpr_Train_disc2, auc_Val_disc2, auc_Train_disc2)
        self.plotROC("_"+self.config["ttbarMC"][0]+"_nJet_disc1", y_Train_disc1, self.trainData)
        self.plotROC("_"+self.config["ttbarMC"][0]+"_nJet_disc2", y_Train_disc2, self.trainData)
        
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
        #np.save(self.config["outputDir"]+"/deepESMbin_dis_nJet.npy", self.trainData)
        
        for key in self.metric:
            print(key, self.metric[key])
        
        self.config["metric"] = self.metric
        with open(self.config["outputDir"]+"/config.json",'w') as configFile:
            json.dump(self.config, configFile, indent=4, sort_keys=True)

        return self.metric

