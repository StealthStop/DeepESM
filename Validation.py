from DataLoader import DataLoader
from Correlation import Correlation as cor

import gc
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
import matplotlib.cm as CM

import seaborn as sns
import pickle

import mplhep as hep
plt.style.use([hep.style.ROOT,hep.style.CMS]) # For now ROOT defaults to CMS
plt.style.use({'legend.frameon':False,'legend.fontsize':16,'legend.edgecolor':'black'})

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

import tracemalloc

import datetime
import math
#plt.rcParams['png.fonttype'] = 42

def timeStamp():
    return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

class Validation:

    def __init__(self, model, config, loader, valLoader, evalLoader, testLoader, result_log=None, do_LRP = False, kfold=False):
        self.model = model
        self.config = config
        self.result_log = result_log
        self.metric = {}
        self.doLog = False
        self.loader = loader
        self.valLoader = valLoader 
        self.evalLoader = evalLoader
        self.testLoader = testLoader
        self.kfold = kfold

        self.sample = {"RPV" : 100, "SYY" : 101, "SHH" : 102}

        self.do_LRP = do_LRP

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

    def getResults(self, output, output_sg, output_bg, outputNum=0, columnNum=0, sum=True):
        if sum:
            return output[outputNum][:,columnNum].ravel(), output_sg[outputNum][:,columnNum].ravel(), output_bg[outputNum][:,columnNum].ravel()
            #return output[outputNum][:,columnNum].ravel() + output[outputNum][:,columnNum+1].ravel() + output[outputNum][:,columnNum+2].ravel(), output_sg[outputNum][:,columnNum].ravel() + output_sg[outputNum][:,columnNum+1].ravel() + output_sg[outputNum][:,columnNum+2].ravel(), output_bg[outputNum][:,columnNum].ravel() + output_bg[outputNum][:,columnNum+1].ravel() + output_bg[outputNum][:,columnNum+2].ravel()
        else:
            return output[outputNum][:,columnNum].ravel(), output_sg[outputNum][:,columnNum].ravel(), output_bg[outputNum][:,columnNum].ravel()

    # Plot a set of 1D hists together, where the hists, colors, labels, weights
    # are provided as a list argument.
    def plotDisc(self, hists, colors, labels, weights, name, xlab, ylab, bins=100, arange=(0,1), doLog=False):
        # Plot predicted mass
        fig, ax = plt.subplots(figsize=(12, 12))
        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2", ax=ax)
        ax.set_ylabel(xlab); ax.set_xlabel(ylab)

        for i in range(0, len(hists)): 
            try:
                mean = round(np.average(hists[i], weights=weights[i]), 2)
                plt.hist(hists[i], bins=bins, range=arange, color="xkcd:"+colors[i], alpha=0.9, histtype='step', lw=2, label=labels[i]+" mean="+str(mean), density=True, log=doLog, weights=weights[i])
            except Exception as e:
                print("\nplotDisc: Could not plot %s hist for figure %s ::"%(labels[i],name), e, "\n")
                continue

        ax.legend(loc=1, frameon=False)
        fig.savefig(self.config["outputDir"]+"/%s.png"%(name), dpi=fig.dpi)        
        with open(self.config["outputDir"]+"/%s.pkl"%(name), 'wb') as f:
            pickle.dump(fig, f)
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

        fig = plt.figure(figsize=(12,12))

        ax = hep.histplot(h=bwnewBinned, bins=binEdges, w2=bw2newBinned, density=True, histtype="step", label="Background", alpha=0.9, lw=2)
        ax = hep.histplot(h=swnewBinned, bins=binEdges, w2=sw2newBinned, density=True, histtype="step", label="Signal (mass = %s GeV)"%(mass), alpha=0.9, lw=2, ax=ax)

        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2", ax=ax)
        ax.set_ylabel('A.U.'); ax.set_xlabel('Disc. %s'%(tag1))

        plt.text(0.05, 0.85, r"$\bf{Disc. %s}$ > %.3f"%(tag2,c), transform=ax.transAxes, fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))

        # Stupid nonsense to remove duplicate entries in legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-2:], labels[-2:], loc=2, frameon=False)
 
        if Njets == -1: 
            fig.savefig(self.config["outputDir"]+"/Disc%s_BvsS_m%s.png"%(tag1,mass))
            with open(self.config["outputDir"]+"/Disc%s_BvsS_m%s.pkl"%(tag1,mass), 'wb') as f:
                pickle.dump(fig, f)
        else:          
            fig.savefig(self.config["outputDir"]+"/Disc%s_BvsS_m%s_Njets%d.png"%(tag1,mass,Njets))
            with open(self.config["outputDir"]+"/Disc%s_BvsS_m%s_Njets%d.pkl"%(tag1,mass,Njets), 'wb') as f:
                pickle.dump(fig, f)

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

        fig = plt.figure(figsize=(12,12))

        ax = hep.histplot(h=dwgtBinned, bins=binEdges, w2=dw2gtBinned, density=True, histtype="step", label="Disc. %s > %.2f"%(tag2,c), alpha=0.9, lw=2)
        ax = hep.histplot(h=dwltBinned, bins=binEdges, w2=dw2ltBinned, density=True, histtype="step", label="Disc. %s < %.2f"%(tag2,c), alpha=0.9, lw=2, ax=ax)

        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2", ax=ax)
        ax.set_ylabel('A.U.'); ax.set_xlabel('Disc. %s'%(tag1))

        # Stupid nonsense to remove duplicate entries in legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-2:], labels[-2:], loc=2, frameon=False)

        if Njets == -1: 
            fig.savefig(self.config["outputDir"]+"/%s%s_Disc%s_Compare_Shapes.png"%(tag3, mass, tag1))
            with open(self.config["outputDir"]+"/%s%s_Disc%s_Compare_Shapes.pkl"%(tag3, mass, tag1), 'wb') as f:
                pickle.dump(fig, f)
        else:           
            fig.savefig(self.config["outputDir"]+"/%s%s_Njets%d_Disc%s_Compare_Shapes.png"%(tag3, mass, Njets, tag1))
            with open(self.config["outputDir"]+"/%s%s_Njets%d_Disc%s_Compare_Shapes.pkl"%(tag3, mass, Njets, tag1), 'wb') as f:
                pickle.dump(fig, f)

        plt.close(fig)

    # Plot loss of training vs test
    def plotAccVsEpoch(self, h1, h2, title, name):

        lambda_names = {'disc': 'disc_lambda', 'disco': 'bkg_disco_lambda', 'closure': 'abcd_close_lambda', 'mass_reg': 'mass_reg_lambda'}

        fig = plt.figure(figsize=(12,12))
        if not title.split(" ")[0]:
            #fig.axes[0].ticklabel_format(useOffset=False)
            plt.ticklabel_format(useOffset=False)
        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2")
        plt.plot(self.result_log.history[h1])
        plt.plot(self.result_log.history[h2])
        if title.split(" ")[0] == "mass_reg":
            plt.yscale("log")
        #plt.title(title, pad=45.0)
        plt.ylabel(title)
        plt.xlabel('Epoch')

        plt.legend(['train', 'test'], loc='best')
        fig.savefig(self.config["outputDir"]+"/%s.pdf"%(name), dpi=fig.dpi)
        with open(self.config["outputDir"]+"/%s.pkl"%(name), 'wb') as f:
            pickle.dump(fig, f)
        plt.close(fig)

    def plotAccVsEpochAll(self, h, n, val, title, name):
        fig = plt.figure(figsize=(12,12))
        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2")
        #plt.title(title, pad=45.0)
        plt.ylabel('training loss')
        plt.xlabel('epoch')

        l = []
        for H in h:
            plt.plot(self.result_log.history["%s%s_loss"%(val,H)])
            l.append(n[h.index(H)])

        plt.yscale("log")
        plt.legend(l, loc='best')
        fig.savefig(self.config["outputDir"]+"/%s.pdf"%(name), dpi=fig.dpi)
        with open(self.config["outputDir"]+"/%s.pkl"%(name), 'wb') as f:
            pickle.dump(fig, f)
        plt.close(fig)

    def plotDiscPerNjet(self, tag, samples, sigMask, nBins=100):
        for sample in samples:
            trainSample = samples[sample][0]
            y_train_Sp = samples[sample][1]
            weights = samples[sample][2] 
            bins = np.linspace(0, 1, nBins)
            fig, ax = plt.subplots(figsize=(12, 12))
            hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2")
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
            fig.savefig(self.config["outputDir"]+"/nJet_"+sample+tag+".png", dpi=fig.dpi)
            with open(self.config["outputDir"]+"/nJet_"+sample+tag+".pkl", 'wb') as f:
                pickle.dump(fig, f)
            plt.close(fig)

    def plotROC(self, dataMaskEval=None, dataMaskVal=None, tag="", y_eval=None, y_val=None, evalData=None, valData=None, xEval=None, xVal=None, yEval=None, yVal=None, evalLab=None, valLab=None, doMass=False, minMass=300, maxMass=1400, y_val_err=None):

        extra = None
        if "disc1" in tag or "Disc1" in tag: extra = "disc1"
        else:                                extra = "disc2"

        if extra not in self.config: self.config[extra] = {"eval_auc" : {}, "val_auc" : {}}

        fig = plt.figure(figsize=(12,12))
        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('', pad=45.0)

        if y_eval is None:
            plt.plot(xVal, yVal, color='xkcd:red', linestyle=":", label='Val (area = {:.3f})'.format(valLab))
            plt.plot(xEval, yEval, color='xkcd:red', label='Train (area = {:.3f})'.format(evalLab))
            self.config[extra]["eval_auc"]["total"] = evalLab
            self.config[extra]["val_auc"]["total"] = valLab

        # Adding a kfold cross-validation to add error bars to roc plots
        # if you want to run this, use --kfold when training
        elif self.kfold and doMass:

            i_mass = [m for m in range(minMass, maxMass, 200)]

            kf = KFold(n_splits=8, shuffle=True)

            for mass in i_mass:

                #try:        
                massMask     = evalData["mass"] == float(mass)
                massMask     |= evalData["mass"] == float(173.0)

                labels            = evalData["label"][dataMaskEval&massMask]
                weights           = evalData["weight"][dataMaskEval&massMask]

                y = y_eval[dataMaskEval&massMask]
                if len(y)==0:
                    continue

                res = []
                val_res = []
                auc_list = []
                val_auc_list = []
                first = None

                for i, (train_idx, test_idx) in enumerate(kf.split(labels)):
                    res.append(roc_curve(labels[train_idx], y[train_idx], sample_weight=weights[train_idx]))
                    auc_list.append(roc_auc_score(labels[train_idx], y[train_idx]))

                    val_res.append(roc_curve(labels[test_idx], y[test_idx], sample_weight=weights[test_idx]))
                    val_auc_list.append(roc_auc_score(labels[test_idx], y[test_idx]))

                    if first is None:
                        first = res[0][0]
                    
 
                fpr_eval = first #np.mean([res[i][0] for i in range(len(res))], axis=0)
                tpr_eval = np.mean([np.interp(first, res[i][0], res[i][1]) for i in range(len(res))], axis=0)
                auc_eval = np.mean(auc_list)

                fpr_val = first #np.mean([val_res[i][0] for i in range(len(val_res))], axis=0)
                tpr_val = np.mean([np.interp(first, val_res[i][0], val_res[i][1]) for i in range(len(val_res))], axis=0)
                auc_val = np.mean(val_auc_list)
                tpr_val_std = np.std([np.interp(first, val_res[i][0], val_res[i][1]) for i in range(len(val_res))], axis=0)

                color = next(plt.gca()._get_lines.prop_cycler)['color']
                plt.plot(fpr_eval, tpr_eval, label="$M_{\mathregular{\\tilde{t}}}$ = %d (Train)"%(int(mass)) + " (area = {:.3f})".format(auc_eval), color=color)
                plt.plot(fpr_val, tpr_val, linestyle=":", label="$M_{\mathregular{\\tilde{t}}}$ = %d (Val)"%(int(mass)) + " (area = {:.3f})".format(auc_val), color=color)
                plt.fill_between(fpr_val, tpr_val-tpr_val_std, tpr_val+tpr_val_std, alpha=0.3, color=color)

                self.config[extra]["eval_auc"]["Mass%d"%(int(mass))] = auc_eval
                #except Exception as e:
                #    print("\nplotROC: Could not plot ROC for Mass = %d ::"%(int(mass)), e, "\n")
                #    continue

        elif doMass:
            i_mass = [m for m in range(minMass, maxMass, 200)]
            for mass in i_mass:
                try:        
                    massMask     = evalData["mass"] == float(mass)
                    massMask     |= evalData["mass"] == float(173.0)

                    labels            = evalData["label"][dataMaskEval&massMask]
                    weights           = evalData["weight"][dataMaskEval&massMask]

                    y = y_eval[dataMaskEval&massMask]
                    if len(y)==0:
                        continue

                    fpr_eval, tpr_eval, thresholds_eval = roc_curve(labels, y, sample_weight=weights)
                    auc_eval = roc_auc_score(labels, y)    
                    plt.plot(fpr_eval, tpr_eval, label="$M_{\mathregular{\\tilde{t}}}$ = %d (Train)"%(int(mass)) + " (area = {:.3f})".format(auc_eval))

                    self.config[extra]["eval_auc"]["Mass%d"%(int(mass))] = auc_eval
                except Exception as e:
                    print("\nplotROC: Could not plot ROC for Mass = %d ::"%(int(mass)), e, "\n")
                    continue

            plt.gca().set_prop_cycle(None)
            for mass in i_mass:
                try:        
                    massMask     = valData["mass"] == float(mass)
                    massMask     |= valData["mass"] == float(173.0)

                    labels            = valData["label"][dataMaskVal&massMask]
                    weights           = valData["weight"][dataMaskVal&massMask]

                    y = y_val[dataMaskVal&massMask]
                    if len(y)==0:
                        continue

                    fpr_val, tpr_val, thresholds_val = roc_curve(labels, y, sample_weight=weights)
                    auc_val = roc_auc_score(labels, y)    
                    plt.plot(fpr_val, tpr_val, linestyle=":", label="$M_{\mathregular{\\tilde{t}}}$ = %d (Val)"%(int(mass)) + " (area = {:.3f})".format(auc_val))

                    self.config[extra]["val_auc"]["Mass%d"%(int(mass))] = auc_val
                except Exception as e:
                    print("\nplotROC: Could not plot ROC for Mass = %d ::"%(int(mass)), e, "\n")
                    continue
            
        else:
            NJetsRange = range(self.config["minNJetBin"], self.config["maxNJetBin"]+1)
            for NJets in NJetsRange:
                try:        
                    njets = float(NJets)
                    if self.config["Mask"] and (int(NJets) in self.config["Mask_nJet"]): continue

                    dataNjetsMaskEval = evalData["njets"]==njets
                    labels            = evalData["label"][dataMaskEval&dataNjetsMaskEval]
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
                    valLabels        = valData["label"][dataMaskVal&dataNjetsMaskVal]
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
        plt.grid()
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/roc_plot"+newtag+".png", dpi=fig.dpi)
        with open(self.config["outputDir"]+"/roc_plot"+newtag+".pkl", 'wb') as f:
            pickle.dump(fig, f)
        plt.close(fig)

    # Plot disc1 vs disc2 for both background and signal
    def plotD1VsD2SigVsBkgd(self, b1, b2, s1, s2, mass, Njets=-1):
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(111)
        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2", ax=ax1)
        ax1.scatter(b1, b2, s=10, c='b', marker="s", label='background')
        ax1.scatter(s1, s2, s=10, c='r', marker="o", label='signal (mass = %s GeV)'%(mass))
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel("Disc. 1")
        ax1.set_ylabel("Disc. 2")
        plt.legend(loc='best');
        if Njets == -1: 
            fig.savefig(self.config["outputDir"]+"/2D_SigVsBkgd_Disc1VsDisc2_m%s.png"%(mass), dpi=fig.dpi)        
            with open(self.config["outputDir"]+"/2D_SigVsBkgd_Disc1VsDisc2_m%s.pkl"%(mass), 'wb') as f:
                pickle.dump(fig, f)
        else:           
            fig.savefig(self.config["outputDir"]+"/2D_SigVsBkgd_Disc1VsDisc2_m%s_Njets%d.png"%(mass,Njets), dpi=fig.dpi)  
            with open(self.config["outputDir"]+"/2D_SigVsBkgd_Disc1VsDisc2_m%s_Njets%d.pkl"%(mass,Njets), 'wb') as f:
                pickle.dump(fig, f)
        plt.close(fig)

    def plotPandR(self, pval, rval, ptrain, rtrain, valLab, trainLab, name):
        fig = plt.figure(figsize=(12,12))
        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2")
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.plot(pval, rval, color='xkcd:black', label='Val (AP = {:.3f})'.format(valLab))
        plt.plot(ptrain, rtrain, color='xkcd:red', label='Train (AP = {:.3f})'.format(trainLab))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision and Recall curve', pad=45.0)
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/PandR_plot_{}.png".format(name), dpi=fig.dpi)        
        with open(self.config["outputDir"]+"/PandR_plot_{}.pkl".format(name), 'wb') as f:
            pickle.dump(fig, f)
        plt.close(fig)

    def plotPandR2D(self, labels, d1_val, d2_val, sample_weight=None, name=None):

        bins = []
        ncuts = 5
        cut = 1./float(ncuts)   

        for i in range(0,ncuts):
            cond = d1_val > cut * i
            cond &= d2_val > cut * i
            try:
                precision_val,  recall_val, thresholds = precision_recall_curve(labels[cond],   d2_val[cond],   sample_weight=sample_weight[cond])
                ap_val  = average_precision_score(labels[cond], d2_val[cond],  sample_weight=sample_weight[cond])
            except Exception as e:
                
                print(e)
                print("Probably not enough events in the last bin, possibly try rebinning for P and R 2D plot")            
                continue

            bins.append({})

            bins[i]["cut"] = cut * i
            bins[i]["pval"] = precision_val
            bins[i]["rval"] = recall_val
            bins[i]["apVal"] = ap_val

        fig = plt.figure(figsize=(12,12))
        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2")
        plt.ylim(0,1)
        plt.xlim(0,1)
        for x in bins:
            plt.plot(x["pval"], x["rval"], label='Val d1,d2 > {:.2f} (AP = {:.3f})'.format(x["cut"], x["apVal"]))
            #plt.plot(ptrain, rtrain, color='xkcd:red', label='Train (AP = {:.3f})'.format(trainLab))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision and Recall curve', pad=45.0)
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/PandR2D_plot_{}.png".format(name), dpi=fig.dpi)        
        with open(self.config["outputDir"]+"/PandR2D_plot_{}.pkl".format(name), 'wb') as f:
            pickle.dump(fig, f)
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
                maskBB = mask1BLT&mask2BGT; maskSB = mask1SLT&mask2SGT
                maskBC = mask1BGT&mask2BLT; maskSC = mask1SGT&mask2SLT
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

    def findABCDedges(self, bcts, scts, bkgNormUnc = 0.3, minBkgEvts = 5):
        # Now calculate signal fraction and significance 
        # Pick c1 and c2 that give 30% sig fraction and maximizes significance
        significance = 0.0; finalc1 = -1.0; finalc2 = -1.0; 
        closureErr = 0.0; metric = 999.0
        signs = []
        signsWNC = []
        predsigns = []
        closeErrs = []
        edges = []
        wBkgA   = []; uwBkgA  = []; wSigA   = []; uwSigA  = []
        wBkgB   = []; uwBkgB  = []; wSigB   = []; uwSigB  = []
        wBkgC   = []; uwBkgC  = []; wSigC   = []; uwSigC  = []
        wBkgD   = []; uwBkgD  = []; wSigD   = []; uwSigD  = []
        sFracsA = []; sFracsB = []; sFracsC = []; sFracsD = []
        normSigFracs = []

        for c1k, c2s in bcts["A"].items():
            for c2k, temp in c2s.items():

                bA = bcts["A"][c1k][c2k]; bB = bcts["B"][c1k][c2k]; bC = bcts["C"][c1k][c2k]; bD = bcts["D"][c1k][c2k]
                ba = bcts["a"][c1k][c2k]; bb = bcts["b"][c1k][c2k]; bc = bcts["c"][c1k][c2k]; bd = bcts["d"][c1k][c2k]
                sA = scts["A"][c1k][c2k]; sB = scts["B"][c1k][c2k]; sC = scts["C"][c1k][c2k]; sD = scts["D"][c1k][c2k]
                sa = scts["a"][c1k][c2k]; sb = scts["b"][c1k][c2k]; sc = scts["c"][c1k][c2k]; sd = scts["d"][c1k][c2k]

                bA2 = bcts["A2"][c1k][c2k]; bB2 = bcts["B2"][c1k][c2k]; bC2 = bcts["C2"][c1k][c2k]; bD2 = bcts["D2"][c1k][c2k]
                sA2 = scts["A2"][c1k][c2k]; sB2 = scts["B2"][c1k][c2k]; sC2 = scts["C2"][c1k][c2k]; sD2 = scts["D2"][c1k][c2k]

                bTotal = bA + bB + bC + bD
                sTotal = sA + sB + sC + sD

                tempsbfracA = -1.0; tempsbfracAunc = -1.0; tempsTotfracA = -1.0; tempbTotfracA = -1.0
                tempsbfracB = -1.0; tempsbfracBunc = -1.0; tempsTotfracB = -1.0; tempbTotfracB = -1.0 
                tempsbfracC = -1.0; tempsbfracCunc = -1.0; tempsTotfracC = -1.0; tempbTotfracC = -1.0 
                tempsbfracD = -1.0; tempsbfracDunc = -1.0; tempsTotfracD = -1.0; tempbTotfracD = -1.0 
                if bA + sA > 0.0:
                    tempsbfracA = sA / (sA + bA)
                    tempsbfracAunc = ((bA * sA2**0.5 / (sA + bA)**2.0)**2.0 + \
                                      (sA * bA2**0.5 / (sA + bA)**2.0)**2.0)**0.5
                if bB + sB > 0.0:
                    tempsbfracB = sB / (sB + bB)
                    tempsbfracBunc = ((bB * sB2**0.5 / (sB + bB)**2.0)**2.0 + \
                                      (sB * bB2**0.5 / (sB + bB)**2.0)**2.0)**0.5
                if bC + sC > 0.0:
                    tempsbfracC = sC / (sC + bC)
                    tempsbfracCunc = ((bC * sC2**0.5 / (sC + bC)**2.0)**2.0 + \
                                      (sC * bC2**0.5 / (sC + bC)**2.0)**2.0)**0.5
                if bD + sD > 0.0:
                    tempsbfracD = sD / (sD + bD)
                    tempsbfracDunc = ((bD * sD2**0.5 / (sD + bD)**2.0)**2.0 + \
                                      (sD * bD2**0.5 / (sD + bD)**2.0)**2.0)**0.5

                tempbfracA = bA / bTotal; tempsfracA = sA / sTotal
                tempbfracB = bB / bTotal; tempsfracB = sB / sTotal
                tempbfracC = bC / bTotal; tempsfracC = sC / sTotal
                tempbfracD = bD / bTotal; tempsfracD = sD / sTotal

                tempsignificance = 0.0; tempclosureerr = -999.0; tempmetric = 999.0; tempclosureerrunc = -999.0; tempsignunc = 0.0; temppredsign = 0.0
                tempsignificanceWNC = 0.0; tempsignuncWNC = 0.0

                if bD > 0.0 and bA > 0.0:
                    tempclosureerr    = abs(1.0 - (bB * bC) / (bA * bD))
                    tempclosureerrunc = (((bB2**0.5 * bC)/(bA * bD))**2.0 + \
                                         ((bB * bC2**0.5)/(bA * bD))**2.0 + \
                                         ((bB * bC * bA2**0.5)/(bA**2.0 * bD))**2.0 + \
                                         ((bB * bC * bD2**0.5)/(bA * bD**2.0))**2.0)**0.5

                if bA > 0.0:
                    tempsignificanceWNC += (sA / (bA + (bkgNormUnc*bA)**2.0 + (tempclosureerr*bA)**2.0)**0.5)
                    tempsignificance += (sA / (bA)**0.5)
                    temppredsign += (sA / (bB * bC / bD)**0.5)
                    tempsignuncWNC      += ((sA2**0.5 / (bA + (bkgNormUnc*bA)**2.0 + (tempclosureerr*bA)**2.0)**0.5)**2.0 + \
                                         ((sA * bA2**0.5 * (2.0 * bA * tempclosureerr**2.0 + 2.0 * bkgNormUnc**2.0 * bA + 1)) / (bA + (bkgNormUnc*bA)**2.0 + (tempclosureerr*bA)**2.0)**1.5)**2.0 + \
                                         ((bA**2.0 * tempclosureerr * sA * tempclosureerrunc) / (bA * (bA * (tempclosureerr**2.0 + bkgNormUnc**2.0) + 1))**1.5)**2.0)**0.5
                    tempsignunc      += tempsignificance * (((sA)**0.5/sA)**2 + (0.5)*(1/(bA**1.5)))**0.5
                                   
                if tempsignificance > 0.0 and tempclosureerr > 0.0:
                    signs.append([tempsignificance, tempsignunc])
                    signsWNC.append([tempsignificanceWNC, tempsignuncWNC])
                    predsigns.append([temppredsign, tempsignunc])
                    closeErrs.append([abs(tempclosureerr), tempclosureerrunc])
                    edges.append([float(c1k),float(c2k)])

                    wBkgA.append([bA, bA2**0.5]); wBkgB.append([bB, bB2**0.5])
                    uwBkgA.append([ba, ba**0.5]); uwBkgB.append([bb, bb**0.5])
                    wSigA.append([sA, sA2**0.5]); wSigB.append([sB, sB2**0.5])
                    uwSigA.append([sa, sa**0.5]); uwSigB.append([sb, sb**0.5])

                    wBkgC.append([bC, bC2**0.5]); wBkgD.append([bD, bD2**0.5])
                    uwBkgC.append([bc, bc**0.5]); uwBkgD.append([bd, bd**0.5])
                    wSigC.append([sC, sC2**0.5]); wSigD.append([sD, sD2**0.5])
                    uwSigC.append([sc, sc**0.5]); uwSigD.append([sd, sd**0.5])

                    sFracsA.append([float(tempsbfracA), float(tempsbfracAunc)])
                    sFracsB.append([float(tempsbfracB), float(tempsbfracBunc)])
                    sFracsC.append([float(tempsbfracC), float(tempsbfracCunc)])
                    sFracsD.append([float(tempsbfracD), float(tempsbfracDunc)])
                    normSigFracs.append([float(tempsbfracA)**-1 * (tempsbfracB + tempsbfracC - tempsbfracD)])

                # Compute metric if...
                # signal fraction in B, C, and D regions is < 10%
                # total background fraction in A is greater than 5%

                if ba > minBkgEvts and \
                   bb > minBkgEvts and \
                   bc > minBkgEvts and \
                   bd > minBkgEvts:

                    if tempsignificance > 0.0:
                        tempmetric = tempclosureerr**2.0 + (1.0 / tempsignificance)**2.0
                       #tempmetric = 1.0 / tempsignificance

                #if tempmetric < metric:
                if c1k == "0.60" and c2k == "0.60":

                    finalc1 = c1k; finalc2 = c2k
                    metric = tempmetric
                    significance = tempsignificance
                    closureErr = tempclosureerr
                
        return finalc1, finalc2, significance, closureErr, np.array(edges), np.array(signs), np.array(signsWNC), np.array(predsigns), np.array(closeErrs), {"A" : np.array(sFracsA), "B" : np.array(sFracsB), "C" : np.array(sFracsC), "D" : np.array(sFracsD)}, {"A" : np.array(wBkgA), "B" : np.array(wBkgB), "C" : np.array(wBkgC), "D" : np.array(wBkgD)}, {"A" : np.array(uwBkgA), "B" : np.array(uwBkgB), "C" : np.array(uwBkgC), "D" : np.array(uwBkgD)}, {"A" : np.array(wSigA), "B" : np.array(wSigB), "C" : np.array(wSigC), "D" : np.array(wSigD)}, {"A" : np.array(uwSigA), "B" : np.array(uwSigB), "C" : np.array(uwSigC), "D" : np.array(uwSigD)}, normSigFracs

    # Define closure as how far away prediction for region D is compared to actual 
    def predictABCD(self, bNB, bNC, bND, bNBerr, bNCerr, bNDerr):
        # Define A: > c1, > c2        B    |    A    
        # Define B: < c1, > c2   __________|__________        
        # Define C: > c1, < c2             |        
        # Define D: < c1, < c2        D    |    C    

        num = bNC * bNB

        bNApred = -1.0; bNApredUnc = 0.0
        if bND > 0.0:
            bNApred = num / bND
            bNApredUnc = ((bNC * bNBerr / bND)**2.0 + (bNCerr * bNB / bND)**2.0 + (bNC * bNB * bNDerr / bND**2.0)**2.0)**0.5

        return bNApred, bNApredUnc

    def countPeaks(self, disc1, disc2):
        def checkAround(hist, i, j, min=0, max=4):
            arr = hist[0]
            
            current = arr[i][j]
            
            left = current > arr[i-1][j] if i is not min else True
            right = current > arr[i+1][j] if i is not max else True
            up = current > arr[i][j-1] if j is not min else True
            down = current > arr[i][j+1] if j is not max else True

            return left and right and up and down 
        
        hist = np.histogram2d(disc1, disc2, bins=5, range=[[0,1],[0,1]], normed=None, weights=None, density=None)

        req = np.sum(hist[0]) * 0.1
        print(hist)

        nPeaks = 0
        i = 0
        for x in hist[0]:
            j = 0
            for y in x:
                if y > req and checkAround(hist, i, j):
                    nPeaks += 1
                j += 1
            i += 1

        print("I found {} peaks".format(nPeaks))


    def plotDisc1vsDisc2(self, disc1, disc2, bw, c1, c2, significance, tag, mass = "", Njets = -1, nBins = 100):
        #self.countPeaks(disc1, disc2)
        
        fig = plt.figure(figsize=(12,12)) 
        corr = 999.0
        try: corr = cor.pearson_corr(disc1, disc2)
        except: print("Correlation coefficient could not be calculated!")
        plt.hist2d(disc1, disc2, bins=[nBins, nBins], range=[[0, 1], [0, 1]], cmap=plt.cm.viridis, weights=bw, cmin = bw.min())#, norm=mpl.colors.LogNorm())
        plt.colorbar(label="Num. Events")
        ax = plt.gca()
        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="red", linewidth=2); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="red", linewidth=2)
        ax.add_line(l1); ax.add_line(l2)
        ax.set_ylabel("Disc. 2"); ax.set_xlabel("Disc. 1")
        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2")

        fig.tight_layout()
        if Njets == -1: 
            fig.savefig(self.config["outputDir"]+"/2D_%s%s_Disc1VsDisc2.png"%(tag,mass), dpi=fig.dpi)
            with open(self.config["outputDir"]+"/2D_%s%s_Disc1VsDisc2.pkl"%(tag,mass), 'wb') as f:
                pickle.dump(fig, f)
        else:           
            fig.savefig(self.config["outputDir"]+"/2D_%s%s_Disc1VsDisc2_Njets%d.png"%(tag,mass,Njets), dpi=fig.dpi)
            with open(self.config["outputDir"]+"/2D_%s%s_Disc1VsDisc2_Njets%d.pkl"%(tag,mass,Njets), 'wb') as f:
                pickle.dump(fig, f)

        plt.close(fig)

        return corr

    def plotVarVsBinEdges(self, var, edges, c1, c2, minEdge, maxEdge, edgeWidth, cmax, vmax, tag, Njets = -1):

        nBins = int((1.0 + edgeWidth)/edgeWidth)

        if tag == "NonClosure":
            lab_tag = "Non-Closure"
        elif tag == "Sign":
            lab_tag = "Significance"
        else:
            lab_tag = tag

        fig = plt.figure(figsize=(12,12)) 
        plt.hist2d(edges[:,0], edges[:,1], bins=[nBins, nBins], range=[[-edgeWidth/2.0, 1+edgeWidth/2.0], [-edgeWidth/2.0, 1+edgeWidth/2.0]], cmap=plt.cm.viridis, weights=var, cmin=10e-10, cmax=cmax, vmin = 0.0, vmax = vmax)
        cb = plt.colorbar(label=lab_tag)
        cb.set_label(label="{}".format(lab_tag), loc='center')
        ax = plt.gca()
        ax.set_ylabel("Disc. 2 Bin Edge"); ax.set_xlabel("Disc. 1 Bin Edge");
        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2")

        l1 = ml.Line2D([c1, c1], [0.0, 1.0], color="black", linewidth=2, linestyle="dashed"); l2 = ml.Line2D([0.0, 1.0], [c2, c2], color="black", linewidth=2, linestyle="dashed")
        ax.add_line(l1); ax.add_line(l2)
        fig.tight_layout()

        if Njets == -1: 
            fig.savefig(self.config["outputDir"]+"/%s_vs_Disc1Disc2.png"%(tag), dpi=fig.dpi)
            with open(self.config["outputDir"]+"/%s_vs_Disc1Disc2.pkl"%(tag), 'wb') as f:
                pickle.dump(fig, f)
        else:           
            fig.savefig(self.config["outputDir"]+"/%s_vs_Disc1Disc2_Njets%s.png"%(tag,Njets), dpi=fig.dpi)
            with open(self.config["outputDir"]+"/%s_vs_Disc1Disc2_Njets%s.pkl"%(tag,Njets), 'wb') as f:
                pickle.dump(fig, f)

        plt.close(fig)

    def plotVarVsDisc(self, vars, edges, ylim = -1.0, ylog = False, ylabel = "", tag = "", Njets = -1):

        print("Plotting var vs disc for {}".format(ylabel))

        x1 = []; x2 = []
       
        var = []
        varUnc = []

        for i in range(0, int(len(vars)/10)):
            x1.append(edges[0][i][0])
            x2.append(edges[0][i][1])
            var.append(vars[i])

        fig, ax = plt.subplots(figsize=(12,12))

        #Z1, xedges, yedges = np.histogram2d(var, x1, bins=100)
        #Z2, xedges, yedges = np.histogram2d(var, x2, bins=(xedges, yedges))

        #Z /= np.max(Z) if abs(np.max(Z)) >= abs(np.min(Z)) else np.min(Z)

        #normalize = mpl.colors.Normalize(vmin=-1, vmax=1)
        #hist = ax.pcolormesh(xedges, yedges, Z, cmap = CM.RdBu_r, norm=normalize)
        #c1 = ax.contour(var, x1, Z1)
        #cbar1 = plt.colorbar(c1, ax=ax)
        #cbar1.ax.set_ylabel("Normalized Events Disc. 1")

        data1 = {"var": var, "x": x1}
        data2 = {"var": var, "x": x2}

        sns.kdeplot(data=data1, x="var", y="x", ax=ax, label="Disc. 1")

        #c2 = ax.contour(var, x2, Z2)
        #cbar2 = plt.colorbar(c2, ax=ax)
        #cbar2.ax.set_ylabel("Normalized Events Disc. 2")

        sns.kdeplot(data=data2, x="var", y="x", ax=ax, label="Disc. 2")

        #ax.hist2d(var, x, label="Disc. 1 - Disc. 2", weights = w)
        #ax[1].hist2d(x2, var, label="Disc. 2")

        if ylim != -1.0:
             ax.set_ylim((0.0, ylim))
             #ax[1].set_ylim((0.0, ylim))

        ax.set_ylabel("Disc. Output"); ax.set_xlabel(ylabel)
        #ax[1].set_ylabel(ylabel); ax[1].set_xlabel("Disc. 2")

        if ylog:
            ax.set_yscale("log")
            #ax[1].set_yscale("log")

        plt.legend(loc='best')

        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2")

        fig.tight_layout()

        if Njets == -1: 
            fig.savefig(self.config["outputDir"]+"/%sbyDisc.png"%(tag), dpi=fig.dpi)
            with open(self.config["outputDir"]+"/%sbyDisc.pkl"%(tag), 'wb') as f:
                pickle.dump(fig, f)
        else:           
            fig.savefig(self.config["outputDir"]+"/%sbyDisc_Njets%s.png"%(tag,Njets), dpi=fig.dpi)
            with open(self.config["outputDir"]+"/%sbyDisc_Njets%s.pkl"%(tag,Njets), 'wb') as f:
                pickle.dump(fig, f)

        plt.close(fig)

    def plotBinEdgeMetricComps(self, finalSign, finalClosureErr, sign, closeErr, edges, d1edge, d2edge, Njets = -1):

        fig = plt.figure(figsize=(12,12))
        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2")
        ax = plt.gca()
        plt.scatter(np.reciprocal(sign[0]), closeErr[0], color='xkcd:silver', marker="o", label="1 - Pred./Obs. vs 1 / Significance")

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
            fig.savefig(self.config["outputDir"]+"/InvSign_vs_NonClosure.png", dpi=fig.dpi)        
            with open(self.config["outputDir"]+"/InvSign_vs_NonClosure.pkl", 'wb') as f:
                pickle.dump(fig, f)
        else:
            fig.savefig(self.config["outputDir"]+"/InvSign_vs_NonClosure_Njets%d.png"%(Njets), dpi=fig.dpi)        
            with open(self.config["outputDir"]+"/InvSign_vs_NonClosure_Njets%d.pkl"%(Njets), 'wb') as f:
                pickle.dump(fig, f)

        plt.close(fig)

        return np.average(closeErr), np.std(closeErr)

    def plotNjets(self, bkgd, sig, label):

        binCenters = [i for i in range(self.config["minNJetBin"], self.config["maxNJetBin"]+1)]
        xErr = [0.5 for i in range(0, len(bkgd))]

        sign = 0.0
        for i in range(0, len(bkgd)):
            if bkgd[i][0] > 0.0: sign += (sig[i][0] / (bkgd[i][0] + (0.3*bkgd[i][0])**2.0)**0.5)**2.0
        sign = sign**0.5

        fig = plt.figure(figsize=(12,12))
        ax = plt.gca()
        ax.set_yscale("log")
        ax.set_ylim([1,10e4])

        ax.errorbar(binCenters, bkgd[:,0], yerr=bkgd[:,1], label="Background", xerr=xErr, fmt='', color="black",   lw=0, elinewidth=2, marker="o", markerfacecolor="black")
        ax.errorbar(binCenters, sig[:,0],  yerr=sig[:,1],  label="Signal",     xerr=xErr, fmt='', color="red",     lw=0, elinewidth=2, marker="o", markerfacecolor="red")

        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2", ax=ax)

        plt.xlabel('$N_{jets}$')
        plt.ylabel('Events')
        plt.legend(loc='best')
        plt.text(0.05, 0.94, r"Significance = %.2f"%(sign), transform=ax.transAxes, fontfamily='sans-serif', fontsize=16, bbox=dict(facecolor='white', alpha=1.0))

        fig.savefig(self.config["outputDir"]+"/Njets_Region_%s.png"%(label))
        with open(self.config["outputDir"]+"/Njets_Region_%s.pkl"%(label), 'wb') as f:
            pickle.dump(fig, f)

        plt.close(fig)

        return sign

    def plotNjetsClosure(self, bkgd, bkgdPred, bkgdSign, tag = ""):

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

            if bkgd[i][1] != 0.0:
                binCenters.append(Njets[i])
                unc.append(bkgd[i][1])
                predUnc.append(bkgdPred[i][1])
                obs.append(bkgd[i][0])
                pred.append(bkgdPred[i][0])
                xErr.append(0.5)
                pull = (bkgdPred[i][0]-bkgd[i][0])/bkgd[i][1]
                closureError = 1.0 - bkgdPred[i][0]/bkgd[i][0]
                abcdPull.append(pull)
                abcdError.append(closureError)
                totalChi2 += pull**2.0
                wtotalChi2 += bkgdSign[i] * pull**2.0
                totalSig += bkgdSign[i]
                ndof += 1

        fig = plt.figure(figsize=(12,12))
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

        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2", ax=ax1)
        
        ax2.set_xlabel('$N_{jets}$')
        ax2.set_ylabel('1 - Pred./Obs.', fontsize="small")
        ax1.set_ylabel('Weighted Events')
        ax1.legend(loc='best')
        
        ax2.set_ylim([-0.6, 0.6])
        
        if tag != "":
            fig.savefig(self.config["outputDir"]+"/Njets_Region_A_PredVsActual_%s.png"%(tag))
            with open(self.config["outputDir"]+"/Njets_Region_A_PredVsActual_%s.pkl"%(tag), 'wb') as f:
                pickle.dump(fig, f)
        else:
            fig.savefig(self.config["outputDir"]+"/Njets_Region_A_PredVsActual.png")
            with open(self.config["outputDir"]+"/Njets_Region_A_PredVsActual.pkl", 'wb') as f:
                pickle.dump(fig, f)
        
        plt.close(fig)

        if totalSig == 0.0:
            wtotalChi2 = totalChi2

        return totalChi2, wtotalChi2, ndof

    def saveValData(self, var_list, var_names, metavar_list, metavar_names, bg_counts, sg_counts, edges, tag):
        
        out_dict = {}
        for k,meta in enumerate(metavar_list):
            print(k,meta)
            print(metavar_names[k])
            if "BE" in metavar_names[k]:
                out_dict[metavar_names[k]] = "({},{})".format(meta[0], meta[1])
            else:
                out_dict[metavar_names[k]] = meta
            
        out_dict["trainLoss"] = self.result_log.history["loss"][-1]
        out_dict["valLoss"] = self.result_log.history["val_loss"][-1]


        for i,e in enumerate(edges):
            out_dict["({},{})".format(e[0],e[1])] = {}

            for j,var in enumerate(var_list):
                out_dict["({},{})".format(e[0],e[1])][var_names[j]] = var[i][0]

            for key in bg_counts.keys():
                out_dict["({},{})".format(e[0],e[1])]["BkgEvts{}".format(key)] = bg_counts[key][i][0]
                out_dict["({},{})".format(e[0],e[1])]["SigEvts{}".format(key)] = sg_counts[key][i][0]

        valData_json = json.dumps(out_dict, indent=4)

        with open("{}/ValData_{}.json".format(self.config["outputDir"],tag), "w") as outfile:
            outfile.write(valData_json)
        outfile.close()
    #roc_curve(valData["label"][massMaskDataVal&sigMaskDataVal],   y_val_dist[massMaskDataVal&sigMaskDataVal],   sample_weight=valData["weight"][massMaskDataVal&sigMaskDataVal])
    def roc_curve_2D(self, labels, disc1, disc2, sample_weight=None):

        thresholds = [x for x in arange(0.0, 1.0, 0.001)]
        tprs = []
        fprs = []

        for t in thresholds:

            t_labels = np.array((disc1 > t) & (disc2 > t)).astype(int)
            tps = np.array((labels == t_labels) & (labels == 1)).astype(int)
            fps = np.array((labels != t_labels) & (t_labels == 1)).astype(int)

            tpr = float(tps.sum()) / len(tps)
            fpr = float(fps.sum()) / len(fps)

            tprs.append(tpr)
            fprs.append(fpr)

        return np.array(fprs), np.array(tprs), np.array(thresholds)

    def makePlots(self, doQuickVal=True, evalMass="400", evalModel="RPV_SYY_SHH"):
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

        variables = self.valLoader.getColumnHeaders()

        # If there is a xvalLoader that means we are evaluating the network
        # on events it has not seen and are not in the train+val+test sets
        if self.evalLoader != None:
            evalData = self.evalLoader.getFlatData(year=self.config["evalYear"])
            evalSig  = self.evalLoader.getFlatData(year=self.config["evalYear"],process=self.sample[evalModel]) 
            evalBkg  = self.evalLoader.getFlatData(year=self.config["evalYear"],process=self.config["evalBkg"])

            self.evalLoader = None
            gc.collect()

        # Making it to the else means the events we want to evaluate
        # are contained within the train+val+test sets
        else:
            trainDataTmp = self.loader.getFlatData(year=self.config["evalYear"])
            trainSigTmp  = self.loader.getFlatData(year=self.config["evalYear"],process=self.sample[evalModel]) 
            trainBkgTmp  = self.loader.getFlatData(year=self.config["evalYear"],process=self.config["evalBkg"])      
                

            self.loader = None
            gc.collect()

            valDataTmp = self.valLoader.getFlatData(year=self.config["evalYear"])
            valSigTmp = self.valLoader.getFlatData(year=self.config["evalYear"],process=self.sample[evalModel]) 
            valBkgTmp = self.valLoader.getFlatData(year=self.config["evalYear"],process=self.config["evalBkg"])

            self.valLoader = None
            gc.collect()

            testDataTmp = self.testLoader.getFlatData(year=self.config["evalYear"])
            testSigTmp  = self.testLoader.getFlatData(year=self.config["evalYear"],process=self.sample[evalModel])
            testBkgTmp  = self.testLoader.getFlatData(year=self.config["evalYear"],process=self.config["evalBkg"])
            
            self.testLoader = None
            gc.collect()

            for key in trainDataTmp.keys():
                evalData[key] = np.concatenate((trainDataTmp[key], valData[key],   testDataTmp[key]), axis=0)
                evalSig[key]  = np.concatenate((trainSigTmp[key],  valSigTmp[key], testSigTmp[key]),  axis=0)
                evalBkg[key]  = np.concatenate((trainBkgTmp[key],  valBkgTmp[key], testBkgTmp[key]),  axis=0)

       
        massMaskEval     = evalSig["mass"] == float(evalMass)
        massMaskVal      = valSig["mass"]  == float(valMass)

        massMaskDataEval = evalData["mass"]==float(evalMass)
        massMaskDataEval |= evalData["mass"]==float(173.0)
        massMaskDataVal  = valData["mass"]==float(valMass)
        massMaskDataVal |= valData["mass"]==float(173.0)

        # Make signal model mask for signal training dataset
        rpvMaskEval = evalSig["model"]==self.sample["RPV"]
        syyMaskEval = evalSig["model"]==self.sample["SYY"]
        shhMaskEval = evalSig["model"]==self.sample["SHH"]

        # Make signal model mask for mixed training dataset
        rpvMaskDataEval = evalData["model"]==self.sample["RPV"]
        syyMaskDataEval = evalData["model"]==self.sample["SYY"]
        shhMaskDataEval = evalData["model"]==self.sample["SHH"]
        bkgMaskDataEval = evalData["model"]==self.config["evalBkg"]

        # Make signal model mask for signal validation dataset
        rpvMaskVal = valSig["model"]==self.sample["RPV"]
        syyMaskVal = valSig["model"]==self.sample["SYY"]
        shhMaskVal = valSig["model"]==self.sample["SHH"]

        # Make signal model mask for mixed validation dataset
        rpvMaskDataVal = valData["model"]==self.sample["RPV"]
        syyMaskDataVal = valData["model"]==self.sample["SYY"]
        shhMaskDataVal = valData["model"]==self.sample["SHH"]
        bkgMaskDataVal = valData["model"]==0

        sigMaskEval = None; sigMaskDataEval = bkgMaskDataEval; sigMaskVal = None; sigMaskDataVal = bkgMaskDataVal
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

        # Separately loaded samples that can have nothing to do with the what was loaded for training
        output_train, output_eval_sg, output_eval_bg = self.getOutput(self.model, evalData["inputs"], evalSig["inputs"], evalBkg["inputs"])
        #if self.do_LRP:
        #    p_list, xaugs_list = get_lrp_score(evalData["inputs"], self.model, 0, len(evalData["inputs"]), len(evalData["inputs"])) 
        #    print("="*30)
        #    print(plist)
        #    print("="*30)

        del evalData["inputs"]; del evalSig["inputs"]; del evalBkg["inputs"]
        gc.collect()

        y_eval_disc1, y_eval_sg_disc1, y_eval_bg_disc1 = self.getResults(output_train, output_eval_sg, output_eval_bg, outputNum=0, columnNum=0)
        y_eval_disc2, y_eval_sg_disc2, y_eval_bg_disc2 = self.getResults(output_train, output_eval_sg, output_eval_bg, outputNum=0, columnNum=1)
        y_eval_mass,  y_eval_mass_sg,  y_eval_mass_bg  = self.getResults(output_train, output_eval_sg, output_eval_bg, outputNum=3, columnNum=0)

        # Part of the training samples that were not used for training
        output_val, output_val_sg, output_val_bg = self.getOutput(self.model, valData["inputs"], valSig["inputs"], valBkg["inputs"])

        #for i, var in enumerate(valData["vars"]):

        #    self.plotVarVsDisc(valBkg["inputs"][:, i], output_val_bg, ylabel = var, tag = var+ "_Bg_")
        #    self.plotVarVsDisc(valSig["inputs"][:, i], output_val_sg, ylabel = var, tag = var+ "_Sg_")

        del valData["inputs"]; del valSig["inputs"]; del valBkg["inputs"]
        gc.collect()

        y_val_disc1,  y_val_sg_disc1,  y_val_bg_disc1  = self.getResults(output_val,   output_val_sg,  output_val_bg,  outputNum=0, columnNum=0)
        y_val_disc2,  y_val_sg_disc2,  y_val_bg_disc2  = self.getResults(output_val,   output_val_sg,  output_val_bg,  outputNum=0, columnNum=1)
        y_val_mass,   y_val_mass_sg,   y_val_mass_bg   = self.getResults(output_val,   output_val_sg,  output_val_bg,  outputNum=3, columnNum=0)

        nBins = 20
        nBinsReg = 100
        masses = [350., 550., 850., 1150.]

        colors = ["red", "green", "blue", "magenta", "cyan"]; labels = ["Bkg Train", "Bkg Val"]

        #self.plotDisc([y_eval_mass_bg, y_val_mass_bg], colors, labels, [evalBkg["weight"], valBkg["weight"]], "mass",     'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg)
        #self.plotDisc([y_eval_mass_bg, y_val_mass_bg], colors, labels, [evalBkg["weight"], valBkg["weight"]], "mass_log", 'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg, doLog=True)

        tempColors = ["black"]; tempNames = ["ttbar"]; tempMass = [y_eval_mass_bg*1000]; tempEvents = [evalBkg["weight"]]; tempMassVal = [y_val_mass_bg*1000]; tempEventsVal = [valBkg["weight"]]
        i = 0
        for imass in masses:
            self.plotDisc([y_eval_mass_sg[(evalSig["mass"]==imass)&sigMaskEval]*1000, y_val_mass_sg[valSig["mass"]==imass]*1000], colors, labels, [evalSig["weight"][(evalSig["mass"]==imass)&sigMaskEval], valSig["weight"][valSig["mass"]==imass]], "mass_%d"%(imass), 'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg)

            tempColors.append(colors[i])
            tempNames.append("mass %d"%(imass))
            tempMass.append(y_eval_mass_sg[(evalSig["mass"]==imass)&sigMaskEval]*1000)
            tempEvents.append(evalSig["weight"][(evalSig["mass"]==imass)&sigMaskEval])

            tempMassVal.append(y_val_mass_sg[valSig["mass"]==imass]*1000)
            tempEventsVal.append(valSig["weight"][valSig["mass"]==imass])

            i += 1
        
        self.plotDisc([y_eval_bg_disc1, y_val_bg_disc1], colors, labels, [evalBkg["weight"], valBkg["weight"]], "Disc1", 'Norm Events', 'Disc. 1')
        self.plotDisc([y_eval_bg_disc2, y_val_bg_disc2], colors, labels, [evalBkg["weight"], valBkg["weight"]], "Disc2", 'Norm Events', 'Disc. 2')
        #if self.config['scaleLog']:
        #    tempMass = np.exp(tempMass)
        #    tempMassVal = np.exp(tempMassVal)
        self.plotDisc(tempMass, tempColors, tempNames, tempEvents, "mass_split",     'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg)
        self.plotDisc(tempMass, tempColors, tempNames, tempEvents, "mass_split_log", 'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg, doLog=True)

        self.plotDisc(tempMassVal, tempColors, tempNames, tempEventsVal, "mass_split_val",     'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg)
        self.plotDisc(tempMassVal, tempColors, tempNames, tempEventsVal, "mass_split_val_log", 'Norm Events', 'predicted mass', arange=(0, 2000), bins=nBinsReg, doLog=True)

        for NJets in NJetsRange:
            
            njets = float(NJets)
            if self.config["Mask"] and (int(NJets) in self.config["Mask_nJet"]): continue

            bkgNjetsMaskEval = evalBkg["njets"] == njets; sigNjetsMaskEval = evalSig["njets"] == njets
            bkgNjetsMaskVal  = valBkg["njets"]  == njets; sigNjetsMaskVal  = valSig["njets"]  == njets

            tempColors = ["black"]; tempNames = ["ttbar"]; tempMass = [y_eval_mass_bg[bkgNjetsMaskEval]*1000]; tempEvents = [evalBkg["weight"][bkgNjetsMaskEval]]; tempMassVal = [y_val_mass_bg[bkgNjetsMaskVal]*1000]; tempEventsVal = [valBkg["weight"][bkgNjetsMaskVal]]
            i = 0
            for imass in masses:
                if imass >= self.config["minStopMass"] and imass <= self.config["maxStopMass"]:
                    mask = "mask_m%d"%(imass)

                    tempColors.append(colors[i])
                    tempNames.append("mass %d"%(imass))
                    tempMass.append(y_eval_mass_sg[(evalSig["mass"]==imass)&sigMaskEval&sigNjetsMaskEval]*1000)
                    tempEvents.append(evalSig["weight"][(evalSig["mass"]==imass)&sigMaskEval&sigNjetsMaskEval])

                    tempMassVal.append(y_val_mass_sg[(valSig["mass"]==imass)&sigMaskVal&sigNjetsMaskVal]*1000)
                    tempEventsVal.append(valSig["weight"][(valSig["mass"]==imass)&sigMaskVal&sigNjetsMaskVal])

                    i += 1

            self.plotDisc(tempMass, tempColors, tempNames, tempEvents, "mass_split_Njets%s"%(NJets),     'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg)
            self.plotDisc(tempMass, tempColors, tempNames, tempEvents, "mass_split_Njets%s_log"%(NJets), 'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg, doLog=True)

            self.plotDisc(tempMassVal, tempColors, tempNames, tempEventsVal, "mass_split_val_Njets%s"%(NJets),     'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg)

            self.plotDisc(tempMassVal, tempColors, tempNames, tempEventsVal, "mass_split_val_Njets%s_log"%(NJets), 'Norm Events', 'predicted mass', arange=(-10, 2000), bins=nBinsReg, doLog=True)
        
        # Plot Acc vs Epoch
        if self.result_log != None:
            self.plotAccVsEpoch('loss', 'val_loss', 'model loss', 'loss_train_val')
            #for rank in ["disco", "disc", "mass_reg"]: self.plotAccVsEpoch('%s_loss'%(rank), 'val_%s_loss'%(rank), '%s loss'%(rank), '%s_loss_train_val'%(rank))        
            for rank in ["disco", "disc", "closure", "mass_reg"]: self.plotAccVsEpoch('%s_loss'%(rank), 'val_%s_loss'%(rank), '%s loss'%(rank), '%s_loss_train_val'%(rank))        
            #for rank in ["disco", "disc1", "disc2", "closure", "mass_reg"]: self.plotAccVsEpoch('%s_loss'%(rank), 'val_%s_loss'%(rank), '%s loss'%(rank), '%s_loss_train_val'%(rank))        

            #self.plotAccVsEpochAll(['disc', 'mass_reg' , 'disco'], ['Combined Disc Loss', 'Mass Regression Loss', 'DisCo Loss'], '',     'train output loss',      'output_loss_train')
            #self.plotAccVsEpochAll(['disc', 'mass_reg' , 'disco'], ['Combined Disc Loss', 'Mass Regression Loss', 'DisCo Loss'], 'val_', 'validation output loss', 'output_loss_val')
            self.plotAccVsEpochAll(['disc', 'disco', 'closure', 'mass_reg'], ['Combined Disc Loss', 'DisCo Loss', 'Closure Loss', 'Mass Reg. Loss'], '',     'train output loss',      'output_loss_train')
            self.plotAccVsEpochAll(['disc', 'disco', 'closure', 'mass_reg'], ['Combined Disc Loss', 'DisCo Loss', 'Closure Loss', 'Mass Reg. Loss'], 'val_', 'validation output loss', 'output_loss_val')

            #self.plotAccVsEpochAll(['disc1', 'disc2', 'disco', 'closure', 'mass_reg'], ['Disc 1 Loss', 'Disc 2 Loss', 'DisCo Loss', 'Closure Loss', 'Mass Reg. Loss'], '',     'train output loss',      'output_loss_train')
            #self.plotAccVsEpochAll(['disc1', 'disc2', 'disco', 'closure', 'mass_reg'], ['Disc 1 Loss', 'Disc 2 Loss', 'DisCo Loss', 'Closure Loss', 'Mass Reg. Loss'], 'val_', 'validation output loss', 'output_loss_val')

        # Plot disc per njet
        self.plotDiscPerNjet("_Disc1", {"Bkg": [evalBkg, y_eval_bg_disc1, evalBkg["weight"]], "Sig": [evalSig, y_eval_sg_disc1, evalSig["weight"]]}, sigMaskEval, nBins=nBins)
        self.plotDiscPerNjet("_Disc2", {"Bkg": [evalBkg, y_eval_bg_disc2, evalBkg["weight"]], "Sig": [evalSig, y_eval_sg_disc2, evalSig["weight"]]}, sigMaskEval, nBins=nBins)
        
        if not doQuickVal:
            #self.plotD1VsD2SigVsBkgd(y_eval_bg_disc1, y_eval_bg_disc2, y_eval_sg_disc1[massMaskEval&sigMaskEval], y_eval_sg_disc2[massMaskEval&sigMaskEval], evalMass)
            # Make arrays for possible values to cut on for both discriminant
            # starting at a minimum of 0.5 for each
            edgeWidth = 0.05; minEdge = 0.0; maxEdge = 1.0 
            c1s = np.arange(minEdge, maxEdge, edgeWidth); c2s = np.arange(minEdge, maxEdge, edgeWidth)

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
            self.plotDisc1vsDisc2(y_eval_bg_disc1, y_eval_bg_disc2, evalBkg["weight"], -1.0, -1.0, -1.0, "BG")
            self.plotDisc1vsDisc2(y_eval_sg_disc1[massMaskEval&sigMaskEval], y_eval_sg_disc2[massMaskEval&sigMaskEval], evalSig["weight"][massMaskEval&sigMaskEval], -1.0, -1.0, -1.0, "SG", mass=evalMass)

            self.plotDisc1vsDisc2(y_val_bg_disc1, y_val_bg_disc2, valBkg["weight"], -1.0, -1.0, -1.0, "valBG")
            self.plotDisc1vsDisc2(y_val_sg_disc1[massMaskVal&sigMaskVal], y_val_sg_disc2[massMaskVal&sigMaskVal], valSig["weight"][massMaskVal&sigMaskVal], -1.0, -1.0, -1.0, "valSG", mass=valMass)

            for mass in [350, 850, 1150]:
                massMaskEvalTemp     = evalSig["mass"] == float(mass)
                massMaskValTemp      = valSig["mass"]  == float(mass)
                self.plotDisc1vsDisc2(y_eval_sg_disc1[massMaskEvalTemp&sigMaskEval], y_eval_sg_disc2[massMaskEvalTemp&sigMaskEval], evalSig["weight"][massMaskEvalTemp&sigMaskEval], -1.0, -1.0, -1.0, "SG", mass=mass)
                self.plotDisc1vsDisc2(y_val_sg_disc1[massMaskValTemp&sigMaskVal], y_val_sg_disc2[massMaskValTemp&sigMaskVal], valSig["weight"][massMaskValTemp&sigMaskVal], -1.0, -1.0, -1.0, "valSG", mass=mass)
                
            
            bkgdNjets = {"A" : [], "B" : [], "C" : [], "D" : [], "a" : [], "b" : [], "c" : [], "d" : []}
            sigNjets  = {"A" : [], "B" : [], "C" : [], "D" : [], "a" : [], "b" : [], "c" : [], "d" : []}
            totNjets  = {"A" : [], "B" : [], "C" : [], "D" : [], "a" : [], "b" : [], "c" : [], "d" : []}

            bkgdNjetsAPred = []
            totNjetsAPred = []
            bkgdNjetsSign = []
            
            # Saving information for money plot using all possible bin edge choices (no masking)
            bc, sc = self.cutAndCount(c1s, c2s, y_eval_bg_disc1, y_eval_bg_disc2, evalBkg["weight"], y_eval_sg_disc1, y_eval_sg_disc2, evalSig["weight"])
            c1, c2, significance, closureErr, edges, signs, signsWNC, predsigns, closeErrs, sFracs, wBkg, uwBkg, wSig, uwSig, normSigFracs= self.findABCDedges(bc, sc)


            #avg_sign = np.mean(signs[:, 0])
            #max_sign = np.max(signs[:, 0])
            #max_sign_be = edges[np.argmax(signs[:, 0])]
            #avg_closure = np.mean(closeErrs[:, 0])
            #var_closure = np.var(closeErrs[:, 0])
            #count_close = np.count_nonzero(closeErrs[:,0] < 0.3)

            #var_names = ["Sign", "SignWithNonClosure", "PredictedSign", "NonClosure", "normSigFracs"]
            #metavar_names = ["AvgNonClosure", "VarNonClosure", "AvgSign", "MaxSign", "MaxSignBE", "CountReasonable"]
            #self.saveValData([signs, signsWNC, predsigns, closeErrs, normSigFracs], var_names, [avg_closure, var_closure, avg_sign, max_sign, max_sign_be, count_close/closeErrs.shape[0]], metavar_names, wBkg, wSig, edges, self.config["atag"])

            bc, sc = self.cutAndCount(c1s, c2s, y_eval_bg_disc1, y_eval_bg_disc2, evalBkg["weight"], y_eval_sg_disc1, y_eval_sg_disc2, evalSig["weight"])
            c1, c2, significance, closureErr, edges, signs, signsWNC, predsigns, closeErrs, sFracs, wBkg, uwBkg, wSig, uwSig, normSigFracs = self.findABCDedges(bc, sc)

            self.plotVarVsBinEdges(closeErrs[:,0], edges, float(c1), float(c2), minEdge, maxEdge, edgeWidth, 20.0, 1.0, "NonClosure")
            self.plotVarVsBinEdges(signs[:,0], edges, float(c1), float(c2), minEdge, maxEdge, edgeWidth, 900.0, None, "Sign")

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
                c1, c2, significance, closureErr, edges, signs, signsWNC, predsigns, closeErrs, sFracs, wBkg, uwBkg, wSig, uwSig, normSigFracs = self.findABCDedges(bc, sc)
                if len(signs) > 0:
                    self.plotVarVsBinEdges(signs[:,0], edges, float(c1), float(c2), minEdge, maxEdge, edgeWidth, 20.0, 2.0, "Sign",    int(NJets))
                    self.plotVarVsBinEdges(signs[:,1], edges, float(c1), float(c2), minEdge, maxEdge, edgeWidth, 20.0, 0.5, "SignUnc", int(NJets))

                    self.plotVarVsBinEdges(closeErrs[:,0], edges, float(c1), float(c2), minEdge, maxEdge, edgeWidth, 20.0, 1.0, "NonClosure",    int(NJets))
                    self.plotVarVsBinEdges(closeErrs[:,1], edges, float(c1), float(c2), minEdge, maxEdge, edgeWidth, 20.0, 1.0, "NonClosureUnc", int(NJets))

                    tempAveClose, tempStdClose = self.plotBinEdgeMetricComps(significance, closureErr, signs, closeErrs, c1s, c1, c2, int(NJets))

                bkgdNjetsSign.append(significance)

                if c1 == -1.0 or c2 == -1.0:
                    bkgdNjets["A"].append([0.0, 0.0]); sigNjets["A"].append([0.0, 0.0]); totNjets["A"].append([0.0, 0.0])
                    bkgdNjets["B"].append([0.0, 0.0]); sigNjets["B"].append([0.0, 0.0]); totNjets["B"].append([0.0, 0.0])
                    bkgdNjets["C"].append([0.0, 0.0]); sigNjets["C"].append([0.0, 0.0]); totNjets["C"].append([0.0, 0.0])
                    bkgdNjets["D"].append([0.0, 0.0]); sigNjets["D"].append([0.0, 0.0]); totNjets["D"].append([0.0, 0.0])

                    bkgdNjets["a"].append([0.0, 0.0]); sigNjets["a"].append([0.0, 0.0]); totNjets["a"].append([0.0, 0.0])
                    bkgdNjets["b"].append([0.0, 0.0]); sigNjets["b"].append([0.0, 0.0]); totNjets["b"].append([0.0, 0.0])
                    bkgdNjets["c"].append([0.0, 0.0]); sigNjets["c"].append([0.0, 0.0]); totNjets["c"].append([0.0, 0.0])
                    bkgdNjets["d"].append([0.0, 0.0]); sigNjets["d"].append([0.0, 0.0]); totNjets["d"].append([0.0, 0.0])

                    bkgdNjetsAPred.append([0.0,0.0])
                    totNjetsAPred.append([0.0,0.0])

                else:
                    Apred, ApredUnc = self.predictABCD(bc["B"][c1][c2], bc["C"][c1][c2], bc["D"][c1][c2], bc["B2"][c1][c2]**0.5, bc["C2"][c1][c2]**0.5, bc["D2"][c1][c2]**0.5)
                    ApredTot, ApredTotUnc = self.predictABCD(bc["B"][c1][c2]+sc["B"][c1][c2], bc["C"][c1][c2]+sc["C"][c1][c2], bc["D"][c1][c2]+sc["D"][c1][c2], (bc["B2"][c1][c2]+sc["B2"][c1][c2])**0.5, (bc["C2"][c1][c2]+sc["C2"][c1][c2])**0.5, (bc["D2"][c1][c2]+sc["D2"][c1][c2])**0.5)

                    bkgdNjets["a"].append([bc["a"][c1][c2], bc["a"][c1][c2]**0.5]); sigNjets["a"].append([sc["a"][c1][c2], sc["a"][c1][c2]**0.5])
                    bkgdNjets["b"].append([bc["b"][c1][c2], bc["b"][c1][c2]**0.5]); sigNjets["b"].append([sc["b"][c1][c2], sc["b"][c1][c2]**0.5])
                    bkgdNjets["c"].append([bc["c"][c1][c2], bc["c"][c1][c2]**0.5]); sigNjets["c"].append([sc["c"][c1][c2], sc["c"][c1][c2]**0.5])
                    bkgdNjets["d"].append([bc["d"][c1][c2], bc["d"][c1][c2]**0.5]); sigNjets["d"].append([sc["d"][c1][c2], sc["d"][c1][c2]**0.5])

                    bkgdNjets["A"].append([bc["A"][c1][c2], bc["A2"][c1][c2]**0.5]); sigNjets["A"].append([sc["A"][c1][c2], sc["A2"][c1][c2]**0.5])
                    bkgdNjets["B"].append([bc["B"][c1][c2], bc["B2"][c1][c2]**0.5]); sigNjets["B"].append([sc["B"][c1][c2], sc["B2"][c1][c2]**0.5])
                    bkgdNjets["C"].append([bc["C"][c1][c2], bc["C2"][c1][c2]**0.5]); sigNjets["C"].append([sc["C"][c1][c2], sc["C2"][c1][c2]**0.5])
                    bkgdNjets["D"].append([bc["D"][c1][c2], bc["D2"][c1][c2]**0.5]); sigNjets["D"].append([sc["D"][c1][c2], sc["D2"][c1][c2]**0.5])

                    totNjets["a"].append([bc["a"][c1][c2]+sc["a"][c1][c2], sc["a"][c1][c2]**0.5])
                    totNjets["b"].append([bc["b"][c1][c2]+sc["b"][c1][c2], sc["b"][c1][c2]**0.5])
                    totNjets["c"].append([bc["c"][c1][c2]+sc["c"][c1][c2], sc["c"][c1][c2]**0.5])
                    totNjets["d"].append([bc["d"][c1][c2]+sc["d"][c1][c2], sc["d"][c1][c2]**0.5])

                    totNjets["A"].append([bc["A"][c1][c2]+sc["A"][c1][c2], (bc["A2"][c1][c2]+sc["A2"][c1][c2])**0.5])
                    totNjets["B"].append([bc["B"][c1][c2]+sc["B"][c1][c2], (bc["B2"][c1][c2]+sc["B2"][c1][c2])**0.5])
                    totNjets["C"].append([bc["C"][c1][c2]+sc["C"][c1][c2], (bc["C2"][c1][c2]+sc["C2"][c1][c2])**0.5])
                    totNjets["D"].append([bc["D"][c1][c2]+sc["D"][c1][c2], (bc["D2"][c1][c2]+sc["D2"][c1][c2])**0.5])

                    bkgdNjetsAPred.append([Apred,ApredUnc])
                    totNjetsAPred.append([ApredTot,ApredTotUnc])

                self.config["c1_nJet_%s"%(("%s"%(NJets)).zfill(2))] = c1
                self.config["c2_nJet_%s"%(("%s"%(NJets)).zfill(2))] = c2

                # Avoid completely masked Njets bins that makes below plotting
                # highly unsafe
                if not any(bkgNjetsMaskEval) or not any(sigNjetsMaskEval): continue

                #self.plotD1VsD2SigVsBkgd(y_eval_bg_disc1[bkgFullMaskEval], y_eval_bg_disc2[bkgFullMaskEval], y_eval_sg_disc1[sigFullMaskEval], y_eval_sg_disc2[sigFullMaskEval], evalMass, NJets)

                # Plot each discriminant for sig and background while making cut on other disc
                #self.plotDiscWithCut(float(c2), y_eval_bg_disc1[bkgFullMaskEval], y_eval_bg_disc2[bkgFullMaskEval], evalBkg["weight"][bkgFullMaskEval], y_eval_sg_disc1[sigFullMaskEval], y_eval_sg_disc2[sigFullMaskEval], evalSig["weight"][sigFullMaskEval], "1", "2", mass=evalMass, Njets=NJets, bins=nBins)
                #self.plotDiscWithCut(float(c1), y_eval_bg_disc2[bkgFullMaskEval], y_eval_bg_disc1[bkgFullMaskEval], evalBkg["weight"][bkgFullMaskEval], y_eval_sg_disc2[sigFullMaskEval], y_eval_sg_disc1[sigFullMaskEval], evalSig["weight"][sigFullMaskEval], "2", "1", mass=evalMass, Njets=NJets, bins=nBins)
            
                #self.plotDiscWithCutCompare(float(c2), y_eval_bg_disc1[bkgFullMaskEval], y_eval_bg_disc2[bkgFullMaskEval], evalBkg["weight"][bkgFullMaskEval], "1", "2", "BG", mass="", Njets=-1, bins=10)
                #self.plotDiscWithCutCompare(float(c2), y_eval_sg_disc1[sigFullMaskEval], y_eval_sg_disc2[sigFullMaskEval], evalSig["weight"][sigFullMaskEval], "1", "2", "SG", mass=evalMass, Njets=NJets, bins=10)
            
                #self.plotDiscWithCutCompare(float(c1), y_eval_bg_disc2[bkgFullMaskEval], y_eval_bg_disc1[bkgFullMaskEval], evalBkg["weight"][bkgFullMaskEval], "2", "1", "BG", mass="", Njets=-1, bins=10)
                #self.plotDiscWithCutCompare(float(c1), y_eval_sg_disc2[sigFullMaskEval], y_eval_sg_disc1[sigFullMaskEval], evalSig["weight"][sigFullMaskEval], "2", "1", "SG", mass=evalMass, Njets=NJets, bins=10)
            
                # Plot 2D of the discriminants
                bkgdCorr = self.plotDisc1vsDisc2(y_eval_bg_disc1[bkgFullMaskEval], y_eval_bg_disc2[bkgFullMaskEval], evalBkg["weight"][bkgFullMaskEval], float(c1), float(c2), significance, "BG", mass="",   Njets=NJets)
                self.plotDisc1vsDisc2(y_eval_sg_disc1[sigFullMaskEval], y_eval_sg_disc2[sigFullMaskEval], evalSig["weight"][sigFullMaskEval], float(c1), float(c2), significance, "SG", mass=evalMass, Njets=NJets)
                self.metric["bkgdCorr_nJet_%s"%(NJets)] = abs(bkgdCorr) 

                self.plotDisc1vsDisc2(y_val_bg_disc1[bkgFullMaskVal], y_val_bg_disc2[bkgFullMaskVal], valBkg["weight"][bkgFullMaskVal], float(c1), float(c2), significance, "valBG", mass="",   Njets=NJets)
                self.plotDisc1vsDisc2(y_val_sg_disc1[sigFullMaskVal], y_val_sg_disc2[sigFullMaskVal], valSig["weight"][sigFullMaskVal], float(c1), float(c2), significance, "valSG", mass=valMass, Njets=NJets)

                '''for mass in [350, 850, 1150]:

                    massMaskEvalTemp     = evalSig["mass"] == float(mass)
                    massMaskValTemp      = valSig["mass"]  == float(mass)

                    bkgFullMaskEval   = bkgNjetsMaskEval; sigFullMaskEval  = sigMaskEval & massMaskEvalTemp & sigNjetsMaskEval
                    bkgFullMaskVal   = bkgNjetsMaskVal; sigFullMaskVal  = sigMaskVal & massMaskValTemp & sigNjetsMaskVal

                    self.plotDisc1vsDisc2(y_eval_sg_disc1[massMaskEvalTemp&sigMaskEval], y_eval_sg_disc2[massMaskEvalTemp&sigMaskEval], evalSig["weight"][massMaskEvalTemp&sigMaskEval], float(c1), float(c2), -1.0, "SG", mass=mass, Njets=NJets)
                    self.plotDisc1vsDisc2(y_val_sg_disc1[massMaskValTemp&sigMaskVal], y_val_sg_disc2[massMaskValTemp&sigMaskVal], valSig["weight"][massMaskValTemp&sigMaskVal], float(c1), float(c2), -1.0, "valSG", mass=mass, Njets=NJets)
                '''
            signA = self.plotNjets(np.array(bkgdNjets["A"]), np.array(sigNjets["A"]), "A")
            signB = self.plotNjets(np.array(bkgdNjets["B"]), np.array(sigNjets["B"]), "B")
            signC = self.plotNjets(np.array(bkgdNjets["C"]), np.array(sigNjets["C"]), "C")
            signD = self.plotNjets(np.array(bkgdNjets["D"]), np.array(sigNjets["D"]), "D")

            # Nominal closure plot for just background
            totalChi2, wtotalChi2, ndof = self.plotNjetsClosure(bkgdNjets["A"], bkgdNjetsAPred, bkgdNjetsSign, "")

            # Compare prediction using sig+bkg to just observed background
            self.plotNjetsClosure(bkgdNjets["A"], totNjetsAPred, bkgdNjetsSign, "Contamination")

            # Compare observed sig+bkg with predicted bkg
            self.plotNjetsClosure(totNjets["A"], bkgdNjetsAPred, bkgdNjetsSign, "Sensitivity")

            # Compare observed sig+bkg with predicted sig+bkg 
            self.plotNjetsClosure(totNjets["A"], totNjetsAPred, bkgdNjetsSign, "PseudoData")

            self.config["Achi2"] = totalChi2
            if ndof != 0:
                self.config["Achi2ndof"] = float(totalChi2/ndof)
                self.config["Awchi2ndof"] = float(wtotalChi2/ndof)
            else:
                self.config["Achi2ndof"] = 9999.0

            # Compute significance with signal contamination factored in
            # Replace inverse significance in old metric with this

            tempTotalSig = 0.0

            for i in range(len(bkgdNjets["A"])):

                # Per njet bin, compute the closure prediction for number of background events in the A region
                predBgRegA = (bkgdNjets["B"][i][0] + sigNjets["B"][i][0] * bkgdNjets["C"][i][0] + sigNjets["C"][i][0]) / (bkgdNjets["D"][i][0] + sigNjets["D"][i][0])

                # Calculate significance per njet bin and sum into total significance

                tempTotalSig += (sigNjets["A"][i][0] / math.sqrt(predBgRegA)) ** 2

            # Take the square root as this is the quadrature sum of the significance per njet bin
            self.config["TotalSignificance"] = math.sqrt(tempTotalSig)

            self.config["Asignificance"] = float(signA)
            self.config["Bsignificance"] = float(signB)
            self.config["Csignificance"] = float(signC)
            self.config["Dsignificance"] = float(signD)
            #self.config["TotalSignificance"] = (signA**2.0 + signB**2.0 + signC**2.0 + signD**2.0)**0.5

            if    self.config["TotalSignificance"] > 0.0: self.metric["InvTotalSignificance"] = 1.0/self.config["TotalSignificance"]
            else: self.metric["InvTotalSignificance"] = 999.0

        y_val_avg = (y_val_disc1 + y_val_disc2) / 2
        y_eval_avg = (y_eval_disc1 + y_eval_disc2) / 2

        y_val_dist = np.sqrt(y_val_disc1**2 + y_val_disc2**2) / np.sqrt(2)
        y_eval_dist = np.sqrt(y_eval_disc1**2 + y_eval_disc2**2) / np.sqrt(2)

        y_val_stack = np.stack([y_val_disc1, y_val_disc2], axis=-1)
        y_val_idx = np.unravel_index(np.argmax(y_val_stack, axis=1), y_val_stack.shape)
        y_val_double = y_val_stack[y_val_idx]

        y_eval_stack = np.stack([y_eval_disc1, y_eval_disc2], axis=-1)
        y_eval_idx = np.unravel_index(np.argmax(y_eval_stack, axis=1), y_eval_stack.shape)
        y_eval_double = y_eval_stack[y_eval_idx]

        # Plot validation roc curve
        fpr_val_disc1, tpr_val_disc1, thresholds_val_disc1    = roc_curve(valData["label"][massMaskDataVal&sigMaskDataVal],   y_val_disc1[massMaskDataVal&sigMaskDataVal],   sample_weight=valData["weight"][massMaskDataVal&sigMaskDataVal])
        fpr_val_disc2, tpr_val_disc2, thresholds_val_disc2    = roc_curve(valData["label"][massMaskDataVal&sigMaskDataVal],   y_val_disc2[massMaskDataVal&sigMaskDataVal],   sample_weight=valData["weight"][massMaskDataVal&sigMaskDataVal])
        fpr_eval_disc1, tpr_eval_disc1, thresholds_eval_disc1 = roc_curve(evalData["label"][massMaskDataEval&sigMaskDataEval], y_eval_disc1[massMaskDataEval&sigMaskDataEval], sample_weight=evalData["weight"][massMaskDataEval&sigMaskDataEval])
        fpr_eval_disc2, tpr_eval_disc2, thresholds_eval_disc2 = roc_curve(evalData["label"][massMaskDataEval&sigMaskDataEval], y_eval_disc2[massMaskDataEval&sigMaskDataEval], sample_weight=evalData["weight"][massMaskDataEval&sigMaskDataEval])

        fpr_val_avg, tpr_val_avg, thresholds_val_avg    = roc_curve(valData["label"][massMaskDataVal&sigMaskDataVal],   y_val_avg[massMaskDataVal&sigMaskDataVal],   sample_weight=valData["weight"][massMaskDataVal&sigMaskDataVal])
        fpr_eval_avg, tpr_eval_avg, thresholds_eval_avg = roc_curve(evalData["label"][massMaskDataEval&sigMaskDataEval], y_eval_avg[massMaskDataEval&sigMaskDataEval], sample_weight=evalData["weight"][massMaskDataEval&sigMaskDataEval])
        fpr_val_dist, tpr_val_dist, thresholds_val_dist    = roc_curve(valData["label"][massMaskDataVal&sigMaskDataVal],   y_val_dist[massMaskDataVal&sigMaskDataVal],   sample_weight=valData["weight"][massMaskDataVal&sigMaskDataVal])
        fpr_eval_dist, tpr_eval_dist, thresholds_eval_dist = roc_curve(evalData["label"][massMaskDataEval&sigMaskDataEval], y_eval_dist[massMaskDataEval&sigMaskDataEval], sample_weight=evalData["weight"][massMaskDataEval&sigMaskDataEval])
        fpr_val_double, tpr_val_double, thresholds_val_double    = roc_curve(valData["label"][massMaskDataVal&sigMaskDataVal],   y_val_double[massMaskDataVal&sigMaskDataVal],   sample_weight=valData["weight"][massMaskDataVal&sigMaskDataVal])
        fpr_eval_double, tpr_eval_double, thresholds_eval_double = roc_curve(evalData["label"][massMaskDataEval&sigMaskDataEval], y_eval_double[massMaskDataEval&sigMaskDataEval], sample_weight=evalData["weight"][massMaskDataEval&sigMaskDataEval])

        auc_val_disc1  = roc_auc_score(valData["label"][massMaskDataVal&sigMaskDataVal],   y_val_disc1[massMaskDataVal&sigMaskDataVal])
        auc_val_disc2  = roc_auc_score(valData["label"][massMaskDataVal&sigMaskDataVal],   y_val_disc2[massMaskDataVal&sigMaskDataVal])
        auc_eval_disc1 = roc_auc_score(evalData["label"][massMaskDataEval&sigMaskDataEval], y_eval_disc1[massMaskDataEval&sigMaskDataEval])
        auc_eval_disc2 = roc_auc_score(evalData["label"][massMaskDataEval&sigMaskDataEval], y_eval_disc2[massMaskDataEval&sigMaskDataEval])

        auc_val_avg  = roc_auc_score(valData["label"][massMaskDataVal&sigMaskDataVal],   y_val_avg[massMaskDataVal&sigMaskDataVal])
        auc_eval_avg = roc_auc_score(evalData["label"][massMaskDataEval&sigMaskDataEval], y_eval_avg[massMaskDataEval&sigMaskDataEval])
        auc_val_dist  = roc_auc_score(valData["label"][massMaskDataVal&sigMaskDataVal],   y_val_dist[massMaskDataVal&sigMaskDataVal])
        auc_eval_dist = roc_auc_score(evalData["label"][massMaskDataEval&sigMaskDataEval], y_eval_dist[massMaskDataEval&sigMaskDataEval])
        auc_val_double  = roc_auc_score(valData["label"][massMaskDataVal&sigMaskDataVal],   y_val_double[massMaskDataVal&sigMaskDataVal])
        auc_eval_double = roc_auc_score(evalData["label"][massMaskDataEval&sigMaskDataEval], y_eval_double[massMaskDataEval&sigMaskDataEval])

        #fpr_val_double, tpr_val_double, thresholds_val_double    = self.roc_curve_2D(valData["label"][massMaskDataVal&sigMaskDataVal],   y_val_disc1[massMaskDataVal&sigMaskDataVal],  y_val_disc2[massMaskDataVal&sigMaskDataVal],  sample_weight=valData["weight"][massMaskDataVal&sigMaskDataVal])
        #fpr_eval_double, tpr_eval_double, thresholds_eval_double    = self.roc_curve_2D(evalData["label"][massMaskDataVal&sigMaskDataVal],   y_eval_disc1[massMaskDataVal&sigMaskDataVal],  y_eval_disc2[massMaskDataVal&sigMaskDataVal],  sample_weight=evalData["weight"][massMaskDataVal&sigMaskDataVal])

        #auc_val_double = np.trapz(tpr_val_double, fpr_val_double)
        #auc_eval_double = np.trapz(tpr_eval_double, fpr_eval_double)

        # Define metrics for the training
        self.metric["OverTrain_Disc1"]   = abs(auc_val_disc1 - auc_eval_disc1)
        self.metric["OverTrain_Disc2"]   = abs(auc_val_disc2 - auc_eval_disc2)
        self.metric["Performance_Disc1"] = abs(1 - auc_eval_disc1)
        self.metric["Performance_Disc2"] = abs(1 - auc_eval_disc2)
       
        # Plot some ROC curves
        self.plotROC(None, None, "_Disc1", None, None, None, None, fpr_eval_disc1, fpr_val_disc1, tpr_eval_disc1, tpr_val_disc1, auc_eval_disc1, auc_val_disc1)
        self.plotROC(None, None, "_Disc2", None, None, None, None, fpr_eval_disc2, fpr_val_disc2, tpr_eval_disc2, tpr_val_disc2, auc_eval_disc2, auc_val_disc2)
        self.plotROC(None, None, "_Avg", None, None, None, None, fpr_eval_avg, fpr_val_avg, tpr_eval_avg, tpr_val_avg, auc_eval_avg, auc_val_avg)
        self.plotROC(None, None, "_Dist", None, None, None, None, fpr_eval_dist, fpr_val_dist, tpr_eval_dist, tpr_val_dist, auc_eval_dist, auc_val_dist)
        self.plotROC(None, None, "_Double", None, None, None, None, fpr_eval_double, fpr_val_double, tpr_eval_double, tpr_val_double, auc_eval_double, auc_val_double)

        self.plotROC(massMaskDataEval&sigMaskDataEval, massMaskDataVal&sigMaskDataVal, "_"+self.config["bkgd"][0]+"_nJet_disc1", y_eval_disc1, y_val_disc1, evalData, valData)
        self.plotROC(massMaskDataEval&sigMaskDataEval, massMaskDataVal&sigMaskDataVal, "_"+self.config["bkgd"][0]+"_nJet_disc2", y_eval_disc2, y_val_disc2, evalData, valData)
        self.plotROC(sigMaskDataEval, sigMaskDataVal, "_mass_split_disc1", y_eval_disc1, y_val_disc1, evalData, valData, doMass=True, minMass=self.config['minStopMass'], maxMass=self.config['maxStopMass'])
        self.plotROC(sigMaskDataEval, sigMaskDataVal, "_mass_split_disc2", y_eval_disc2, y_val_disc2, evalData, valData, doMass=True, minMass=self.config['minStopMass'], maxMass=self.config['maxStopMass'])
       
        # Plot validation precision recall
        precision_val_disc1,  recall_val_disc1,  _ = precision_recall_curve(valData["label"][massMaskDataVal&sigMaskDataVal],   y_val_disc1[massMaskDataVal&sigMaskDataVal],   sample_weight=valData["weight"][massMaskDataVal&sigMaskDataVal])
        precision_eval_disc1, recall_eval_disc1, _ = precision_recall_curve(evalData["label"][massMaskDataEval&sigMaskDataEval], y_eval_disc1[massMaskDataEval&sigMaskDataEval], sample_weight=evalData["weight"][massMaskDataEval&sigMaskDataEval])
        ap_val_disc1  = average_precision_score(valData["label"],  y_val_disc1,  sample_weight=valData["weight"])
        ap_eval_disc1 = average_precision_score(evalData["label"], y_eval_disc1, sample_weight=evalData["weight"])
        
        self.plotPandR(precision_val_disc1, recall_val_disc1, precision_eval_disc1, recall_eval_disc1, ap_val_disc1, ap_eval_disc1, "disc1")
        
        precision_val_disc2,  recall_val_disc2,  _ = precision_recall_curve(valData["label"][massMaskDataVal&sigMaskDataVal],   y_val_disc2[massMaskDataVal&sigMaskDataVal],   sample_weight=valData["weight"][massMaskDataVal&sigMaskDataVal])
        precision_eval_disc2, recall_eval_disc2, _ = precision_recall_curve(evalData["label"][massMaskDataEval&sigMaskDataEval], y_eval_disc2[massMaskDataEval&sigMaskDataEval], sample_weight=evalData["weight"][massMaskDataEval&sigMaskDataEval])
        ap_val_disc2  = average_precision_score(valData["label"],  y_val_disc2,  sample_weight=valData["weight"])
        ap_eval_disc2 = average_precision_score(evalData["label"], y_eval_disc2, sample_weight=evalData["weight"])
        
        self.plotPandR(precision_val_disc2, recall_val_disc2, precision_eval_disc2, recall_eval_disc2, ap_val_disc2, ap_eval_disc2, "disc2")

        self.plotPandR2D(valData["label"][sigMaskDataVal], y_val_disc2[sigMaskDataVal], y_val_disc1[sigMaskDataVal], sample_weight=valData["weight"][sigMaskDataVal], name="disc1")
        self.plotPandR2D(valData["label"][sigMaskDataVal], y_val_disc1[sigMaskDataVal], y_val_disc2[sigMaskDataVal], sample_weight=valData["weight"][sigMaskDataVal], name="disc2")

        masses = ["350", "550", "850", "1150"]

        for mass in masses:

            tempMassMaskDataVal  = valData["mass"]==float(mass)
            tempMassMaskDataVal |= valData["mass"]==float(173.0)

            self.plotPandR2D(valData["label"][tempMassMaskDataVal&sigMaskDataVal], y_val_disc2[tempMassMaskDataVal&sigMaskDataVal], y_val_disc1[tempMassMaskDataVal&sigMaskDataVal], sample_weight=valData["weight"][tempMassMaskDataVal&sigMaskDataVal], name="disc1_mass{}".format(mass))
            self.plotPandR2D(valData["label"][tempMassMaskDataVal&sigMaskDataVal], y_val_disc1[tempMassMaskDataVal&sigMaskDataVal], y_val_disc2[tempMassMaskDataVal&sigMaskDataVal], sample_weight=valData["weight"][tempMassMaskDataVal&sigMaskDataVal], name="disc2_mass{}".format(mass))


        for key in self.metric:
            print(key, self.metric[key])

        self.config["metric"] = self.metric
        with open(self.config["outputDir"]+"/config.json",'w') as configFile:
            json.dump(self.config, configFile, indent=4, sort_keys=True)

        return self.metric
