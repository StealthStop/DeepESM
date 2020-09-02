from DataGetter import get_data, getSamplesToRun
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.ROOT,hep.style.CMS]) # For now ROOT defaults to CMS
plt.style.use({'legend.frameon':False,'legend.fontsize':16,'legend.edgecolor':'black'})
from matplotlib.colors import LogNorm
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

        corr = cor.pearson_corr(xIn, yIn)

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

    def plotDisc(self, hists, colors, labels, weights, bins, name, xlab, ylab):

        # Plot predicted mass
        fig, ax = plt.subplots(figsize=(10, 10))
        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel(xlab); ax.set_xlabel(ylab)

        for i in range(0, len(hists)): plt.hist(hists[i], bins, color="xkcd:"+colors[i], alpha=0.9, histtype='step', lw=2, label=labels[i], density=True, log=self.doLog, weights=weights[i])

        ax.legend(loc='best', frameon=False)
        fig.savefig(self.config["outputDir"]+"/%s.png"%(name), dpi=fig.dpi)        

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

    def plotDiscPerNjet(self, tag, samples):

        for sample in samples:
            trainSample = samples[sample][0]
            y_train_Sp = samples[sample][1]
            weights = samples[sample][2] 
            bins = np.linspace(0, 1, 100)
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
            fig.savefig(self.config["outputDir"]+"/discriminator_nJet_"+sample+tag+".png", dpi=fig.dpi)
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

    def plotDiscVsNew(self, b, bnew, s, snew):

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax1)
        ax1.scatter(b, bnew, s=10, c='b', marker="s", label='background')
        ax1.scatter(s, snew, s=10, c='r', marker="o", label='signal')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        plt.legend(loc='best');
        fig.savefig(self.config["outputDir"]+"/discriminator_discriminator_new.png", dpi=fig.dpi)        
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

    def makePlots(self):

        sgValSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+"*Val.root") for mass in self.config["massModels"]) , [])
        bgValSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+ttbar+"*Val.root") for ttbar in self.config["ttbarMC"][1]), [])
        sgOTrainSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+"*Val.root") for mass in self.config["othermassModels"]) , [])
        bgOTrainSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+ttbar+"*Val.root") for ttbar in self.config["otherttbarMC"][1]), [])

        valData, valSg, valBg = get_data(sgValSet, bgValSet, self.config)
        trainOData, trainOSg, trainOBg = get_data(sgOTrainSet, bgOTrainSet, self.config)

        output_Val, output_Val_Sg, output_Val_Bg = self.getOutput(self.model, valData["data"], valSg["data"], valBg["data"])
        output_Train, output_Train_Sg, output_Train_Bg = self.getOutput(self.model, self.trainData["data"], self.trainSg["data"], self.trainBg["data"])
        output_OTrain, output_OTrain_Sg, output_OTrain_Bg = self.getOutput(self.model, trainOData["data"], trainOSg["data"], trainOBg["data"])

        y_Val, y_Val_Sg, y_Val_Bg = self.getResults(output_Val, output_Val_Sg, output_Val_Bg, 0, 0)
        y_Val_new, y_Val_Sg_new, y_Val_Bg_new = self.getResults(output_Val, output_Val_Sg, output_Val_Bg, 1, 0)
        y_Val_mass, y_Val_mass_Sg, y_Val_mass_Bg = self.getResults(output_Val, output_Val_Sg, output_Val_Bg, 4, 0)
        y_Train, y_Train_Sg, y_Train_Bg = self.getResults(output_Train, output_Train_Sg, output_Train_Bg, 0, 0)
        y_Train_new, y_Train_Sg_new, y_Train_Bg_new = self.getResults(output_Train, output_Train_Sg, output_Train_Bg, 1, 0)
        y_Train_mass, y_Train_mass_Sg, y_Train_mass_Bg = self.getResults(output_Train, output_Train_Sg, output_Train_Bg, 4, 0)
        y_OTrain, y_OTrain_Sg, y_OTrain_Bg = self.getResults(output_OTrain, output_OTrain_Sg, output_OTrain_Bg, 0, 0)

        colors = ["red", "green", "blue", "magenta"]; labels = ["Sg Train", "Sg Val", "Bg Train", "Bg Val"]
        self.plotDisc([y_Train_mass_Sg, y_Val_mass_Sg, y_Train_mass_Bg, y_Val_mass_Bg], colors, labels, [self.trainSg["Weight"], valSg["Weight"], self.trainBg["Weight"], valBg["Weight"]], np.linspace(0, 1500, 150), "mass", 'Norm Events', 'predicted mass')

        self.plotDisc([y_Train_Sg, y_Val_Sg, y_Train_Bg, y_Val_Bg], colors, labels, [self.trainSg["Weight"], valSg["Weight"], self.trainBg["Weight"], valBg["Weight"]], np.linspace(0, 1, 100), "discriminator", 'Norm Events', 'Discriminator')

        self.plotDisc([y_Train_Sg_new, y_Val_Sg_new, y_Train_Bg_new, y_Val_Bg_new], colors, labels, [self.trainSg["Weight"], valSg["Weight"], self.trainBg["Weight"], valBg["Weight"]], np.linspace(0, 1, 100), "discriminator_new", 'Norm Events', 'Discriminator new')

        self.plotDisc([y_Train_mass_Sg[self.trainSg["mask_m550"]], y_Train_mass_Sg[self.trainSg["mask_m850"]], y_Train_mass_Sg[self.trainSg["mask_m1200"]], y_Train_mass_Bg], colors, ["mass 550", "mass 850", "mass 1200", "ttbar"], [self.trainSg["Weight"][self.trainSg["mask_m550"]], self.trainSg["Weight"][self.trainSg["mask_m850"]], self.trainSg["Weight"][self.trainSg["mask_m1200"]], self.trainBg["Weight"]], np.linspace(0, 1500, 150), "mass_split", 'Norm Events', 'predicted mass')

        self.plotDiscVsNew(y_Train_Bg, y_Train_Bg_new, y_Train_Sg, y_Train_Sg_new)
        
        fig = plt.figure() 
        corr = cor.pearson_corr(y_Train_Bg, y_Train_Bg_new)
        plt.hist2d(y_Train_Bg, y_Train_Bg_new, bins=[100, 100], range=[[0, 1], [0, 1]], cmap=plt.cm.jet, weights=self.trainBg["Weight"][:,0], cmin = self.trainSg["Weight"][:,0].min())
        plt.colorbar()
        plt.text(0.65, 0.08, r"$\bf{CC}$ = %.3f"%(corr), fontfamily='sans-serif', fontsize=24, bbox=dict(facecolor='white', alpha=1.0))
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        fig.savefig(self.config["outputDir"]+"/2D_BG_discriminators.png", dpi=fig.dpi)
        
        fig = plt.figure() 
        corr = cor.pearson_corr(y_Train_Sg, y_Train_Sg_new)
        plt.hist2d(y_Train_Sg, y_Train_Sg_new, bins=[100, 100], range=[[0, 1], [0, 1]], cmap=plt.cm.jet, weights=self.trainSg["Weight"][:,0], cmin = self.trainSg["Weight"][:,0].min())
        plt.colorbar()
        plt.text(0.65, 0.08, r"$\bf{CC}$ = %.3f"%(corr), fontfamily='sans-serif', fontsize=24, bbox=dict(facecolor='white', alpha=1.0))
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        fig.savefig(self.config["outputDir"]+"/2D_SG_discriminators.png", dpi=fig.dpi)

        # Plot Acc vs Epoch
        self.plotAccVsEpoch('loss', 'val_loss', 'model loss', 'loss_train_val')
        for rank in ["first", "second", "third", "fourth"]: self.plotAccVsEpoch('%s_output_loss'%(rank), 'val_%s_output_loss'%(rank), '%s output loss'%(rank), '%s_output_loss_train_val'%(rank))        
        self.plotAccVsEpoch('correlation_layer_loss', 'val_correlation_layer_loss', 'correlation_layer output loss', 'correlation_layer_loss_train_val')

        # Plot disc per njet
        self.plotDiscPerNjet("",     {"Bg": [self.trainBg, y_Train_Bg,     self.trainBg["Weight"]], "Sg": [self.trainSg, y_Train_Sg,     self.trainSg["Weight"]]})
        self.plotDiscPerNjet("_new", {"Bg": [self.trainBg, y_Train_Bg_new, self.trainBg["Weight"]], "Sg": [self.trainSg, y_Train_Sg_new, self.trainSg["Weight"]]})

        # Plot validation roc curve
        fpr_Val, tpr_Val, thresholds_Val = roc_curve(valData["labels"][:,0], y_Val, sample_weight=valData["Weight"][:,0])
        fpr_Val_new, tpr_Val_new, thresholds_Val_new = roc_curve(valData["labels"][:,0], y_Val_new, sample_weight=valData["Weight"][:,0])
        fpr_Train, tpr_Train, thresholds_Train = roc_curve(self.trainData["labels"][:,0], y_Train, sample_weight=self.trainData["Weight"][:,0])
        fpr_Train_new, tpr_Train_new, thresholds_Train_new = roc_curve(self.trainData["labels"][:,0], y_Train_new, sample_weight=self.trainData["Weight"][:,0])
        fpr_OTrain, tpr_OTrain, thresholds_OTrain = roc_curve(trainOData["labels"][:,0], y_OTrain, sample_weight=trainOData["Weight"][:,0])
        auc_Val = roc_auc_score(valData["labels"][:,0], y_Val)
        auc_Val_new = roc_auc_score(valData["labels"][:,0], y_Val_new)
        auc_Train = roc_auc_score(self.trainData["labels"][:,0], y_Train)
        auc_Train_new = roc_auc_score(self.trainData["labels"][:,0], y_Train_new)
        auc_OTrain = roc_auc_score(trainOData["labels"][:,0], y_OTrain)
        
        # Define metrics for the training
        self.metric["OverTrain"] = abs(auc_Val - auc_Train)
        self.metric["OverTrain_new"] = abs(auc_Val_new - auc_Train_new)
        self.metric["Performance"] = abs(1 - auc_Train)
        self.metric["Performance_new"] = abs(1 - auc_Train_new)

        # Plot some ROC curves
        self.plotROC("", None, None, fpr_Val, tpr_Val, fpr_Train, tpr_Train, auc_Val, auc_Train)
        self.plotROC("_new", None, None, fpr_Val_new, tpr_Val_new, fpr_Train_new, tpr_Train_new, auc_Val_new, auc_Train_new)
        self.plotROC("_TT_TTJets", None, None, fpr_OTrain, tpr_OTrain, fpr_Train, tpr_Train, auc_OTrain, auc_Train)
        self.plotROC("_"+self.config["ttbarMC"][0]+"_nJet", y_Train, self.trainData)
        self.plotROC("_"+self.config["ttbarMC"][0]+"_nJet_new", y_Train_new, self.trainData)
        self.plotROC("_"+self.config["otherttbarMC"][0]+"_nJet", y_OTrain, trainOData) 

        # Plot validation precision recall
        precision_Val, recall_Val, _ = precision_recall_curve(valData["labels"][:,0], y_Val, sample_weight=valData["Weight"][:,0])
        precision_Train, recall_Train, _ = precision_recall_curve(self.trainData["labels"][:,0], y_Train, sample_weight=self.trainData["Weight"][:,0])
        ap_Val = average_precision_score(valData["labels"][:,0], y_Val, sample_weight=valData["Weight"][:,0])
        ap_Train = average_precision_score(self.trainData["labels"][:,0], y_Train, sample_weight=self.trainData["Weight"][:,0])
        
        self.plotPandR(precision_Val, recall_Val, precision_Train, recall_Train, ap_Val, ap_Train)
        
        # Plot NJet dependance
        binxl = self.config["minNJetBin"]
        binxh = self.config["maxNJetBin"] + 1
        numbin = binxh - binxl        
        self.plot2DVar(name="nJet", binxl=binxl, binxh=binxh, numbin=numbin, xIn=self.trainBg["nJet"][:,0], yIn=y_Train_Bg, nbiny=50)
        
        # Save useful stuff
        self.trainData["y"] = y_Train
        np.save(self.config["outputDir"]+"/deepESMbin_dis_nJet.npy", self.trainData)
        
        for key in self.metric:
            print(key, self.metric[key])
        
        self.config["metric"] = self.metric
        with open(self.config["outputDir"]+"/config.json",'w') as configFile:
            json.dump(self.config, configFile, indent=4, sort_keys=True)

        return self.metric
