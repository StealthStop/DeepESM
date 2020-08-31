from DataGetter import get_data, getSamplesToRun
import numpy as np
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

    def plot(self):
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

        # Plot predicted mass
        bins = np.linspace(0, 1500, 150)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel('Norm Events')
        ax.set_xlabel('predicted mass')
        plt.hist(y_Train_mass_Sg, bins, color='xkcd:red', alpha=0.9, histtype='step', lw=2, label='Sg Train', density=True, log=self.doLog, weights=self.trainSg["Weight"])
        plt.hist(y_Val_mass_Sg, bins, color='xkcd:green', alpha=0.9, histtype='step', lw=2, label='Sg Val', density=True, log=self.doLog, weights=valSg["Weight"])
        plt.hist(y_Train_mass_Bg, bins, color='xkcd:blue', alpha=0.9, histtype='step', lw=2, label='Bg Train', density=True, log=self.doLog, weights=self.trainBg["Weight"])
        plt.hist(y_Val_mass_Bg, bins, color='xkcd:magenta', alpha=0.9, histtype='step', lw=2, label='Bg Val', density=True, log=self.doLog, weights=valBg["Weight"])
        #plt.yscale('log')
        ax.legend(loc='best', frameon=False)
        fig.savefig(self.config["outputDir"]+"/mass.png", dpi=fig.dpi)        

        bins = np.linspace(0, 1500, 150)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel('Norm Events')
        ax.set_xlabel('predicted mass')

        #plt.hist(y_Train_mass_Sg[self.trainSg["mask_m350"]], bins, color='xkcd:black', alpha=0.9, histtype='step', lw=2, label='mass 350', 
        #         density=True, log=self.doLog, weights=self.trainSg["Weight"][self.trainSg["mask_m350"]])
        plt.hist(y_Train_mass_Sg[self.trainSg["mask_m550"]], bins, color='xkcd:red', alpha=0.9, histtype='step', lw=2, label='mass 550', 
                 density=True, log=self.doLog, weights=self.trainSg["Weight"][self.trainSg["mask_m550"]])
        plt.hist(y_Train_mass_Sg[self.trainSg["mask_m850"]], bins, color='xkcd:green', alpha=0.9, histtype='step', lw=2, label='mass 850', 
                 density=True, log=self.doLog, weights=self.trainSg["Weight"][self.trainSg["mask_m850"]])
        plt.hist(y_Train_mass_Sg[self.trainSg["mask_m1200"]], bins, color='xkcd:blue', alpha=0.9, histtype='step', lw=2, label='mass 1200', 
                 density=True, log=self.doLog, weights=self.trainSg["Weight"][self.trainSg["mask_m1200"]])
        plt.hist(y_Train_mass_Bg, bins, color='xkcd:magenta', alpha=0.9, histtype='step', lw=2, label='ttbar', density=True, log=self.doLog, weights=self.trainBg["Weight"])
        #plt.yscale('log')
        ax.legend(loc='best', frameon=False)
        fig.savefig(self.config["outputDir"]+"/mass_split.png", dpi=fig.dpi)        

        # Plot discriminator distribution
        bins = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel('Norm Events')
        ax.set_xlabel('Discriminator')
        plt.hist(y_Train_Sg, bins, color='xkcd:red', alpha=0.9, histtype='step', lw=2, label='Sg Train', density=True, log=self.doLog, weights=self.trainSg["Weight"])
        plt.hist(y_Val_Sg, bins, color='xkcd:green', alpha=0.9, histtype='step', lw=2, label='Sg Val', density=True, log=self.doLog, weights=valSg["Weight"])
        plt.hist(y_Train_Bg, bins, color='xkcd:blue', alpha=0.9, histtype='step', lw=2, label='Bg Train', density=True, log=self.doLog, weights=self.trainBg["Weight"])
        plt.hist(y_Val_Bg, bins, color='xkcd:magenta', alpha=0.9, histtype='step', lw=2, label='Bg Val', density=True, log=self.doLog, weights=valBg["Weight"])
        ax.legend(loc='best', frameon=False)
        fig.savefig(self.config["outputDir"]+"/discriminator.png", dpi=fig.dpi)
        
        bins = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel('Norm Events')
        ax.set_xlabel('Discriminator new')
        plt.hist(y_Train_Sg_new, bins, color='xkcd:red', alpha=0.9, histtype='step', lw=2, label='Sg Train', density=True, log=self.doLog, weights=self.trainSg["Weight"])
        plt.hist(y_Val_Sg_new, bins, color='xkcd:green', alpha=0.9, histtype='step', lw=2, label='Sg Val', density=True, log=self.doLog, weights=valSg["Weight"])
        plt.hist(y_Train_Bg_new, bins, color='xkcd:blue', alpha=0.9, histtype='step', lw=2, label='Bg Train', density=True, log=self.doLog, weights=self.trainBg["Weight"])
        plt.hist(y_Val_Bg_new, bins, color='xkcd:magenta', alpha=0.9, histtype='step', lw=2, label='Bg Val', density=True, log=self.doLog, weights=valBg["Weight"])
        ax.legend(loc='best', frameon=False)
        fig.savefig(self.config["outputDir"]+"/discriminator_new.png", dpi=fig.dpi)
        
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1 = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax1)
        ax1.scatter(y_Train_Sg, y_Train_Sg_new, s=10, c='r', marker="o", label='signal')
        ax1.scatter(y_Train_Bg, y_Train_Bg_new, s=10, c='b', marker="s", label='background')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        plt.legend(loc='best');
        fig.savefig(self.config["outputDir"]+"/discriminator_discriminator_new1.png", dpi=fig.dpi)        

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1 = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax1)
        ax1.scatter(y_Train_Bg, y_Train_Bg_new, s=10, c='b', marker="s", label='background')
        ax1.scatter(y_Train_Sg, y_Train_Sg_new, s=10, c='r', marker="o", label='signal')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        plt.legend(loc='best');
        fig.savefig(self.config["outputDir"]+"/discriminator_discriminator_new2.png", dpi=fig.dpi)        
        
        fig = plt.figure() 
        plt.hist2d(y_Train_Bg, y_Train_Bg_new, bins=[100, 100], range=[[0, 1], [0, 1]], cmap=plt.cm.jet, weights=self.trainBg["Weight"][:,0], cmin = self.trainSg["Weight"][:,0].min())
        plt.colorbar()
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        fig.savefig(self.config["outputDir"]+"/2D_BG_discriminators.png", dpi=fig.dpi)
        
        fig = plt.figure() 
        plt.hist2d(y_Train_Sg, y_Train_Sg_new, bins=[100, 100], range=[[0, 1], [0, 1]], cmap=plt.cm.jet, weights=self.trainSg["Weight"][:,0], cmin = self.trainSg["Weight"][:,0].min())
        plt.colorbar()
        hep.cms.label(data=True, paper=False, year=self.config["year"])
        fig.savefig(self.config["outputDir"]+"/2D_SG_discriminators.png", dpi=fig.dpi)
        
        ## Make input variable plots
        #index=0
        #for var in self.config["allVars"]:
        #    fig = plt.figure()
        #    plt.hist(self.trainBg["data"][:,index], bins=30, histtype='step', density=True, log=False, label=var+" Bg", weights=self.trainBg["Weight"])
        #    plt.hist(self.trainSg["data"][:,index], bins=30, histtype='step', density=True, log=False, label=var+" Sg", weights=self.trainSg["Weight"])
        #    plt.legend(loc='best')
        #    plt.ylabel('norm')
        #    plt.xlabel(var)
        #    fig.savefig(self.config["outputDir"]+"/"+var+".png", dpi=fig.dpi)
        #    index += 1
        #
        ## Normalize
        #index=0
        #tBg = self.trainData["scale"]*(self.trainBg["data"] - self.trainData["mean"])
        #tSg = self.trainData["scale"]*(self.trainSg["data"] - self.trainData["mean"])
        #for var in self.config["allVars"]:
        #    fig = plt.figure()
        #    plt.hist(tBg[:,index], bins=30, histtype='step', density=True, log=False, label=var+" Bg", weights=self.trainBg["Weight"])
        #    plt.hist(tSg[:,index], bins=30, histtype='step', density=True, log=False, label=var+" Sg", weights=self.trainSg["Weight"])
        #    plt.legend(loc='best')
        #    plt.ylabel('norm')
        #    plt.xlabel("norm "+var)
        #    fig.savefig(self.config["outputDir"]+"/norm_"+var+".png", dpi=fig.dpi)
        #    index += 1
        
        # Plot loss of training vs test
        print("Plotting loss and acc vs epoch")
        fig = plt.figure()
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        plt.plot(self.result_log.history['loss'])
        plt.plot(self.result_log.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        fig.savefig(self.config["outputDir"]+"/loss_train_val.png", dpi=fig.dpi)
        
        fig = plt.figure()
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        plt.plot(self.result_log.history['first_output_loss'])
        plt.plot(self.result_log.history['val_first_output_loss'])
        plt.title('first output loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        fig.savefig(self.config["outputDir"]+"/first_output_loss_train_val.png", dpi=fig.dpi)
        
        fig = plt.figure()
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        plt.plot(self.result_log.history['second_output_loss'])
        plt.plot(self.result_log.history['val_second_output_loss'])
        plt.title('second output loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        fig.savefig(self.config["outputDir"]+"/second_output_loss_train_val.png", dpi=fig.dpi)

        fig = plt.figure()
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        plt.plot(self.result_log.history['third_output_loss'])
        plt.plot(self.result_log.history['val_third_output_loss'])
        plt.title('third output loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        fig.savefig(self.config["outputDir"]+"/third_output_loss_train_val.png", dpi=fig.dpi)

        fig = plt.figure()
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        plt.plot(self.result_log.history['fourth_output_loss'])
        plt.plot(self.result_log.history['val_fourth_output_loss'])
        plt.title('fourth output loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        fig.savefig(self.config["outputDir"]+"/fourth_output_loss_train_val.png", dpi=fig.dpi)
        
        fig = plt.figure()
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        plt.plot(self.result_log.history['correlation_layer_loss'])
        plt.plot(self.result_log.history['val_correlation_layer_loss'])
        plt.title('correlation_layer output loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        fig.savefig(self.config["outputDir"]+"/correlation_layer_loss_train_val.png", dpi=fig.dpi)

        # Plot discriminator distribution
        bins = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        ax.set_ylabel('Norm Events')
        ax.set_xlabel('Discriminator')
        plt.hist(y_Train_Sg, bins, color='xkcd:red', alpha=0.9, histtype='step', lw=2, label='Sg Train', density=True, log=self.doLog, weights=self.trainSg["Weight"])
        plt.hist(y_Val_Sg, bins, color='xkcd:green', alpha=0.9, histtype='step', lw=2, label='Sg Val', density=True, log=self.doLog, weights=valSg["Weight"])
        plt.hist(y_Train_Bg, bins, color='xkcd:blue', alpha=0.9, histtype='step', lw=2, label='Bg Train', density=True, log=self.doLog, weights=self.trainBg["Weight"])
        plt.hist(y_Val_Bg, bins, color='xkcd:magenta', alpha=0.9, histtype='step', lw=2, label='Bg Val', density=True, log=self.doLog, weights=valBg["Weight"])
        ax.legend(loc='best', frameon=False)
        fig.savefig(self.config["outputDir"]+"/discriminator.png", dpi=fig.dpi)
        
        samples = {"Bg": [self.trainBg, y_Train_Bg, self.trainBg["Weight"]], "Sg": [self.trainSg, y_Train_Sg, self.trainSg["Weight"]]}
        for sample in samples:
            trainSample = samples[sample][0]
            y_train_Sp = samples[sample][1]
            weights = samples[sample][2] 
            bins = np.linspace(0, 1, 100)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
            ax.set_ylabel('Norm Events')
            ax.set_xlabel('Discriminator')
            for key in sorted(trainSample.keys()):
                if key.find("mask_nJet") != -1:
                    yt = y_train_Sp[trainSample[key]]                
                    wt = weights[trainSample[key]]
                    if yt.size != 0 and wt.size != 0:
                        plt.hist(yt, bins, alpha=0.9, histtype='step', lw=2, label=sample+" Train "+key, density=True, log=self.doLog, weights=wt)
            plt.legend(loc='best')
            fig.savefig(self.config["outputDir"]+"/discriminator_nJet_"+sample+".png", dpi=fig.dpi)

        samples = {"Bg": [self.trainBg, y_Train_Bg_new, self.trainBg["Weight"]], "Sg": [self.trainSg, y_Train_Sg_new, self.trainSg["Weight"]]}
        for sample in samples:
            trainSample = samples[sample][0]
            y_train_Sp = samples[sample][1]
            weights = samples[sample][2] 
            bins = np.linspace(0, 1, 100)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
            ax.set_ylabel('Norm Events')
            ax.set_xlabel('Discriminator new')
            for key in sorted(trainSample.keys()):
                if key.find("mask_nJet") != -1:
                    yt = y_train_Sp[trainSample[key]]                
                    wt = weights[trainSample[key]]
                    if yt.size != 0 and wt.size != 0:
                        plt.hist(yt, bins, alpha=0.9, histtype='step', lw=2, label=sample+" Train "+key, density=True, log=self.doLog, weights=wt)
            plt.legend(loc='best')
            fig.savefig(self.config["outputDir"]+"/discriminator_nJet_"+sample+"_new.png", dpi=fig.dpi)
        
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
        
        fig = plt.figure()
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_Val, tpr_Val, color='xkcd:black', label='Val (area = {:.3f})'.format(auc_Val))
        plt.plot(fpr_Train, tpr_Train, color='xkcd:red', label='Train (area = {:.3f})'.format(auc_Train))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/roc_plot.png", dpi=fig.dpi)

        fig = plt.figure()
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_Val_new, tpr_Val_new, color='xkcd:black', label='Val (area = {:.3f})'.format(auc_Val_new))
        plt.plot(fpr_Train_new, tpr_Train_new, color='xkcd:red', label='Train (area = {:.3f})'.format(auc_Train_new))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve new')
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/roc_plot_new.png", dpi=fig.dpi)
        
        fig = plt.figure()
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_OTrain, tpr_OTrain, color='xkcd:black', label="Train "+self.config["otherttbarMC"][0]+" (area = {:.3f})".format(auc_OTrain))
        plt.plot(fpr_Train, tpr_Train, color='xkcd:red', label="Train "+self.config["ttbarMC"][0]+" (area = {:.3f})".format(auc_Train))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/roc_plot_TT_TTJets.png", dpi=fig.dpi)
        
        fig = plt.figure()
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        njetPerformance = []
        for key in sorted(self.trainData.keys()):
            if key.find("mask_nJet") != -1:
                labels = self.trainData["labels"][self.trainData[key]]
                weights = self.trainData["Weight"][self.trainData[key]][:,0]
                y = y_Train[self.trainData[key]]
                if len(y)==0:
                    continue
                fpr_Train, tpr_Train, thresholds_Train = roc_curve(labels[:,0], y, sample_weight=weights)
                auc_Train = roc_auc_score(labels[:,0], y)    
                njetPerformance.append(auc_Train)
                plt.plot(fpr_Train, tpr_Train, label="Train "+key+" (area = {:.3f})".format(auc_Train))
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/roc_plot_"+self.config["ttbarMC"][0]+"_nJet.png", dpi=fig.dpi)    

        fig = plt.figure()
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve new')
        njetPerformance = []
        for key in sorted(self.trainData.keys()):
            if key.find("mask_nJet") != -1:
                labels = self.trainData["labels"][self.trainData[key]]
                weights = self.trainData["Weight"][self.trainData[key]][:,0]
                y = y_Train_new[self.trainData[key]]
                if len(y)==0:
                    continue
                fpr_Train, tpr_Train, thresholds_Train = roc_curve(labels[:,0], y, sample_weight=weights)
                auc_Train = roc_auc_score(labels[:,0], y)    
                njetPerformance.append(auc_Train)
                plt.plot(fpr_Train, tpr_Train, label="Train "+key+" (area = {:.3f})".format(auc_Train))
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/roc_plot_"+self.config["ttbarMC"][0]+"_nJet_new.png", dpi=fig.dpi)    
        
        fig = plt.figure()
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(self.config["otherttbarMC"][0]+" ROC curve")
        njetPerformance = []
        for key in sorted(trainOData.keys()):
            if key.find("mask_nJet") != -1:
                labels = trainOData["labels"][trainOData[key]]
                weights = trainOData["Weight"][trainOData[key]][:,0]
                y = y_OTrain[trainOData[key]]
                if len(y)==0:
                    continue
                fpr_Train, tpr_Train, thresholds_Train = roc_curve(labels[:,0], y, sample_weight=weights)
                auc_Train = roc_auc_score(labels[:,0], y)    
                njetPerformance.append(auc_Train)
                plt.plot(fpr_Train, tpr_Train, label="Train "+key+" (area = {:.3f})".format(auc_Train))
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/roc_plot_"+self.config["otherttbarMC"][0]+"_nJet.png", dpi=fig.dpi)    

        # Plot validation precision recall
        precision_Val, recall_Val, _ = precision_recall_curve(valData["labels"][:,0], y_Val, sample_weight=valData["Weight"][:,0])
        precision_Train, recall_Train, _ = precision_recall_curve(self.trainData["labels"][:,0], y_Train, sample_weight=self.trainData["Weight"][:,0])
        ap_Val = average_precision_score(valData["labels"][:,0], y_Val, sample_weight=valData["Weight"][:,0])
        ap_Train = average_precision_score(self.trainData["labels"][:,0], y_Train, sample_weight=self.trainData["Weight"][:,0])
        
        fig = plt.figure()
        ax = hep.cms.label(data=True, paper=False, year=self.config["year"], ax=ax)
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.plot(precision_Val, recall_Val, color='xkcd:black', label='Val (AP = {:.3f})'.format(ap_Val))
        plt.plot(precision_Train, recall_Train, color='xkcd:red', label='Train (AP = {:.3f})'.format(ap_Train))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision and Recall curve')
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/PandR_plot.png", dpi=fig.dpi)        
        
        # Plot NJet dependance
        binxl = self.config["minNJetBin"]
        binxh = self.config["maxNJetBin"] + 1
        numbin = binxh - binxl        
        self.plot2DVar(name="nJet", binxl=binxl, binxh=binxh, numbin=numbin, xIn=self.trainBg["nJet"][:,0], yIn=y_Train_Bg, nbiny=50)
        #for i in range(len(self.config["allVars"])):
        #    binxl = np.amin(self.trainBg["data"][:,i])
        #    binxh = np.amax(self.trainBg["data"][:,i])
        #    numbin = abs(int(binxh - binxl))
        #    plot2DVar(name=self.config["allVars"][i], binxl=binxl, binxh=binxh, numbin=numbin, xIn=self.trainBg["data"][:,i], yIn=y_Train_Bg, nbiny=50)
        #
        ## Make njet distribution for 4 different bins
        #nMVABins = 4
        #nJetDeepESMBins = None
        #sorted_y_split = None
        #for key in sorted(trainOBg.keys()):
        #    if key.find("mask_nJet") != -1:
        #        y = y_OTrain_Bg[trainOBg[key]]
        #        nJet = trainOBg["nJet"][:,0][trainOBg[key]]
        #        inds = y.argsort()
        #        sortednJet = nJet[inds[::-1]]
        #        sorted_y = y[inds[::-1]]
        #        perNJetBins = np.array_split(sortednJet, nMVABins)
        #        perNJet_y_split = np.array_split(sorted_y, nMVABins)
        #        perNJet_bins = []
        #        print("-----------------------------------------------------------------------------------------------------------------------------------")
        #        for i in range(len(perNJet_y_split)):
        #            print("NJet: ", key, " DeepESM bin ", len(perNJet_y_split) - i, ": "," bin cuts: ", perNJet_y_split[i][0], " ", perNJet_y_split[i][-1])
        #            perNJet_bins.append([str(perNJet_y_split[i][0]), str(perNJet_y_split[i][-1])]) 
        #        self.config[key] = perNJet_bins 
        #        if nJetDeepESMBins == None:
        #            nJetDeepESMBins = perNJetBins
        #            sorted_y_split = perNJet_y_split
        #        else:
        #            for i in range(len(nJetDeepESMBins)):
        #                nJetDeepESMBins[i] = np.hstack((nJetDeepESMBins[i], perNJetBins[i]))
        #                sorted_y_split[i] = np.hstack((sorted_y_split[i], perNJet_y_split[i]))
        #        
        #index=0
        #fig = plt.figure()
        #bins = []
        #for a in nJetDeepESMBins:
        #    print("DeepESM bin ", len(nJetDeepESMBins) - index, ": ", " NEvents: ", len(a)," bin cuts: ", sorted_y_split[index][0], " ", sorted_y_split[index][-1])
        #    plt.hist(a, bins=numbin, range=(binxl, binxh), histtype='step', density=True, log=self.doLog, label='Bin {}'.format(len(nJetDeepESMBins) - index))
        #    bins.append([str(sorted_y_split[index][0]), str(sorted_y_split[index][-1])])
        #    index += 1
        #plt.hist(self.trainBg["nJet"][:,0], bins=numbin, range=(binxl, binxh), histtype='step', density=True, log=self.doLog, label='Total')
        #plt.legend(loc='best')
        #fig.savefig(self.config["outputDir"]+"/nJet_log.png", dpi=fig.dpi)
        #
        #index=0
        #MVABinNJetShapeContent = []
        #fig = plt.figure()
        #for a in nJetDeepESMBins:
        #    n, _, _ = plt.hist(a, bins=numbin, range=(binxl, binxh), histtype='step', density=True, log=False, label='Bin {}'.format(len(nJetDeepESMBins) - index))
        #    MVABinNJetShapeContent.append(n)
        #    index += 1
        #TotalMVAnJetShape, _, _ = plt.hist(self.trainBg["nJet"][:,0], bins=numbin, range=(binxl, binxh), histtype='step', density=True, log=False, label='Total')
        #plt.legend(loc='best')
        #fig.savefig(self.config["outputDir"]+"/nJet.png", dpi=fig.dpi)
        
        # Save useful stuff
        self.trainData["y"] = y_Train
        np.save(self.config["outputDir"]+"/deepESMbin_dis_nJet.npy", self.trainData)
        
        for key in self.metric:
            print(key, self.metric[key])
        
        self.config["metric"] = self.metric
        with open(self.config["outputDir"]+"/config.json",'w') as configFile:
            json.dump(self.config, configFile, indent=4, sort_keys=True)

        return self.metric
