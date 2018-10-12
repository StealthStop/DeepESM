from glob import glob
from DataGetter import get_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, auc
import json

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
        
    def plot2DVar(self, name, binxl, binxh, numbin, xIn, yIn, nbiny):
        fig = plt.figure()
        h, xedges, yedges, image = plt.hist2d(xIn, yIn, bins=[numbin, nbiny], range=[[binxl, binxh], [0, 1]], cmap=plt.cm.binary)
        plt.colorbar()
    
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

    def plot(self):
        sgValSet = sum( (glob(self.config["dataSet"]+"trainingTuple_*_division_1_*_"+mass+"*_validation_0.h5") for mass in self.config["massModels"]) , [])
        bgValSet = sum( (glob(self.config["dataSet"]+"trainingTuple_*_division_1_"+ttbar+"_validation_0.h5") for ttbar in self.config["ttbarMC"][1]) , [])
        bgOTrainSet = sum( (glob(self.config["dataSet"]+"trainingTuple_*_division_0_"+ttbar+"_training_0.h5") for ttbar in self.config["otherttbarMC"][1]) , [])

        valData, valSg, valBg = get_data(sgValSet, bgValSet, self.config)
        trainOData, trainOSg, trainOBg = get_data(self.sgTrainSet, bgOTrainSet, self.config)
        y_Val = self.model.predict(valData["data"])[0][:,0].ravel()
        y_Val_Sg = self.model.predict(valSg["data"])[0][:,0].ravel()
        y_Val_Bg = self.model.predict(valBg["data"])[0][:,0].ravel()

        y_Train = self.model.predict(self.trainData["data"])[0][:,0].ravel()
        y_Train_Sg = self.model.predict(self.trainSg["data"])[0][:,0].ravel()
        y_Train_Bg = self.model.predict(self.trainBg["data"])[0][:,0].ravel()

        y_OTrain = self.model.predict(trainOData["data"])[0][:,0].ravel()
        y_OTrain_Bg = self.model.predict(trainOBg["data"])[0][:,0].ravel()
    
        ## Make input variable plots
        #index=0
        #for var in self.config["allVars"]:
        #    fig = plt.figure()
        #    plt.hist(self.trainBg["data"][:,index], bins=30, histtype='step', density=True, log=False, label=var+" Bg", weights=self.trainBg["Weight"])
        #    plt.hist(self.trainSg["data"][:,index], bins=30, histtype='step', density=True, log=False, label=var+" Sg", weights=self.trainSg["Weight"])
        #    plt.legend(loc='upper right')
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
        #    plt.legend(loc='upper right')
        #    plt.ylabel('norm')
        #    plt.xlabel("norm "+var)
        #    fig.savefig(self.config["outputDir"]+"/norm_"+var+".png", dpi=fig.dpi)
        #    index += 1

        # Plot loss of training vs test
        fig = plt.figure()
        plt.plot(self.result_log.history['loss'])
        plt.plot(self.result_log.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig.savefig(self.config["outputDir"]+"/loss_train_val.png", dpi=fig.dpi)
        
        fig = plt.figure()
        plt.plot(self.result_log.history['first_output_loss'])
        plt.plot(self.result_log.history['val_first_output_loss'])
        plt.title('first output loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig.savefig(self.config["outputDir"]+"/first_output_loss_train_val.png", dpi=fig.dpi)
        
        fig = plt.figure()
        plt.plot(self.result_log.history['first_output_acc'])
        plt.plot(self.result_log.history['val_first_output_acc'])
        plt.title('first output acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig.savefig(self.config["outputDir"]+"/first_output_acc_train_val.png", dpi=fig.dpi)
        
        fig = plt.figure()
        plt.plot(self.result_log.history['second_output_loss'])
        plt.plot(self.result_log.history['val_second_output_loss'])
        plt.title('second output loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig.savefig(self.config["outputDir"]+"/second_output_loss_train_val.png", dpi=fig.dpi)
        
        fig = plt.figure()
        plt.plot(self.result_log.history['second_output_acc'])
        plt.plot(self.result_log.history['val_second_output_acc'])
        plt.title('second output acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig.savefig(self.config["outputDir"]+"/second_output_acc_train_val.png", dpi=fig.dpi)
        
        # Plot discriminator distribution
        bins = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title('')
        ax.set_ylabel('Norm Events')
        ax.set_xlabel('Discriminator')
        plt.hist(y_Train_Sg, bins, color='xkcd:red', alpha=0.9, histtype='step', lw=2, label='Sg Train', density=True, log=True, weights=self.trainSg["Weight"])
        plt.hist(y_Val_Sg, bins, color='xkcd:green', alpha=0.9, histtype='step', lw=2, label='Sg Val', density=True, log=True, weights=valSg["Weight"])
        plt.hist(y_Train_Bg, bins, color='xkcd:blue', alpha=0.9, histtype='step', lw=2, label='Bg Train', density=True, log=True, weights=self.trainBg["Weight"])
        plt.hist(y_Val_Bg, bins, color='xkcd:magenta', alpha=0.9, histtype='step', lw=2, label='Bg Val', density=True, log=True, weights=valBg["Weight"])
        ax.legend(loc='best', frameon=False)
        fig.savefig(self.config["outputDir"]+"/discriminator.png", dpi=fig.dpi)
        
        samples = {"Bg": [self.trainBg, y_Train_Bg, self.trainBg["Weight"]], "Sg": [self.trainSg, y_Train_Sg, self.trainSg["Weight"]]}
        for sample in samples:
            trainSample = samples[sample][0]
            y_train_Sp = samples[sample][1]
            weights = samples[sample][2] 
            bins = np.linspace(0, 1, 100)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_title('')
            ax.set_ylabel('Norm Events')
            ax.set_xlabel('Discriminator')
            for key in sorted(trainSample.keys()):
                if key.find("mask") != -1:
                    yt = y_train_Sp[trainSample[key]]                
                    wt = weights[trainSample[key]]
                    if yt.size != 0 and wt.size != 0:
                        plt.hist(yt, bins, alpha=0.9, histtype='step', lw=2, label=sample+" Train "+key, density=True, log=True, weights=wt)
            plt.legend(loc='best')
            fig.savefig(self.config["outputDir"]+"/discriminator_nJet_"+sample+".png", dpi=fig.dpi)
        
        # Plot validation roc curve
        fpr_Val, tpr_Val, thresholds_Val = roc_curve(valData["labels"][:,0], y_Val, sample_weight=valData["Weight"][:,0])
        fpr_Train, tpr_Train, thresholds_Train = roc_curve(self.trainData["labels"][:,0], y_Train, sample_weight=self.trainData["Weight"][:,0])
        fpr_OTrain, tpr_OTrain, thresholds_OTrain = roc_curve(trainOData["labels"][:,0], y_OTrain, sample_weight=trainOData["Weight"][:,0])
        auc_Val = auc(fpr_Val, tpr_Val)
        auc_Train = auc(fpr_Train, tpr_Train)
        auc_OTrain = auc(fpr_OTrain, tpr_OTrain)
        
        fig = plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_Val, tpr_Val, color='xkcd:black', label='Val (area = {:.3f})'.format(auc_Val))
        plt.plot(fpr_Train, tpr_Train, color='xkcd:red', label='Train (area = {:.3f})'.format(auc_Train))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/roc_plot.png", dpi=fig.dpi)
        
        fig = plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_OTrain, tpr_OTrain, color='xkcd:black', label="Train "+self.config["otherttbarMC"][0]+" (area = {:.3f})".format(auc_OTrain))
        plt.plot(fpr_Train, tpr_Train, color='xkcd:red', label="Train "+self.config["ttbarMC"][0]+" (area = {:.3f})".format(auc_Train))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/roc_plot_TT_TTJets.png", dpi=fig.dpi)
        
        fig = plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        njetPerformance = []
        for key in sorted(self.trainData.keys()):
            if key.find("mask") != -1:
                labels = self.trainData["labels"][self.trainData[key]]
                weights = self.trainData["Weight"][self.trainData[key]][:,0]
                y = y_Train[self.trainData[key]]
                if len(y)==0:
                    continue
                fpr_Train, tpr_Train, thresholds_Train = roc_curve(labels[:,0], y, sample_weight=weights)
                auc_Train = auc(fpr_Train, tpr_Train)    
                njetPerformance.append(auc_Train)
                plt.plot(fpr_Train, tpr_Train, label="Train "+key+" (area = {:.3f})".format(auc_Train))
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/roc_plot_"+self.config["ttbarMC"][0]+"_nJet.png", dpi=fig.dpi)    
        
        fig = plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(self.config["otherttbarMC"][0]+" ROC curve")
        njetPerformance = []
        for key in sorted(trainOData.keys()):
            if key.find("mask") != -1:
                labels = trainOData["labels"][trainOData[key]]
                weights = trainOData["Weight"][trainOData[key]][:,0]
                y = y_OTrain[trainOData[key]]
                if len(y)==0:
                    continue
                fpr_Train, tpr_Train, thresholds_Train = roc_curve(labels[:,0], y, sample_weight=weights)
                auc_Train = auc(fpr_Train, tpr_Train)    
                njetPerformance.append(auc_Train)
                plt.plot(fpr_Train, tpr_Train, label="Train "+key+" (area = {:.3f})".format(auc_Train))
        plt.legend(loc='best')
        fig.savefig(self.config["outputDir"]+"/roc_plot_"+self.config["otherttbarMC"][0]+"_nJet.png", dpi=fig.dpi)    
        
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
        
        # Make njet distribution for 4 different bins
        nMVABins = 4
        inds = y_Train_Bg.argsort()
        sortednJet = self.trainBg["nJet"][:,0][inds[::-1]]
        sorted_y = y_Train_Bg[inds[::-1]]
        nJetDeepESMBins = np.array_split(sortednJet, nMVABins)
        sorted_y_split = np.array_split(sorted_y, nMVABins)
        index=0
        fig = plt.figure()
        bins = []
        for a in nJetDeepESMBins:
            print "DeepESM bin ", len(nJetDeepESMBins) - index, ": ", " NEvents: ", len(a)," bin cuts: ", sorted_y_split[index][0], " ", sorted_y_split[index][-1]
            plt.hist(a, bins=numbin, range=(binxl, binxh), histtype='step', density=True, log=True, label='Bin {}'.format(len(nJetDeepESMBins) - index))
            bins.append([str(sorted_y_split[index][0]), str(sorted_y_split[index][-1])])
            index += 1
        plt.hist(sortednJet, bins=numbin, range=(binxl, binxh), histtype='step', density=True, log=True, label='Total')
        plt.legend(loc='upper right')
        fig.savefig(self.config["outputDir"]+"/nJet_log.png", dpi=fig.dpi)
        
        index=0
        MVABinNJetShapeContent = []
        fig = plt.figure()
        for a in nJetDeepESMBins:
            n, _, _ = plt.hist(a, bins=numbin, range=(binxl, binxh), histtype='step', density=True, log=False, label='Bin {}'.format(len(nJetDeepESMBins) - index))
            MVABinNJetShapeContent.append(n)
            index += 1
        TotalMVAnJetShape, _, _ = plt.hist(sortednJet, bins=numbin, range=(binxl, binxh), histtype='step', density=True, log=False, label='Total')
        plt.legend(loc='upper right')
        fig.savefig(self.config["outputDir"]+"/nJet.png", dpi=fig.dpi)

        # Define metrics for the training
        self.metric["OverTrain"] = abs(auc_Val - auc_Train)
        self.metric["Performance"] = abs(1 - auc_Train)
        if not self.config["Mask"]:
            self.metric["nJetPerformance"] = 0.0
            for i in njetPerformance:
                self.metric["nJetPerformance"] += abs(i - self.metric["Performance"])
        if not self.config["Mask"]:
            self.metric["nJetShape"] = 0.0
            for l in MVABinNJetShapeContent:
                for i in range(len(l)):
                    self.metric["nJetShape"] += abs(l[i] - TotalMVAnJetShape[i])
                    
        # Save useful stuff
        self.trainData["y"] = y_Train
        np.save(self.config["outputDir"]+"/deepESMbin_dis_nJet.npy", self.trainData)
        self.config["bins"] = bins
        with open(self.config["outputDir"]+"/config.json",'w') as configFile:
            json.dump(self.config, configFile, indent=4, sort_keys=True)

        return self.config, self.metric
