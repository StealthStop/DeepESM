#!/bin/env python
import os
import json
import time
import shutil
import argparse
import datetime
import subprocess
import tracemalloc
import numpy as np
from glob import glob
import multiprocessing
import tensorflow as tf
import tensorflow.keras as K
from DataLoader import getFlatData

import sys, ast
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from Validation import Validation
from Correlation import Correlation as cor
from DataLoader import DataLoader
from Models import main_model
from MeanShiftTF import MeanShift
from CustomCallback import CustomCallback
from ShapUtils import waterfall2 as waterfall


def timeStamp():
    return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

class Train:
    def __init__(self, USER, inputs, outputDir, nJets, useJECs, debug, seed, replay, saveAndPrint, hyperconfig, doQuickVal=True, scaleJetPt=True, scaleLog=False, minStopMass=300, maxStopMass=1400, trainModel="RPV_SYY_SHH", evalMass=500, evalModel="RPV_SYY_SHH", evalYear = "2016preVFP", trainYear = "2016preVFP_2016postVFP_2017_2018", tree = "myMiniTree", maskNjet = [-1], procCats=False, massCats=False, njetsCats=False, debugModel=False, massScale=1000):

        print("%s [INFO]: Creating instance of Train."%(timeStamp()))

        proc = subprocess.Popen(["hostname", "-f"], stdout=subprocess.PIPE)
        host = proc.stdout.readlines()[0].rstrip().decode("UTF-8")
        atMN = ".umn." in host

        self.user                  = USER

        if not atMN:
            self.logdir                = "/storage/local/data1/gpuscratch/%s"%(self.user)
        else: 
            self.logdir                = "/scratch.global/%s"%(self.user)

        self.config                = {}
        self.config["host"]        = host
        self.config["seed"]        = seed
        self.config["debug"]       = debug
        self.config["minStopMass"] = int(minStopMass)
        self.config["maxStopMass"] = int(maxStopMass)
        self.config["useJECs"]     = useJECs
        self.config["scaleJetPt"]  = scaleJetPt
        self.config["scaleLog"]    = scaleLog
        self.config["nJets"]       = float(nJets)

        # Depending on final state, different pt requirements
        # and resultant objects are used
        ptCut = "pt30"

        # Labels for extracting relevant information from the
        # dataframes constructed from the inputs ROOT files
        self.config["massLabel"]       = "mass"
        self.config["domainLabel"]     = "NGoodJets_%s_double"%(ptCut)
        self.config["regressionLabel"] = "stop1_ptrank_mass"
        self.config["modelLabel"]      = "model"
        self.config["weightLabel"]     = "Weight"

        self.doQuickVal           = doQuickVal
        self.saveAndPrint         = saveAndPrint
        self.debugModel           = debugModel
        self.config["trainModel"] = trainModel
        self.config["evalMass"]   = evalMass
        self.config["massScale"]  = massScale
        self.config["evalModel"]  = evalModel
        self.config["trainYear"]  = trainYear
        self.config["evalYear"]   = evalYear
        self.config["tree"]       = tree

        self.config["procCats"]   = procCats
        self.config["massCats"]   = massCats
        self.config["njetsCats"]  = njetsCats

        if "0l" in tree:
            self.config["minNJetBin"] = 8
            self.config["maxNJetBin"] = 12
        elif "1l" in tree:
            self.config["minNJetBin"] = 7
            self.config["maxNJetBin"] = 11
        elif "2l" in tree:
            self.config["minNJetBin"] = 6
            self.config["maxNJetBin"] = 10
            

        self.config["verbose"]    = 1

        # Mask njet bins for 0l and 1l
        self.config["Mask_nJet"] = maskNjet
        if -1 in maskNjet:
            self.config["Mask"] = False
        else:
            self.config["Mask"] = True

        # The loader will hold all events used for training
        # The valLoader will hold 10% of events not used in training
        # The testLoader will hold remaning 10% of events not directly trained on
        # The evalLoader will hold events for any sample the user wants to validate the network with
        self.loader = None
        self.valLoader = None
        self.evalLoader = None
        self.testLoader = None

        TT_2016preVFP      = None; TT_2016postVFP      = None; TT_2017      = None; TT_2018      = None
        TT_2016preVFP_eval = None; TT_2016postVFP_eval = None; TT_2017_eval = None; TT_2018_eval = None

        Signal_2016preVFP      = []; Signal_2016postVFP      = []; Signal_2017      = []; Signal_2018      = []
        Signal_2016preVFP_eval = []; Signal_2016postVFP_eval = []; Signal_2017_eval = []; Signal_2018_eval = []

        self.config["evalBkg"] = None

        channel = "ToSemiLep"
        if "0l" in tree:
            channel = "ToHad"

        if "2l" in tree:
            channel="To2L"
        ################### Samples to train on #####################
        extra = "_[TV]"

        TT_2016preVFP  = ["2016preVFP_TT%s*"%(channel)]
        TT_2016postVFP = ["2016postVFP_TT%s*"%(channel)]
        TT_2017        = ["2017_TT%s*"%(channel)]
        TT_2018        = ["2018_TT%s*"%(channel)]

        ################### Samples to validate on #####################
        if   "vmad" in hyperconfig["atag"]:
            TT_2016preVFP_eval  = ["2016preVFP_TT%sJets%s*"%(channel,extra)]
            TT_2016postVFP_eval = ["2016postVFP_TT%sJets%s*"%(channel,extra)]
            TT_2017_eval        = ["2017_TT%sJets%s*"%(channel,extra)]
            TT_2018_eval        = ["2018_TT%sJets%s*"%(channel,extra)]
            self.config["evalBkg"] = 1

        elif "verd" in hyperconfig["atag"]:
            TT_2016preVFP_eval  = ["2016preVFP_TT%s*erdON%s*"%(channel,extra)]
            TT_2016postVFP_eval = ["2016postVFP_TT%s*erdON%s*"%(channel,extra)]
            TT_2017_eval        = ["2017_TT%s*erdON%s*"%(channel,extra)]
            TT_2018_eval        = ["2018_TT%s*erdON%s*"%(channel,extra)]
            self.config["evalBkg"] = 2

        elif "vhdampu" in hyperconfig["atag"]:
            TT_2016preVFP_eval  = ["2016preVFP_TT%s*hdampUP%s*"%(channel,extra)]
            TT_2016postVFP_eval = ["2016postVFP_TT%s*hdampUP%s*"%(channel,extra)]
            TT_2017_eval        = ["2017_TT%s*hdampUP%s*"%(channel,extra)]
            TT_2018_eval        = ["2018_TT%s*hdampUP%s*"%(channel,extra)]
            self.config["evalBkg"] = 3

        elif "vhdampd" in hyperconfig["atag"]:
            TT_2016preVFP_eval  = ["2016preVFP_TT%s*hdampDOWN%s*"%(channel,extra)]
            TT_2016postVFP_eval = ["2016postVFP_TT%s*hdampDOWN%s*"%(channel,extra)]
            TT_2017_eval        = ["2017_TT%s*hdampDOWN%s*"%(channel,extra)]
            TT_2018_eval        = ["2018_TT%s*hdampDOWN%s*"%(channel,extra)]
            self.config["evalBkg"] = 4

        elif "vueu" in hyperconfig["atag"]:
            TT_2016preVFP_eval  = ["2016preVFP_TT%s*TuneCP5up%s*"%(channel,extra)]
            TT_2016postVFP_eval = ["2016postVFP_TT%s*TuneCP5up%s*"%(channel,extra)]
            TT_2017_eval        = ["2017_TT%s*TuneCP5up%s*"%(channel,extra)]
            TT_2018_eval        = ["2018_TT%s*TuneCP5up%s*"%(channel,extra)]
            self.config["evalBkg"] = 5

        elif "vued" in hyperconfig["atag"]:
            TT_2016preVFP_eval  = ["2016preVFP_TT%s*TuneCP5down%s*"%(channel,extra)]
            TT_2016postVFP_eval = ["2016postVFP_TT%s*TuneCP5down%s*"%(channel,extra)]
            TT_2017_eval        = ["2017_TT%s*TuneCP5down%s*"%(channel,extra)]
            TT_2018_eval        = ["2018_TT%s*TuneCP5down%s*"%(channel,extra)]
            self.config["evalBkg"] = 6
        else:
            TT_2016preVFP_eval  = ["2016preVFP_TT%s*[cu]%s[!u]*"%(channel,extra)]
            TT_2016postVFP_eval = ["2016postVFP_TT%s*[cu]%s[!u]*"%(channel,extra)]
            TT_2017_eval        = ["2017_TT%s*[cu]%s[!u]*"%(channel,extra)]
            TT_2018_eval        = ["2018_TT%s*[cu]%s[!u]*"%(channel,extra)]
            self.config["evalBkg"] = 0

        if "vjcu" in hyperconfig["atag"]:
            self.config["evalBkg"] += 10
        elif "vjcd" in hyperconfig["atag"]:
            self.config["evalBkg"] += 20
        elif "vjru" in hyperconfig["atag"]:
            self.config["evalBkg"] += 30
        elif "vjrd" in hyperconfig["atag"]:
            self.config["evalBkg"] += 40

        # Add user-requested validation mass point to a pre-defined list
        for model in evalModel.split("_"):
            evalMasses = {"350", "550", "850", "1150"} 
            evalMasses.add(evalMass)
            for mass in evalMasses:
                Signal_2016preVFP_eval.append("2016preVFP*%s*mStop-%s"%(model,mass))
                Signal_2016postVFP_eval.append("2016postVFP*%s*mStop-%s"%(model,mass))
                Signal_2017_eval.append("2017*%s*mStop-%s"%(model,mass))
                Signal_2018_eval.append("2018*%s*mStop-%s"%(model,mass))
        

        for model in trainModel.split("_"):
            Signal_2016preVFP  += list("2016preVFP*%s*mStop-"%(model)+str(m) for m in range(self.config["minStopMass"],self.config["maxStopMass"]+50,50))
            Signal_2016postVFP += list("2016postVFP*%s*mStop-"%(model)+str(m) for m in range(self.config["minStopMass"],self.config["maxStopMass"]+50,50))
            Signal_2017        += list("2017*%s*mStop-"%(model)+str(m) for m in range(self.config["minStopMass"],self.config["maxStopMass"]+50,50))
            Signal_2018        += list("2018*%s*mStop-"%(model)+str(m) for m in range(self.config["minStopMass"],self.config["maxStopMass"]+50,50))

        TT = []; TTeval = []; Signal = []; SignalEval = []; self.config["lumi"] = 0
        if "2016preVFP" in self.config["trainYear"] or "Run2" in self.config["trainYear"] or "2016All" in self.config["trainYear"]: 
            TT                  += TT_2016preVFP
            Signal              += Signal_2016preVFP
            self.config["lumi"] += 19520 
        if "2016postVFP" in self.config["trainYear"] or "Run2" in self.config["trainYear"] or "2016All" in self.config["trainYear"]: 
            TT                  += TT_2016postVFP
            Signal              += Signal_2016postVFP
            self.config["lumi"] += 16810
        if "2017" in self.config["trainYear"] or "Run2" in self.config["trainYear"]:
            TT                  += TT_2017
            Signal              += Signal_2017
            self.config["lumi"] += 41500
        if "2018" in self.config["trainYear"] or "Run2" in self.config["trainYear"]:
            TT                  += TT_2018
            Signal              += Signal_2018
            self.config["lumi"] += 59800

        if "2016preVFP" in self.config["evalYear"] or "Run2" in self.config["evalYear"] or "2016All" in self.config["evalYear"]: 
            TTeval              += TT_2016preVFP_eval
            SignalEval          += Signal_2016preVFP_eval
        if "2016postVFP" in self.config["evalYear"] or "Run2" in self.config["evalYear"] or "2016All" in self.config["evalYear"]: 
            TTeval              += TT_2016postVFP_eval
            SignalEval          += Signal_2016postVFP_eval
        if "2017" in self.config["evalYear"] or "Run2" in self.config["evalYear"]:
            TTeval              += TT_2017_eval
            SignalEval          += Signal_2017_eval
        if "2018" in self.config["evalYear"] or "Run2" in self.config["evalYear"]:
            TTeval              += TT_2018_eval
            SignalEval          += Signal_2018_eval

        self.config["bkgd"]          = ("TT", TT)
        self.config["bkgdEval"]      = ("TTeval", TTeval)
        self.config["signal"]        = Signal
        self.config["signalEval"]    = SignalEval
        self.config["bkgdShift"]     = ("TT", TT)
        self.config["dataSet"]       = inputs
        print(inputs)
        self.config["doBgWeight"]    = True
        self.config["doSgWeight"]    = True
        self.config["class_weight"]  = None
        self.config["sample_weight"] = None
        self.config["metrics"]       = ['accuracy']
        print("%s [INFO]: Using "%(timeStamp())+self.config["dataSet"]+" data set")
        print("%s [INFO]: Training on signal: "%(timeStamp()),       self.config["signal"])
        print("%s [INFO]: Training on background: "%(timeStamp()),   self.config["bkgd"][1])
        print("%s [INFO]: Validating on signal: "%(timeStamp()),     self.config["signalEval"])
        print("%s [INFO]: Validating on background: "%(timeStamp()), self.config["bkgdEval"][1])

        # Define ouputDir based on input config
        self.makeOutputDir(hyperconfig, outputDir, replay)
        self.config.update(hyperconfig)

        if not os.path.exists(self.logdir): os.makedirs(self.logdir)

    # Define loss functions
    def loss_mass_reg(self, c):
        def regLoss(y_true, y_pred):
            return c * K.losses.mean_squared_error(y_true, y_pred)
        return regLoss

    def loss_disco(self, c, current_epoch, start_epoch):
        def discoLoss(y_mask, y_pred):
            case = tf.greater(current_epoch, start_epoch)

            val_1 = tf.reshape(y_pred[:,  :1], [-1])
            val_2 = tf.reshape(y_pred[:, 1:2], [-1])
            normedweight = tf.ones_like(val_1)

            #Mask all signal events
            mask_sg = tf.reshape(tf.abs(1 - y_mask[:,  :1]), [-1])
            val_1_bg = tf.boolean_mask(val_1, mask_sg)
            val_2_bg = tf.boolean_mask(val_2, mask_sg)

            normedweight_bg = tf.boolean_mask(normedweight, mask_sg)

            #rdc = cor.rdc(val_1_bg, val_2_bg)
            dcorr = cor.distance_corr(val_1, val_2, normedweight, 1)
            #dcorr = cor.distance_corr(val_1_bg, val_2_bg, normedweight_bg, 1)

            return c * tf.cast(case, "float32") * (dcorr)
        return discoLoss
    
    def loss_closure(self, c, g, nBinEdge, current_epoch, start_epoch):
        def closureLoss(y_mask, y_pred):

            case = tf.greater(current_epoch, start_epoch)

            val_1 = tf.reshape(y_pred[:,  :1], [-1])
            val_2 = tf.reshape(y_pred[:, 1:2], [-1])
            normedweight = tf.ones_like(val_1)

            #Mask all signal events
            mask_sg = tf.reshape(tf.abs(1 - y_mask[:,  :1]), [-1])
            temp1 = tf.boolean_mask(val_1, mask_sg)
            temp2 = tf.boolean_mask(val_2, mask_sg)

            #temptile1 = tf.reshape(tf.tile(temp1, [nBinEdge]), shape=(nBinEdge,-1))
            #temptile2 = tf.reshape(tf.tile(temp2, [nBinEdge]), shape=(nBinEdge,-1))

            d1 = g.uniform(shape=(), minval=0.0, maxval=1.0)
            d2 = g.uniform(shape=(), minval=0.0, maxval=1.0)
            #d1 = 0.5
            #d2 = 0.5

            #d1_peak = tf.reshape(tf.range(0.2, 1.0, 0.2), shape=(4, 1))
            #d2_peak = tf.reshape(tf.range(0.2, 1.0, 0.2), shape=(4, 1))
            
            #histy_bins = tf.histogram_fixed_width_bins(val_1, (0.0, 1.0), nbins=5, dtype=tf.dtype.float32)
            
            #H = tf.map_fn(lambda i: tf.histogram_fixed_width(val_2[histy_bins == i], (0.0, 1.0), nbins=5), tf.range(5), dtype=tf.dtype.float32)

            #tot = tf.reduce_sum(H)

            #H = H / tot

            #tf.print(H)
            #tf.print(tf.reduce_sum(H))


            dval1 = d1 / 2.0
            dval2 = d2 / 2.0

            nbTot = 2.0*tf.reduce_sum(tf.sigmoid(0.0*temp1))

            # Calculate non-closure in full ABCD region defined by (d1, d2) edges
            nbA = tf.reduce_sum(tf.sigmoid(1e2*(temp1-d1))*tf.sigmoid(1e2*(temp2-d2)))
            nbB = tf.reduce_sum(tf.sigmoid(1e2*(d1 - temp1))*tf.sigmoid(1e2*(temp2-d2)))
            nbC = tf.reduce_sum(tf.sigmoid(1e2*(temp1 - d1))*tf.sigmoid(1e2*(d2 - temp2)))
            nbD = tf.reduce_sum(tf.sigmoid(1e2*(d1 - temp1))*tf.sigmoid(1e2*(d2 - temp2)))
           
            nbA = nbA + 0.01
            nbB = nbB + 0.01
            nbC = nbC + 0.01
            nbD = nbD + 0.01

            ''' 
            min_N = 1 #tf.reduce_mean(nbTot) * 0.001

            nbA = nbA[nbD > min_N]
            nbB = nbB[nbD > min_N]
            nbC = nbC[nbD > min_N]
            nbD = nbD[nbD > min_N]
            nbTot = nbTot[nbD > min_N]

            nbA = nbA[nbC > min_N]
            nbB = nbB[nbC > min_N]
            nbC = nbC[nbC > min_N]
            nbD = nbD[nbC > min_N]
            nbTot = nbTot[nbC > min_N]

            nbA = nbA[nbB > min_N]
            nbB = nbB[nbB > min_N]
            nbC = nbC[nbB > min_N]
            nbD = nbD[nbB > min_N]
            nbTot = nbTot[nbB > min_N]

            nbA = nbA[nbA > min_N]
            nbB = nbB[nbA > min_N]
            nbC = nbC[nbA > min_N]
            nbD = nbD[nbA > min_N]
            nbTot = nbTot[nbA > min_N]
            '''
            #nbApred = nbB*nbC/nbD
            
            #fracs = abs(nbA - nbApred) / (nbA)
             
            # New pull based loss function
            '''
            var = nbD ** 2 * nbA + nbC **2 * nbB + nbB ** 2 * nbC + nbA ** 2 * nbD
            var = nbD ** 2 * cov(nbA, nbA) + nbC ** 2 * cov(nbB, nbB) + nbB ** 2 * cov(nbC, nbC) + nbA ** 2 *cov(nbD, nbD) 
            covar = - 2 * nbC * nbD * cov(nbA, nbB) - 2 * nbB * nbD * cov(nbA, nbC) + 2 * nbA * nbD * cov(nbA, nbD) + 2 * nbB * nbC * cov(nbB, nbC) - 2 * nbA * nbC * cov(nbB, nbD) - 2 * nbA * nbB * cov(nbC, nbD)
            dl = tf.sqrt(var)

            fracs = (nbA * nbD - nbB * nbC) / dl
            '''

            fracs = ((nbA * nbD - nbB * nbC)/(nbA * nbD + nbB * nbC))**2

            frac = tf.reduce_sum(fracs)
            #frac = K.losses.mean_squared_error(tf.zeros_like(fracs), fracs)
            #frac = tf.reduce_mean(fracs)

            return c * tf.cast(case, "float32") * (frac)
        return closureLoss 
            
    def loss_disc(self, c, current_epoch, start_epoch):
        def loss_model_disc(y_true, y_pred):         
            case = tf.greater(current_epoch, start_epoch)

            # Decat truth and predicted
            val_1_disco_true = y_true[:, :1]
            val_2_disco_true = y_true[:, 1:]
            #val_1_disco_true = tf.reshape(y_true[:, :4], [-1])
            #val_2_disco_true = tf.reshape(y_true[:, 4:], [-1])
            #val_disco_true = tf.reshape(y_true[:, :4], [-1])

            val_1_disco_pred = y_pred[:, :1]
            val_2_disco_pred = y_pred[:, 1:]
            #val_1_disco_pred = tf.reshape(y_pred[:, :4], [-1])
            #val_2_disco_pred = tf.reshape(y_pred[:, 4:], [-1])
            #val_disco_pred = tf.reshape(y_pred[:, :4], [-1])

            cce = K.losses.BinaryCrossentropy()

            # Calculate loss function
            val_1_disco_loss = cce(val_1_disco_true, val_1_disco_pred)
            val_2_disco_loss = cce(val_2_disco_true, val_2_disco_pred)

            #alpha = 0.90
            #gamma = 2.0

            #alpha1 = tf.math.abs(val_1_disco_true - alpha)
            #alpha2 = tf.math.abs(val_2_disco_true - alpha)

            #pt1 = tf.math.abs(1 - val_1_disco_true - val_1_disco_pred)
            #pt2 = tf.math.abs(1 - val_2_disco_true - val_2_disco_pred)

            #FL1 = tf.reduce_sum(-alpha1 * (1 - pt1) ** gamma * tf.math.log(pt1))
            #FL2 = tf.reduce_sum(-alpha2 * (1 - pt2) ** gamma * tf.math.log(pt2))

            #return c * tf.cast(case, "float32") * (FL1 + FL2)
            return c * tf.cast(case, "float32") * (val_1_disco_loss + val_2_disco_loss)
        return loss_model_disc

    def make_model(self, scales, means, regShape, discoShape, inputShape, bias):

        print("%s [INFO]: Constructing model."%(timeStamp()))

        model = main_model(self.config, scales, means, regShape, discoShape, inputShape, bias)
        if self.debugModel:
            K.utils.plot_model(model, show_shapes=True, to_file="model.png")
            model.summary()
        g = tf.random.Generator.from_seed(self.config["seed"]) 
        current_epoch = K.backend.variable(1.)

        opt = K.optimizers.Adam(learning_rate=self.config["lr"])

        self.cb = CustomCallback(current_epoch)
        model.compile(loss={'disc': self.loss_disc(c=self.config["disc_lambda"], current_epoch=self.cb.current_epoch, start_epoch=self.config["disc_start"]), 'disco': self.loss_disco(c=self.config["bkg_disco_lambda"], current_epoch=self.cb.current_epoch, start_epoch=self.config["disco_start"]), 'closure': self.loss_closure(c=self.config["abcd_close_lambda"], g=g, nBinEdge=1, current_epoch=self.cb.current_epoch, start_epoch=self.config["abcd_start"]), 'mass_reg': self.loss_mass_reg(c=self.config["mass_reg_lambda"])}, optimizer=opt, metrics={'disc': [K.metrics.Precision(), K.metrics.Recall()], 'mass_reg': K.metrics.MeanSquaredError()})
        #model.compile(loss=[self.loss_disc(c=self.config["disc_lambda"]), self.loss_disco(c=self.config["bkg_disco_lambda"], current_epoch=1), self.loss_mass_reg(c=self.config["mass_reg_lambda"])], optimizer="adam")#, metrics=self.config["metrics"])
        return model, self.cb

    def getSamplesToRun(self, names):
        s = glob(names)
        if len(s) == 0:
            raise Exception("No files find that correspond to: "+names)
        return s

    def get_callbacks(self):
        #tbCallBack = K.callbacks.TensorBoard(log_dir=self.logdir+"/log_graph",            histogram_freq=0,   write_graph=True,               write_images=True)
        log_model  = K.callbacks.ModelCheckpoint(self.config["outputDir"]+"/BestNN.hdf5", monitor='val_loss', verbose=self.config["verbose"], save_best_only=True)
        earlyStop  = K.callbacks.EarlyStopping(monitor="disc_loss",                        min_delta=0,        patience=10, verbose=0,          mode="auto", baseline=None)

        callbacks  = []
        if self.config["verbose"] == 1: 
            #callbacks = [log_model, tbCallBack, earlyStop]
            #callbacks = [log_model, tbCallBack]
            #callbacks = [log_model, earlyStop]
            #callbacks = [tbCallBack]
            callbacks = []
        return callbacks

    def gpu_allow_mem_grow(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print("%s [ERROR]: gpu_mem_grow failed: "%(timeStamp()),e)

    def save_model_pb(self, model):
        #https://github.com/leimao/Frozen_Graph_TensorFlow/tree/master/TensorFlow_v2

        # Save model as hdf5 format
        model.save(self.config["outputDir"]+"/keras_model")

        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: model(x))
        full_model= full_model.get_concrete_function(x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        self.config["input_output"] = list(x.name.split(':')[0] for x in frozen_func.inputs + frozen_func.outputs)
        
        # Save frozen graph from frozen ConcreteFunction to hard drive
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=self.config["outputDir"], name="keras_frozen.pb", as_text=False)

    def plot_model(self, model):
        try:
            K.utils.plot_model(model, to_file=self.config["outputDir"]+"/model.png", show_shapes=True)
        except AttributeError as e:
            print("%s [ERROR]: plot_model failed: "%(timeStamp()),e)

    def makeOutputDir(self,d,outputDirStub,replay):
        outputDir = "%s/Output/%s__"%(outputDirStub, d["atag"])
        nodesStr = "nodes_"
        learningStr = "lr_"
        layersStr = "layers_"
        hyperStr = "lambda_"
        otherStr = ""
        for key in sorted(d.keys()):

            if "atag" in key: continue

            trimStr = "".join(key.split("_")[:-1]) + str(d[key]) + "_"
            if "_nodes" in key:
                nodesStr += trimStr
            elif "_layers" in key:
                layersStr += trimStr
            elif "_lambda" in key:
                hyperStr += trimStr
            elif "_lr" in key:
                learningStr += trimStr
            else:
                otherStr += key + str(d[key]) + "_"
        
        self.config["outputDir"] = outputDir + hyperStr + "_" + nodesStr + "_" + layersStr + "_" + learningStr + "_" + otherStr
        self.config["outputDir"] = self.config["outputDir"][:-1]
        if os.path.exists(self.config["outputDir"]) and not replay:
            print("%s [INFO]: Removing old training files: "%(timeStamp()), self.config["outputDir"])
            shutil.rmtree(self.config["outputDir"])

        if not replay: os.makedirs(self.config["outputDir"]+"/log_graph")    


    def defineVars(self):

        print("%s [INFO]: Defining input variables."%(timeStamp()))

        htVec           = ["HT_trigger_pt30"]
        nJetsVec        = ["NGoodJets_pt30_double"]
        fwmVec          = ["fwm2_top6",    "fwm3_top6",    "fwm4_top6",   "fwm5_top6"]
        jmtVec          = ["jmt_ev0_top6", "jmt_ev1_top6", "jmt_ev2_top6"]
        j4Vec           = ["Jet_pt_", "Jet_eta_", "Jet_phi_"]#,"Jet_E_"]
        jFlavVec        = ["Jet_flavb_", "Jet_flavuds_", "Jet_flavq_", "Jet_flavg_", "Jet_flavc_"]
        jComb6Vec        = ["combined6thToLastJet_pt_cm", "combined6thToLastJet_eta_cm", "combined6thToLastJet_m_cm", "combined6thToLastJet_phi_cm"]
        jComb7Vec        = ["combined7thToLastJet_pt_cm", "combined7thToLastJet_eta_cm", "combined7thToLastJet_m_cm", "combined7thToLastJet_phi_cm"]
        jComb8Vec        = ["combined8thToLastJet_pt_cm", "combined8thToLastJet_eta_cm", "combined8thToLastJet_m_cm", "combined8thToLastJet_phi_cm"]
        jqgDiscVec      = ["Jet_ptD_", "Jet_axismajor_", "Jet_axisminor_"]
        lvMETVec        = ["lvMET_cm_pt", "lvMET_cm_eta", "lvMET_cm_phi", "lvMET_cm_m",]
        l1Vec            = ["GoodLeptons_pt_1", "GoodLeptons_eta_1", "GoodLeptons_phi_1", "GoodLeptons_m_1",]
        l2Vec            = ["GoodLeptons_pt_1", "GoodLeptons_eta_1", "GoodLeptons_phi_1", "GoodLeptons_m_1", "GoodLeptons_pt_2", "GoodLeptons_eta_2", "GoodLeptons_phi_2", "GoodLeptons_m_2",]
        stop1OldSeed    = ["Stop1_mass_cm_OldSeed", "Stop1_pt_cm_OldSeed", "Stop1_phi_cm_OldSeed", "Stop1_eta_cm_OldSeed"]
        stop2OldSeed    = ["Stop2_mass_cm_OldSeed", "Stop2_pt_cm_OldSeed", "Stop2_phi_cm_OldSeed", "Stop2_eta_cm_OldSeed"]
        stop1TopSeed    = ["Stop1_mass_cm_TopSeed", "Stop1_pt_cm_TopSeed", "Stop1_phi_cm_TopSeed", "Stop1_eta_cm_TopSeed"]
        stop2TopSeed    = ["Stop2_mass_cm_TopSeed", "Stop2_pt_cm_TopSeed", "Stop2_phi_cm_TopSeed", "Stop2_eta_cm_TopSeed"]
        drOldSeed       = ["dR_Stop1Stop2_cm_OldSeed"]
        drTopSeed       = ["dR_Stop1Stop2_cm_TopSeed"]
        dphiOldSeed     = ["dPhi_Stop1Stop2_cm_OldSeed"]
        dphiTopSeed     = ["dPhi_Stop1Stop2_cm_TopSeed"]
        mt2OldSeed      = ["MT2_cm_OldSeed"]
        mt2TopSeed      = ["MT2_cm_TopSeed"]
        stop1SPtOldSeed = ["Stop1_scalarPt_cm_OldSeed"]
        stop2SPtOldSeed = ["Stop2_scalarPt_cm_OldSeed"]
        stop1SPtTopSeed = ["Stop1_scalarPt_cm_TopSeed"]
        stop2SPtTopSeed = ["Stop2_scalarPt_cm_TopSeed"]

        nJets = int(self.config["nJets"]); theVars = None

        if "0l" in self.config["tree"]:
            theVars = j4Vec + jComb8Vec + jFlavVec 
            jetNum = 7
        elif "1l" in self.config["tree"]:
            theVars = j4Vec + jComb7Vec + jFlavVec + l1Vec
            jetNum = 6
        elif "2l" in self.config["tree"]:
            theVars = j4Vec + jComb6Vec + jFlavVec + l2Vec
            jetNum = 5
            
        if not self.config["scaleJetPt"]:
            theVars += htVec

        theVars += fwmVec
        theVars += jmtVec

        if "0l" in self.config["tree"]:
            theVars += stop1TopSeed 
            theVars += stop2TopSeed
        else:
            theVars += stop1OldSeed 
            theVars += stop2OldSeed

        newVars = []; auxVars = []
        for var in theVars:

            if "Jet_" in var[0:4]:
                start = 0

                if "phi" in var:
                    start = 1
        
                for nJet in range(start,jetNum):
                    newVars.append(var + str(nJet+1))

            else: newVars.append(var)

        self.config["trainVars"] = newVars

        print("Number of variables: ", len(self.config["trainVars"]), self.config["trainVars"])
        
        # We load auxiliary variables that are not to be used as direct inputs
        # DataLoader handles these separately
        auxVars.append(self.config["weightLabel"])
        auxVars.append(self.config["modelLabel"])
        auxVars.append(self.config["regressionLabel"])
        auxVars.append(self.config["massLabel"])
        auxVars.append(self.config["domainLabel"])
        auxVars.append("HT_trigger_pt30")

        self.config["auxVars"] = auxVars

    def importData(self):
        # Import data
        temp = "*"

        print("%s [INFO]: Preparing input data sets."%(timeStamp()))
        #Get Data set used in training and validation
        sgTrainSet = sum( (glob(self.config["dataSet"]+"MyAnalysis_"+mass+temp+"Train.root") for mass in self.config["signal"]) , [])
        bgTrainSet = sum( (glob(self.config["dataSet"]+"MyAnalysis_"+bkgd+temp+"Train.root") for bkgd in self.config["bkgd"][1]), [])

        sgTestSet  = sum( (glob(self.config["dataSet"]+"MyAnalysis_"+mass+temp+"Test.root")  for mass in self.config["signal"]) , [])
        bgTestSet  = sum( (glob(self.config["dataSet"]+"MyAnalysis_"+bkgd+temp+"Test.root")  for bkgd in self.config["bkgd"][1]), [])

        sgEvalSet   = sum( (glob(self.config["dataSet"]+"MyAnalysis_"+mass+temp+".root")      for mass in self.config["signalEval"]) , [])
        bgEvalSet   = sum( (glob(self.config["dataSet"]+"MyAnalysis_"+bkgd+temp+".root")      for bkgd in self.config["bkgdEval"][1]), [])

        sgValSet  = sum( (glob(self.config["dataSet"]+"MyAnalysis_"+mass+temp+"Val.root")   for mass in self.config["signal"]) , [])
        bgValSet  = sum( (glob(self.config["dataSet"]+"MyAnalysis_"+bkgd+temp+"Val.root")   for bkgd in self.config["bkgd"][1]), [])

        sgSet = sgTrainSet + sgTestSet + sgValSet
        bgSet = bgTrainSet + bgTestSet + bgValSet

        needeval = False
        for sample in sgEvalSet:
            if sample not in sgSet:
                needeval = True
                break
        for sample in bgEvalSet:
            if sample not in bgSet:
                needeval = True
                break

        if needeval: self.evalLoader = DataLoader(self.config, sgEvalSet, bgEvalSet)

        cfg_string = json.dumps(self.config)
       
        with open("config.json", "w") as f:
            f.write(cfg_string)
        f.close()
       
        self.loader     = DataLoader(self.config, sgTrainSet, bgTrainSet)
        self.valLoader  = DataLoader(self.config, sgValSet,   bgValSet)
        self.testLoader = DataLoader(self.config, sgTestSet,  bgTestSet)

    def train(self):
        # Define vars for training
        self.defineVars()

        # Get stuff from input ROOT files
        self.importData()

        self.config["nBkgTrainEvents"] = self.loader.getNumBkgEvents()
        self.config["nSigTrainEvents"] = self.loader.getNumSigEvents()

        scales = self.loader.getDataScales()
        means  = self.loader.getDataMeans()

        regShape, domainShape, discoShape, inputShape = self.loader.getShapes()

        initial_bias = np.log(self.loader.getNumSigEvents()/self.loader.getNumBkgEvents())

        # Make model
        print("%s [INFO]: Preparing the training model."%(timeStamp()))
        # Kelvin says no
        self.gpu_allow_mem_grow()
        g = tf.random.Generator.from_seed(self.config["seed"]) 
        model, self.cb     = self.make_model(scales, means, regShape, discoShape, inputShape, initial_bias)
        callbacks = self.get_callbacks()
        callbacks.append(self.cb)

        # Training model
        print("%s [INFO]: Training the model."%(timeStamp()))
        result_log = model.fit(self.loader, epochs=self.config["epochs"], callbacks=callbacks, validation_data=self.testLoader)

        if self.saveAndPrint:
            # Model Visualization
            print("%s [INFO]: Printing the model."%(timeStamp()))

            try:
                self.plot_model(model)
            except Exception as e:
                print("%s [WARNING]: Could not print model !"%(timeStamp()), e)

            # Save trainig model as a protocol buffers file
            print("%s [INFO]: Saving model as in protobuffer format."%(timeStamp()))

            try:
                self.save_model_pb(model)
            except Exception as e:
                print("%s [WARNING]: Could not save model pb !"%(timeStamp()), e)
      
        self.config['outputDir'] += "/" +  self.config['evalYear']
        if not os.path.isdir(self.config['outputDir']):        
            os.makedirs(self.config['outputDir'])
 
        #Plot results
        print("%s [INFO]: Running validation of model."%(timeStamp()))
        val = Validation(model, self.config, self.loader, self.valLoader, self.evalLoader, self.testLoader, result_log)

        #work in progress
        #SHAP result for first model prediction
        
        waterfall(model, getFlatData(self), 0)
        
        metric = val.makePlots(self.doQuickVal, self.config["evalMass"], self.config["evalModel"])
        del val
        
        #Clean up training
        del model
    
        return metric

    def replay(self):

        self.defineVars()

        current_epoch = K.backend.variable(1.)
        self.cb = CustomCallback(current_epoch)

        g = tf.random.Generator.from_seed(self.config["seed"]) 

        self.importData()

        model = K.models.load_model(self.config["outputDir"]+"/keras_model", custom_objects={'loss_model_disc': self.loss_disc(c=self.config["disc_lambda"], current_epoch=self.cb.current_epoch, start_epoch=self.config["disc_start"]), 'discoLoss': self.loss_disco(c=self.config["bkg_disco_lambda"], current_epoch=self.cb.current_epoch, start_epoch=self.config["disco_start"]), 'closureLoss': self.loss_closure(c=self.config["abcd_close_lambda"], g=g, nBinEdge=1, current_epoch=self.cb.current_epoch, start_epoch=self.config["abcd_start"]) })

        self.config['outputDir'] += "/" + self.config['evalYear']
        if not os.path.isdir(self.config['outputDir']):        
            os.makedirs(self.config['outputDir'])
 
        #trainData, trainSg, trainBg = get_data(sgTrainSet, bgTrainSet, self.config)
        val = Validation(model, self.config, self.loader, self.valLoader, self.evalLoader, self.testLoader)
        metric = val.makePlots(self.doQuickVal, self.config["evalMass"], self.config["evalModel"])
        del val

if __name__ == '__main__':

    tracemalloc.start()

    usage = "usage: %prog [options]"
    parser = argparse.ArgumentParser(usage)
    parser.add_argument("--quickVal",     dest="quickVal",     help="Do quick (partial) validation",  action="store_true", default=False                                  ) 
    parser.add_argument("--json",         dest="json",         help="JSON config file",               default="NULL"                                                      ) 
    parser.add_argument("--minMass",      dest="minMass",      help="Minimum stop mass to train on",  default=300                                                         )
    parser.add_argument("--maxMass",      dest="maxMass",      help="Maximum stop mass to train on",  default=1400                                                        ) 
    parser.add_argument("--evalMass",     dest="evalMass",     help="Stop mass to evaluate on",       default=500                                                         ) 
    parser.add_argument("--evalModel",    dest="evalModel",    help="Signal model to evaluate on",    default="RPV"                                                       ) 
    parser.add_argument("--evalYear",     dest="evalYear",     help="Year(s) to eval on",             type=str,            default="2016preVFP"                           ) 
    parser.add_argument("--trainModel",   dest="trainModel",   help="Signal model to train on",       type=str,            default="RPV"                                  ) 
    parser.add_argument("--replay",       dest="replay",       help="Replay saved model",             action="store_true", default=False                                  ) 
    parser.add_argument("--trainYear",    dest="trainYear",    help="Year(s) to train on",            type=str,            default="2016preVFP_2016postVFP_2017_2018"     ) 
    parser.add_argument("--inputs",       dest="inputs",       help="Path to input files",            type=str,            default="NN_inputs/"                           ) 
    parser.add_argument("--tree",         dest="tree",         help="TTree to load events from",      type=str,            default="myMiniTree"                           )
    parser.add_argument("--saveAndPrint", dest="saveAndPrint", help="Save pb and print model",        action="store_true", default=False                                  )
    parser.add_argument("--seed",         dest="seed",         help="Use specific seed for env",      type=int,            default=-1                                     )
    parser.add_argument("--nJets",        dest="nJets",        help="Minimum number of jets",         type=int,            default=7                                      )
    parser.add_argument("--debug",        dest="debug",        help="Debug with small set of events", action="store_true", default=False                                  )
    parser.add_argument("--debugModel",   dest="debugModel",   help="Debug model, no training done",  action="store_true", default=False                                  )
    parser.add_argument("--scaleJetPt",   dest="scaleJetPt",   help="Scale Jet pt by HT",             default=True,        action="store_true"                            )
    parser.add_argument("--scaleLog",     dest="scaleLog",     help="Scale variables with log",       default=False,       action="store_true"                            )
    parser.add_argument("--massScale",    dest="massScale",    help="Scaling for mass regression",    type=float,          default=1000                                   )
    parser.add_argument("--useJECs",      dest="useJECs",      help="Use JEC/JER variations",         action="store_true", default=False                                  )
    parser.add_argument("--maskNjet",     dest="maskNjet",     help="mask Njet bin(s) in training",   default=[-1], nargs="+", type=int                                   )
    parser.add_argument("--procCats",     dest="procCats",     help="Balance batches bkg/sig",        default=False,       action="store_true"                            )
    parser.add_argument("--massCats",     dest="massCats",     help="Balance batches among masses",   default=False,       action="store_true"                            )
    parser.add_argument("--njetsCats",    dest="njetsCats",    help="Balance batches among njets",    default=False,       action="store_true"                            )
    parser.add_argument("--outputDir",    dest="outputDir",    help="Output directory path",          type=str,            default="/home/nstrobbe/USER/Train/DeepESM")

    args = parser.parse_args()

    theDay  = datetime.date.today()
    theTime = datetime.datetime.now()
    
    d = theDay.strftime("%Y%m%d")
    t = theTime.strftime("%H%M%S")

    # Get seed from time, but allow user to reseed with their own number
    masterSeed     = int(time.time())
    if args.seed  != -1:
        masterSeed = args.seed

    # Seed the tensorflow here, seed numpy in datagetter
    tf.random.set_seed(masterSeed)

    # For reproduceability, try these resetting/clearing commands
    K.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    K.backend.set_floatx('float32')

    USER = os.getenv("USER")

    outputDir = args.outputDir.replace("USER", USER)

    replay = args.replay

    hyperconfig = {}
    if args.json != "NULL": 
        with open(str(args.json), "r") as f:
            hyperconfig = json.load(f)
    else: 
        hyperconfig = {"atag" : "Perfect", "disc_lambda": 5.0, "bkg_disco_lambda": 1000.0, "mass_reg_lambda": 0.0001, "abcd_close_lambda" : 2.0, "disc_nodes":300, "mass_reg_nodes":100, "disc_layers":1, "mass_reg_layers":1, "dropout":0.3, "batch":20000, "epochs":20, "other_lr" : 0.001, "disc_lr":0.001, "mass_reg_lr" : 1.0}

    t = Train(USER, args.inputs, outputDir, args.nJets, args.useJECs, args.debug, masterSeed, replay, args.saveAndPrint, hyperconfig, args.quickVal, args.scaleJetPt, args.scaleLog, minStopMass=args.minMass, maxStopMass=args.maxMass, trainModel=args.trainModel, evalMass=args.evalMass, evalModel=args.evalModel, evalYear=args.evalYear, trainYear=args.trainYear, tree=args.tree, maskNjet=args.maskNjet, procCats=args.procCats, massCats=args.massCats, njetsCats=args.njetsCats, debugModel=args.debugModel, massScale=args.massScale)

    if replay: t.replay()

    elif args.json != "NULL": 

        metric = t.train()

        metricName = args.json.replace(".json", "_metric.json")
        with open(str(metricName), 'w') as f:
            json.dump(metric, f)
    else:
        t.train()

    print("%s [INFO]: Maximum memory useage = %3.2f GB"%(timeStamp(), tracemalloc.get_traced_memory()[1]/1e9))
    print("%s [INFO]: End of trainNew.py"%(timeStamp()))
