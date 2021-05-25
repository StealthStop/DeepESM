#!/bin/env python
import os
import json
import time
import shutil
import argparse
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras as K

import sys, ast, os
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from Validation import Validation
from Correlation import Correlation as cor
from DataGetter import get_data,getSamplesToRun
from Models import main_model, model_reg, model_doubleDisco
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

class Train:
    def __init__(self, USER, debug, seed, replay, saveAndPrint, hyperconfig, doQuickVal=False, doReweight=False, minStopMass=300, maxStopMass=1400, trainModel="RPV_SYY_SHH", valMass=500, valModel="RPV_SYY_SHH", year = "2016_2017_2018", tree = "myMiniTree", maskNjet = [-1], bkgSampleFactor=1, sigSampleFactor=1):
        self.user                  = USER
        self.logdir                = "/storage/local/data1/gpuscratch/%s"%(self.user)
        self.config                = {}
        self.config["seed"]        = seed
        #self.config["case"]        = int(hyperconfig["case"])
        self.config["debug"]       = debug
        self.config["minStopMass"] = int(minStopMass)
        self.config["maxStopMass"] = int(maxStopMass)
        self.config["doReweight"]  = doReweight

        self.doQuickVal           = doQuickVal
        self.saveAndPrint         = saveAndPrint
        self.config["trainModel"] = trainModel
        self.config["valMass"]    = valMass
        self.config["valModel"]   = valModel
        self.config["year"]       = year
        self.config["tree"]       = tree

        if "0l" in self.config["tree"]:
            self.config["minNJetBin"] = 6
            self.config["maxNJetBin"] = 12
            self.config["verbose"]    = 1
        else:
            self.config["minNJetBin"] = 7
            self.config["maxNJetBin"] = 11
            self.config["verbose"]    = 1

        # mask njet bins for 0l and 1l
        self.config["Mask_nJet"] = maskNjet
        if -1 in maskNjet:
            self.config["Mask"] = False
        else:
            self.config["Mask"] = True

        self.config["bkgSampleFactor"] = bkgSampleFactor 
        self.config["sigSampleFactor"] = sigSampleFactor

        TT_2016     = None; TT_2017     = None; TT_2018     = None
        TT_2016_val = None; TT_2017_val = None; TT_2018_val = None

        Signal_2016     = []; Signal_2017     = []; Signal_2018     = []
        Signal_2016_val = []; Signal_2017_val = []; Signal_2018_val = []

        ################### Samples to train on #####################
        if "tpowmad" in hyperconfig["atag"]:
            # Train on POWHEG and MADGRAPH
            TT_2016 = ["2016_TT_?_", "2016_TT_??_", "2016_TT_???_", "2016_TTJets"]

        elif "twsysts" in hyperconfig["atag"]:
            # Train on POWHEG with systs variations
            # Using the X and XX numbered files gives the closest number events to the total 300 to 1400 RPV set
            if "tRPVSYY" in hyperconfig["atag"]:
                TT_2016 = [
                            "2016_TT_???_",       "2016_TT_hdampDown_???_", "2016_TT_hdampUp_???_", "2016_TT_underlyingEvtUp_???_", "2016_TT_underlyingEvtDown_???_", 
                            "2016_TT_erdOn_???_", "2016_TT_isrUp_???_",     "2016_TT_isrDown_???_", "2016_TT_fsrUp_???_",           "2016_TT_fsrDown_???_"
                ]
 
            # Closest number of event when using XXX numbered files for 300 to 1400 RPV+SYY
            elif "tRPV" in hyperconfig["atag"]:
                TT_2016 = [
                            "2016_TT_??_",        "2016_TT_?_",                  "2016_TT_hdampDown_??_",      "2016_TT_hdampDown_?_",          "2016_TT_hdampUp_??_", 
                            "2016_TT_hdampUp_?_", "2016_TT_underlyingEvtUp_??_", "2016_TT_underlyingEvtUp_?_", "2016_TT_underlyingEvtDown_??_", "2016_TT_underlyingEvtDown_?_", 
                            "2016_TT_erdOn_??_",  "2016_TT_erdOn_?_",            "2016_TT_isrUp_??_",          "2016_TT_isrUp_?_",              "2016_TT_isrDown_??_", 
                            "2016_TT_isrDown_?_", "2016_TT_fsrUp_??_",           "2016_TT_fsrUp_?_",           "2016_TT_fsrDown_??_",           "2016_TT_fsrDown_?_"
                ]

            else:
                TT_2016 = ["2016_TT_*_"]
        elif "twfsr" in hyperconfig["atag"]:
                TT_2016 = ["2016_TT_?_", "2016_TT_??_", "2016_TT_???_", "2016_TT_fsr*_"]
        else: 
            # Train on default tt POWHEG
            TT_2016 = ["2016_TT_?_", "2016_TT_??_", "2016_TT_???_"]

        if debug:
            TT_2016 = ["2016_TT_0_"]

        ################### Samples to validate on #####################
        if "visrd" in hyperconfig["atag"]:
            TT_2016_val = ["2016_TT_isrDown_"]
    
        elif "visru" in hyperconfig["atag"]:
            TT_2016_val = ["2016_TT_isrUp_"]

        elif "vfsrd" in hyperconfig["atag"]:
            TT_2016_val = ["2016_TT_fsrDown_"]
    
        elif "vfsru" in hyperconfig["atag"]:
            TT_2016_val = ["2016_TT_fsrUp_"]

        elif "vhdampd" in hyperconfig["atag"]:
            TT_2016_val = ["2016_TT_hdampDown_"]
    
        elif "vhdampu" in hyperconfig["atag"]:
            TT_2016_val = ["2016_TT_hdampUp_"]

        elif "vued" in hyperconfig["atag"]:
            TT_2016_val = ["2016_TT_underlyingEvtDown_"]
    
        elif "vueu" in hyperconfig["atag"]:
            TT_2016_val = ["2016_TT_underlyingEvtUp_"]

        elif "verd" in hyperconfig["atag"]:
            TT_2016_val = ["2016_TT_erdOn_"]

        elif "vmad" in hyperconfig["atag"]:
            TT_2016_val = ["2016_TTJets_"]

        else:
            TT_2016_val = ["2016_TT_?_", "2016_TT_??_", "2016_TT_???_"]

        for model in valModel.split("_"):
            valMasses = {"350", "550", "850", "1150"} 
            valMasses.add(valMass)
            for mass in valMasses:
                Signal_2016_val.append("2016*%s*mStop-%s*"%(model,mass))

        if "0l" in self.config["tree"]:
            TT_2017 = ["2017_TTToHadronic"]
            TT_2018 = ["2018pre_TTToHadronic"]
        elif "2l" in self.config["tree"]:
            TT_2017 = ["2017_TTTo2L2Nu"]
            TT_2018 = ["2018pre_TTTo2L2Nu"]
        else:
            TT_2017 = ["2017_TTToSemiLeptonic"]
            TT_2018 = ["2018pre_TTToSemiLeptonic"]

        for model in trainModel.split("_"):
            Signal_2016 += list("2016*%s*mStop*"%(model)+str(m) for m in range(self.config["minStopMass"],self.config["maxStopMass"]+50,50))
            Signal_2017 += list("2017*%s*mStop*"%(model)+str(m) for m in range(self.config["minStopMass"],self.config["maxStopMass"]+50,50))
            Signal_2018 += list("2018*%s*mStop*"%(model)+str(m) for m in range(self.config["minStopMass"],self.config["maxStopMass"]+50,50))

        TT = []; TTval = []; Signal = []; SignalVal = []; self.config["lumi"] = 0
        if "2016" in self.config["year"]: 
            TT                  += TT_2016
            TTval               += TT_2016_val
            Signal              += Signal_2016
            SignalVal           += Signal_2016_val
            self.config["lumi"] += 35900
        if "2017" in self.config["year"]:
            TT                  += TT_2017
            Signal              += Signal_2017
            self.config["lumi"] += 41500
        if "2018" in self.config["year"]:
            TT                  += TT_2018
            Signal              += Signal_2018
            self.config["lumi"] += 59800

        self.config["bkgd"]          = ("TT", TT)
        self.config["bkgdVal"]       = ("TTval", TTval)
        self.config["signal"]        = Signal
        self.config["signalVal"]     = SignalVal
        self.config["bkgdShift"]     = ("TT", TT_2016)
        #self.config["dataSet"]       = "2016_DisCo_0l_1l_WP0.98_05.05.2021/"
        self.config["dataSet"]       = "2016_DisCo_0l_1l_WP0.98_21.05.2021/"
        self.config["doBgWeight"]    = True
        self.config["doSgWeight"]    = True
        self.config["class_weight"]  = None
        self.config["sample_weight"] = None
        self.config["metrics"]       = ['accuracy']
        print("Using "+self.config["dataSet"]+" data set")
        print("Training on signal: ",       self.config["signal"])
        print("Training on background: ",   self.config["bkgd"][1])
        print("Validating on signal: ",     self.config["signalVal"])
        print("Validating on background: ", self.config["bkgdVal"][1])

        # Define ouputDir based on input config
        self.makeOutputDir(hyperconfig, replay)
        self.config.update(hyperconfig)

        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        
    # Makes a fully connected DNN 
    def DNN_model(self, n_var, n_first_layer, n_hidden_layers, n_last_layer, drop_out):

        inputs  = K.layers.Input(shape=(n_var,))
        M_layer = K.layers.Dense(n_first_layer, activation='relu')(inputs)

        for n in n_hidden_layers:
            M_layer = K.layers.Dense(n, activation='relu')(M_layer)
        
        M_layer   = K.layers.Dropout(drop_out)(M_layer)
        M_layer   = K.layers.Dense(n_last_layer, activation='relu')(M_layer)
        mainModel = K.models.Model(inputs=inputs, outputs=M_layer)
        
        return mainModel
    
    # Define loss functions
    def loss_crossentropy(self, c):
        def loss_model(y_true, y_pred):
            return c * K.losses.binary_crossentropy(y_true, y_pred)
        return loss_model
    
    def make_loss_adversary(self, c):
        def loss_adversary(y_true, y_pred):
            return c * K.losses.categorical_crossentropy(y_true, y_pred)
            #return c * K.backend.binary_crossentropy(y_true, y_pred)
        return loss_adversary

    def make_loss_MSE(self, c):
        def loss_MSE(y_true, y_pred):
            return c * K.losses.mean_squared_error(y_true, y_pred)
            #return c * K.losses.mean_squared_logarithmic_error(y_true, y_pred)
        return loss_MSE

    def make_loss_MAPE(self, c):
        def loss_MAPE(y_true, y_pred):
            return c * K.losses.mean_absolute_percentage_error(y_true, y_pred)
        return loss_MAPE

    def loss_corr(self, c):
        def correlationLoss(fake, y_pred):
            y1          = y_pred[:,  :1]
            y2          = y_pred[:, 2:3]
            y1_mean     = K.backend.mean(y1, axis=0)
            y1_centered = K.backend.abs(y1 - y1_mean)
            y2_mean     = K.backend.mean(y2, axis=0)
            y2_centered = K.backend.abs(y2 - y2_mean)
            corr_nr     = K.backend.sum(y1_centered * y2_centered, axis=0) 
            corr_dr1    = K.backend.sqrt(K.backend.sum(y1_centered * y1_centered, axis=0) + 1e-8)
            corr_dr2    = K.backend.sqrt(K.backend.sum(y2_centered * y2_centered, axis=0) + 1e-8)
            corr_dr     = corr_dr1 * corr_dr2
            corr        = corr_nr / corr_dr 
            return c * K.backend.sum(corr)
        return correlationLoss

    def loss_disco_par(self, c):
        def discoLossPar(y_mask, y_pred):            
            val_1        = tf.reshape(y_pred[:,  :1], [-1])
            val_2        = tf.reshape(y_pred[:, 2:3], [-1])
            normedweight = tf.ones_like(val_1)

            # Mask all signal events
            mask_sg         = tf.reshape(y_mask[:,  1:2], [-1])
            val_1_bg        = tf.boolean_mask(val_1, mask_sg)
            val_2_bg        = tf.boolean_mask(val_2, mask_sg)
            normedweight_bg = tf.boolean_mask(normedweight, mask_sg)

            return c * cor.distance_corr(val_1_bg, val_2_bg, normedweight_bg, 1)
        return discoLossPar

    def loss_disco(self, c1, c2):
        def discoLoss(y_mask, y_pred):            
            val_1        = tf.reshape(y_pred[:,  :1], [-1])
            val_2        = tf.reshape(y_pred[:, 2:3], [-1])
            normedweight = tf.ones_like(val_1)

            # Mask all signal events
            mask_sg         = tf.reshape(y_mask[:,  1:2], [-1])
            val_1_bg        = tf.boolean_mask(val_1, mask_sg)
            val_2_bg        = tf.boolean_mask(val_2, mask_sg)
            normedweight_bg = tf.boolean_mask(normedweight, mask_sg)

            mask_bg         = tf.reshape(y_mask[:, 0:1], [-1])
            val_1_sg        = tf.boolean_mask(val_1, mask_bg)
            val_2_sg        = tf.boolean_mask(val_2, mask_bg)
            normedweight_sg = tf.boolean_mask(normedweight, mask_bg)

            return c1 * cor.distance_corr(val_1_bg, val_2_bg, normedweight_bg, 1) + c2 * cor.distance_corr(val_1_sg, val_2_sg, normedweight_sg, 1)
        return discoLoss

    def loss_pear(self, c1, c2):
        def pearLoss(y_mask, y_pred):
            val_1 = tf.reshape(y_pred[:,  :1], [-1])
            val_2 = tf.reshape(y_pred[:, 2:3], [-1])

            # Mask all signal events
            mask_sg  = tf.reshape(y_mask[:,  1:2], [-1])
            val_1_bg = tf.boolean_mask(val_1, mask_sg)
            val_2_bg = tf.boolean_mask(val_2, mask_sg)

            mask_bg  = tf.reshape(y_mask[:, 0:1], [-1])
            val_1_sg = tf.boolean_mask(val_1, mask_bg)
            val_2_sg = tf.boolean_mask(val_2, mask_bg)

            return c1 * cor.pearson_corr_tf(val_1_bg, val_2_bg) + c2 * cor.pearson_corr_tf(val_1_sg, val_2_sg)
        return pearLoss

    def loss_abcd(self, c1, c2):
        def abcdLoss(y_mask, y_pred):

            val_1 = tf.reshape(y_pred[:,  :1], [-1])
            val_2 = tf.reshape(y_pred[:, 2:3], [-1])

            # Mask all signal events
            mask_sg  = tf.reshape(y_mask[:,  1:2], [-1])
            val_1_bg = tf.boolean_mask(val_1, mask_sg)
            val_2_bg = tf.boolean_mask(val_2, mask_sg)

            mask_bg  = tf.reshape(y_mask[:, 0:1], [-1])
            val_1_sg = tf.boolean_mask(val_1, mask_bg)
            val_2_sg = tf.boolean_mask(val_2, mask_bg)

            arand = np.random.uniform(0.2, 0.8)
            brand = np.random.uniform(0.2, 0.8)

            bA = tf.math.logical_and((val_1_bg>arand), (val_2_bg>brand)) 
            bB = tf.math.logical_and((val_1_bg<arand), (val_2_bg>brand)) 
            bC = tf.math.logical_and((val_1_bg>arand), (val_2_bg<brand)) 
            bD = tf.math.logical_and((val_1_bg<arand), (val_2_bg<brand)) 

            nbA = tf.size(tf.where(tf.equal(bA, True)))
            nbB = tf.size(tf.where(tf.equal(bB, True)))
            nbC = tf.size(tf.where(tf.equal(bC, True)))
            nbD = tf.size(tf.where(tf.equal(bD, True)))

            nbApred = tf.dtypes.cast(nbB*nbC/nbD, tf.float32)

            chi2 = tf.math.square((tf.dtypes.cast(nbA, tf.float32)-nbApred)/tf.math.sqrt(tf.dtypes.cast(nbA, tf.float32)))
            return c1 * chi2 + c2 * cor.pearson_corr_tf(val_1_bg, val_2_bg)
        return abcdLoss

    def loss_disco_alt(self, c):
        def discoLossAlt(y_true, y_pred):            
            # Decat truth and predicted
            val_1_disco_true = tf.reshape(y_true[:,  :2], [-1])
            val_2_disco_true = tf.reshape(y_true[:, 2:4], [-1])

            val_1_disco_pred = tf.reshape(y_pred[:,  :2], [-1])
            val_2_disco_pred = tf.reshape(y_pred[:, 2:4], [-1])

            # Calculate loss function
            val_1_disco_loss = K.losses.binary_crossentropy(val_1_disco_true, val_1_disco_pred)
            val_2_disco_loss = K.losses.binary_crossentropy(val_2_disco_true, val_2_disco_pred)

            val_1 = tf.reshape(y_pred[:,  :1], [-1])
            val_2 = tf.reshape(y_pred[:, 2:3], [-1])
            normedweight = tf.ones_like(val_1)

            return val_1_disco_loss + val_2_disco_loss + c * cor.distance_corr(val_1, val_2, normedweight, 1)
        return discoLossAlt

    def loss_crossentropy_comb(self, c1, c2):
        def loss_model_comb(y_true, y_pred):            
            # Decat truth and predicted
            val_1_disco_true = tf.reshape(y_true[:,  :2], [-1])
            val_2_disco_true = tf.reshape(y_true[:, 2:4], [-1])

            val_1_disco_pred = tf.reshape(y_pred[:,  :2], [-1])
            val_2_disco_pred = tf.reshape(y_pred[:, 2:4], [-1])

            # Calculate loss function
            val_1_disco_loss = K.losses.binary_crossentropy(val_1_disco_true, val_1_disco_pred)
            val_2_disco_loss = K.losses.binary_crossentropy(val_2_disco_true, val_2_disco_pred)

            val_1 = tf.reshape(y_pred[:,  :1], [-1])
            val_2 = tf.reshape(y_pred[:, 2:3], [-1])

            return c1 * (K.losses.mean_squared_error(val_1, val_2)) + c2 * (val_1_disco_loss + val_2_disco_loss)
            #return c1 * (K.losses.mean_squared_error(val_1, val_2)) + c2 * (val_1_disco_loss)

        return loss_model_comb


    def loss_disc(self, c):
        def loss_model_disc(y_true, y_pred):            

            # Calculate loss function
            val_loss = K.losses.binary_crossentropy(y_true, y_pred)

            return c * val_loss

        return loss_model_disc

    def make_model(self, trainData, trainDataTT):
        model, optimizer = main_model(self.config, trainData, trainDataTT)
        model.compile(loss=[self.loss_crossentropy_comb(c1=self.config["disc_comb_lambda"], c2=self.config["disc_lambda"]), self.make_loss_adversary(c=self.config["gr_lambda"]), self.loss_disco(c1=self.config["bg_cor_lambda"], c2=self.config["sg_cor_lambda"]), 
                            self.make_loss_MSE(c=self.config["reg_lambda"])], optimizer=optimizer, metrics=self.config["metrics"])
        return model

    def make_model_doubleDisco(self, trainData, trainDataTT):
        model, optimizer = model.doubleDisco(self.config, trainData, trainDataTT)
        model.compile(loss=[self.loss_crossentropy(c=self.config["disc_lambda"]), self.loss_crossentropy(c=self.config["disc_lambda"]), self.make_loss_adversary(c=self.config["gr_lambda"]), self.loss_disco_par(c=self.config["bg_cor_lambda"])], 
                      optimizer=optimizer, metrics=self.config["metrics"])
        return model

    def make_model_reg(self, trainData, trainDataTT):
        model, optimizer = model_reg(self.config, trainData, trainDataTT)
        model.compile(loss=K.losses.MeanSquaredError(), optimizer=optimizer)
        #model.compile(loss=[self.make_loss_MSE(c=1.0)], optimizer=optimizer)
        #model.compile(loss=[self.make_loss_MAPE(c=1.0)], optimizer=optimizer)
        return model

    def get_callbacks(self):
        tbCallBack = K.callbacks.TensorBoard(log_dir=self.logdir+"/log_graph",            histogram_freq=0,   write_graph=True,               write_images=True)
        log_model  = K.callbacks.ModelCheckpoint(self.config["outputDir"]+"/BestNN.hdf5", monitor='val_loss', verbose=self.config["verbose"], save_best_only=True)
        earlyStop  = K.callbacks.EarlyStopping(monitor="val_loss",                        min_delta=0,        patience=5, verbose=0,          mode="auto", baseline=None)
        callbacks  = []
        if self.config["verbose"] == 1: 
            #callbacks = [log_model, tbCallBack, earlyStop]
            #callbacks = [log_model, tbCallBack]
            callbacks = [tbCallBack]
        return callbacks

    def gpu_allow_mem_grow(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print("Error: gpu_mem_grow failed: ",e)

    def save_model_pb(self, model):
        #https://github.com/leimao/Frozen_Graph_TensorFlow/tree/master/TensorFlow_v2

        # Save model as hdf5 format
        #model.save(self.config["outputDir"]+"/keras_model")

        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

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
            print("Error: plot_model failed: ",e)

    def makeOutputDir(self,d,replay):
        outputDir = "Output/"
        for key in sorted(d.keys()):
            if "drop_out" in key or "lr" in key: continue
            outputDir += key+"_"+str(d[key])+"_"
        d["outputDir"] = outputDir
        if os.path.exists(d["outputDir"]) and not replay:
            print("Removing old training files: ", d["outputDir"])
            shutil.rmtree(d["outputDir"])

        if not replay: os.makedirs(d["outputDir"]+"/log_graph")    


    def defineVars(self):

        htVec_1l        = ["HT_trigger_pt30"]
        htVec_0l        = ["HT_trigger_pt45"]
        fwmVec_0l       = ["fwm2_top6_0l",    "fwm3_top6_0l",    "fwm4_top6_0l",   "fwm5_top6_0l"]
        jmtVec_0l       = ["jmt_ev0_top6_0l", "jmt_ev1_top6_0l", "jmt_ev2_top6_0l"]
        fwmVec_1l       = ["fwm2_top6_1l",    "fwm3_top6_1l",    "fwm4_top6_1l",   "fwm5_top6_1l"]
        jmtVec_1l       = ["jmt_ev0_top6_1l", "jmt_ev1_top6_1l", "jmt_ev2_top6_1l"]
        j4Vec           = ["Jet_pt_", "Jet_eta_", "Jet_m_", "Jet_phi_"]
        jFlavVec        = ["Jet_flavb_", "Jet_flavc_", "Jet_flavuds_", "Jet_flavq_", "Jet_flavg_"]
        jqgDiscVec      = ["Jet_ptD_", "Jet_axismajor_", "Jet_axisminor_"]
        jpfVec          = ["Jet_nEF_", "Jet_cEF_", "Jet_nHF_", "Jet_cHF_", "Jet_multiplicity_"]
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

        nJets = None; theVars = None

        if "0l" in self.config["tree"]:
            nJets = 6
            label = "_0l"
            theVars = j4Vec + jFlavVec + htVec_0l + fwmVec_0l + jmtVec_0l + stop1OldSeed + stop2OldSeed
        
        else:
            nJets = 7
            label = "_1l"
            theVars = j4Vec + jFlavVec + htVec_1l + fwmVec_1l + jmtVec_1l + stop1OldSeed + stop2OldSeed

        newVars = []
        for var in theVars:

            if "Jet_" in var:
                start = 0

                if "phi" in var:
                    start = 1
        
                for nJet in range(start,nJets):
                    newVars.append(var + str(nJet+1) + str(label))

            else: newVars.append(var)

        self.config["allVars"] = newVars


    def importData(self):
        # Import data
        print("----------------Preparing data------------------")
        temp = "*"
        #if self.config["debug"]:
        #    temp = "*_0_*"

        #Get Data set used in training and validation
        sgTrainSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+temp+"Train.root") for mass in self.config["signal"]) , [])
        bgTrainSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+bkgd+temp+"Train.root") for bkgd in self.config["bkgd"][1]), [])
        sgTestSet  = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+temp+"Test.root")  for mass in self.config["signal"]) , [])
        bgTestSet  = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+bkgd+temp+"Test.root")  for bkgd in self.config["bkgd"][1]), [])
        
        trainData, trainSg, trainBg = get_data(sgTrainSet, bgTrainSet, self.config, bgSampleFactor = self.config["bkgSampleFactor"], sgSampleFactor = self.config["sigSampleFactor"])
        testData,  testSg,  testBg  = get_data(sgTestSet,  bgTestSet,  self.config)

        self.config["nBkgTrainEvents"] = len(trainBg["data"])
        self.config["nSigTrainEvents"] = len(trainSg["data"])
        
        return sgTrainSet, bgTrainSet, sgTestSet, bgTestSet, trainData, trainSg, trainBg, testData, testSg, testBg

       
    def train(self):
        # Define vars for training
        self.defineVars()
        print("Training variables:")
        print(len(self.config["allVars"]), self.config["allVars"])

        # Get stuff from input ROOT files
        sgTrainSet, bgTrainSet, sgTestSet, bgTestSet, trainData, trainSg, trainBg, testData, testSg, testBg = self.importData()

        # Data set used to shift and scale the mean and std to 0 and 1 for all input variales into the network 
        trainDataTT = trainData

        # Make model
        print("----------------Preparing training model------------------")
        # Kelvin says no
        self.gpu_allow_mem_grow()
        model     = self.make_model(trainData, trainDataTT)
        callbacks = self.get_callbacks()
        maskTrain = np.concatenate((trainData["labels"], trainData["labels"]), axis=1)
        maskTest  = np.concatenate((testData["labels"],  testData["labels"]),  axis=1)

        # Training model
        print("----------------Training model------------------")
        result_log = model.fit(trainData["data"], [maskTrain, trainData["domain"], maskTrain, trainData["massesReco"]], 
                               batch_size=self.config["batch_size"], epochs=self.config["epochs"], callbacks=callbacks,
                               validation_data=(testData["data"], [maskTest, testData["domain"], maskTest, testData["massesReco"]], testData["sample_weight"]), 
                               sample_weight=trainData["sample_weight"])

        if self.saveAndPrint:
            # Model Visualization
            print("----------------Printed model layout------------------")
            self.plot_model(model)
            
            # Save trainig model as a protocol buffers file
            print("----------------Saving model------------------")
            self.save_model_pb(model)
       
        #Plot results
        print("----------------Validation of training------------------")
        val = Validation(model, self.config, trainData, trainSg, trainBg, result_log)

        metric = val.makePlots(self.doQuickVal, self.config["valMass"], self.config["valModel"])
        del val
        
        #Clean up training
        del model
    
        return metric


    def replay(self):

        model = K.models.load_model("./"+self.config["outputDir"]+"/keras_model")

        sgTrainSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+"*Train.root") for mass in self.config["signal"]) , [])
        bgTrainSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+bkgd+"*Train.root") for bkgd in self.config["bkgd"][1]), [])
        sgTestSet  = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+"*Test.root")  for mass in self.config["signal"]) , [])
        bgTestSet  = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+bkgd+"*Test.root")  for bkgd in self.config["bkgd"][1]), [])

        trainData, trainSg, trainBg = get_data(sgTrainSet, bgTrainSet, self.config)

        val = Validation(model, self.config, trainData, trainSg, trainBg)
        metric = val.makePlots(doQuickVal, valMass)
        del val

if __name__ == '__main__':
    usage = "usage: %prog [options]"
    parser = argparse.ArgumentParser(usage)
    parser.add_argument("--quickVal",        dest="quickVal",        help="Do quick validation",            action="store_true", default=False) 
    parser.add_argument("--reweight",        dest="reweight",        help="Do event reweighting",           action="store_true", default=False) 
    parser.add_argument("--json",            dest="json",            help="JSON config file",               default="NULL"                    ) 
    parser.add_argument("--minMass",         dest="minMass",         help="Minimum stop mass to train on",  default=300                       )
    parser.add_argument("--maxMass",         dest="maxMass",         help="Maximum stop mass to train on",  default=1400                      ) 
    parser.add_argument("--valMass",         dest="valMass",         help="Stop mass to validate on",       default=500                       ) 
    parser.add_argument("--valModel",        dest="valModel",        help="Signal model to validate on",    default="RPV_SYY_SHH"             ) 
    parser.add_argument("--model",           dest="model",           help="Signal model to train on",       type=str, default="RPV_SYY_SHH"   ) 
    parser.add_argument("--replay",          dest="replay",          help="Replay saved model",             action="store_true", default=False) 
    parser.add_argument("--year",            dest="year",            help="Year(s) to train on",            type=str, default="2016_2017_2018") 
    parser.add_argument("--tree",            dest="tree",            help="myMiniTree to train on",         default="myMiniTree"              )
    parser.add_argument("--saveAndPrint",    dest="saveAndPrint",    help="Save pb and print model",        action="store_true", default=False)
    parser.add_argument("--seed",            dest="seed",            help="Use specific seed",              type=int, default=-1              )
    parser.add_argument("--debug",           dest="debug",           help="Do some debugging",              action="store_true", default=False)
    parser.add_argument("--maskNjet",        dest="maskNjet",        help="mask Njet bin/bins in training", default=[-1], nargs="+", type=int )
    parser.add_argument("--bkgSampleFactor", dest="bkgSampleFactor", help="how many times to sample bkg",   default=1, type=int               )
    parser.add_argument("--sigSampleFactor", dest="sigSampleFactor", help="how many times to sample sig",   default=1, type=int               )

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

    replay = args.replay

    hyperconfig = {}
    if args.json != "NULL": 
        with open(str(args.json), "r") as f:
            hyperconfig = json.load(f)
    else: 
        hyperconfig = {"atag" : "TEST", "disc_comb_lambda": 0.0, "gr_lambda": 5.0, "disc_lambda": 10.0, "bg_cor_lambda": 2000.0, "sg_cor_lambda" : 2000.0, "reg_lambda": 0.001, "nNodes":100, "nNodesD":1, "nNodesM":100, "nHLayers":1, "nHLayersD":1, "nHLayersM":1, "drop_out":0.3, "batch_size":10000, "epochs":15, "lr":0.001}

    t = Train(USER, args.debug, masterSeed, replay, args.saveAndPrint, hyperconfig, args.quickVal, args.reweight, minStopMass=args.minMass, maxStopMass=args.maxMass, trainModel=args.model, valMass=args.valMass, valModel=args.valModel, year=args.year, tree=args.tree, maskNjet=args.maskNjet, bkgSampleFactor=args.bkgSampleFactor, sigSampleFactor=args.sigSampleFactor)

    if replay: t.replay()

    elif args.json != "NULL": 

        metric = t.train()

        with open(str(args.json), 'w') as f:
            json.dump(metric, f)
    else:
        t.train()
        print("----------------Ran with default config------------------")
