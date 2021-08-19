#!/bin/env python
import os
import json
import time
import shutil
import argparse
import datetime
import subprocess
import numpy as np
from glob import glob
import multiprocessing
import tensorflow as tf
import tensorflow.keras as K

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

class Train:
    def __init__(self, USER, nTops, dRbjets, useJECs, debug, seed, replay, saveAndPrint, hyperconfig, doQuickVal=False, doReweight=False, minStopMass=300, maxStopMass=1400, trainModel="RPV_SYY_SHH", evalMass=500, evalModel="RPV_SYY_SHH", year = "2016_2017_2018", tree = "myMiniTree", maskNjet = [-1], procCats=False, massCats=False, njetsCats=False):

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
        self.config["doReweight"]  = doReweight
        self.config["useJECs"]     = useJECs
        self.config["nTops"]       = nTops
        self.config["dRbjets"]     = dRbjets

        # Depending on final state, different pt requirements
        # and resultant objects are used
        ptCut = "pt30"
        if "0l" in tree:
            ptCut = "pt45"

        # Labels for extracting relevant information from the
        # dataframes constructed from the inputs ROOT files
        self.config["massLabel"]       = "mass"
        self.config["domainLabel"]     = "NGoodJets_%s_double"%(ptCut)
        self.config["regressionLabel"] = "stop1_ptrank_mass"
        self.config["modelLabel"]      = "model"
        self.config["weightLabel"]     = "Weight"
        self.config["ntopsLabel"]      = "ntops"
        self.config["dRbjetsLabel"]    = "dR_bjets"

        self.doQuickVal           = doQuickVal
        self.saveAndPrint         = saveAndPrint
        self.config["trainModel"] = trainModel
        self.config["evalMass"]   = evalMass
        self.config["evalModel"]  = evalModel
        self.config["year"]       = year
        self.config["tree"]       = tree

        self.config["procCats"]   = procCats
        self.config["massCats"]   = massCats
        self.config["njetsCats"]  = njetsCats

        if "0l" in self.config["tree"]:
            self.config["minNJetBin"] = 6
            self.config["maxNJetBin"] = 12
            self.config["verbose"]    = 1
        else:
            self.config["minNJetBin"] = 7
            self.config["maxNJetBin"] = 11
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

        TT_2016      = None; TT_2017      = None; TT_2018      = None
        TT_2016_eval = None; TT_2017_eval = None; TT_2018_eval = None

        Signal_2016      = []; Signal_2017      = []; Signal_2018      = []
        Signal_2016_eval = []; Signal_2017_eval = []; Signal_2018_eval = []

        self.config["evalBkg"] = None

        ################### Samples to train on #####################
        extra = "_[TV]*"

        TT_2016 = ["2016_TT*"]

        ################### Samples to validate on #####################
        if "vpow" in hyperconfig["atag"]:
            TT_2016_eval = ["2016_TT%s"%(extra)]
            self.config["evalBkg"] = 0

        elif "vmad" in hyperconfig["atag"]:
            TT_2016_eval = ["2016_TTJets%s"%(extra)]
            self.config["evalBkg"] = 1

        elif "verd" in hyperconfig["atag"]:
            TT_2016_eval = ["2016_TT_erdOn%s"%(extra)]
            self.config["evalBkg"] = 2

        elif "vhdampu" in hyperconfig["atag"]:
            TT_2016_eval = ["2016_TT_hdampUp%s"%(extra)]
            self.config["evalBkg"] = 3

        elif "vhdampd" in hyperconfig["atag"]:
            TT_2016_eval = ["2016_TT_hdampDown%s"%(extra)]
            self.config["evalBkg"] = 4

        elif "vueu" in hyperconfig["atag"]:
            TT_2016_eval = ["2016_TT_underlyingEvtUp%s"%(extra)]
            self.config["evalBkg"] = 5

        elif "vued" in hyperconfig["atag"]:
            TT_2016_eval = ["2016_TT_underlyingEvtDown%s"%(extra)]
            self.config["evalBkg"] = 6

        elif "vfsru" in hyperconfig["atag"]:
            TT_2016_eval = ["2016_TT_fsrUp%s"%(extra)]
            self.config["evalBkg"] = 7

        elif "vfsrd" in hyperconfig["atag"]:
            TT_2016_eval = ["2016_TT_fsrDown%s"%(extra)]
            self.config["evalBkg"] = 8

        elif "visru" in hyperconfig["atag"]:
            TT_2016_eval = ["2016_TT_isrUp%s"%(extra)]
            self.config["evalBkg"] = 9

        elif "visrd" in hyperconfig["atag"]:
            TT_2016_eval = ["2016_TT_isrDown%s"%(extra)]
            self.config["evalBkg"] = 10

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
                Signal_2016_eval.append("2016*%s*mStop-%s"%(model,mass))

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
            Signal_2016 += list("2016*%s*mStop-"%(model)+str(m) for m in range(self.config["minStopMass"],self.config["maxStopMass"]+50,50))
            Signal_2017 += list("2017*%s*mStop-"%(model)+str(m) for m in range(self.config["minStopMass"],self.config["maxStopMass"]+50,50))
            Signal_2018 += list("2018*%s*mStop-"%(model)+str(m) for m in range(self.config["minStopMass"],self.config["maxStopMass"]+50,50))

        TT = []; TTeval = []; Signal = []; SignalEval = []; self.config["lumi"] = 0
        if "2016" in self.config["year"]: 
            TT                  += TT_2016
            TTeval              += TT_2016_eval
            Signal              += Signal_2016
            SignalEval          += Signal_2016_eval
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
        self.config["bkgdEval"]      = ("TTeval", TTeval)
        self.config["signal"]        = Signal
        self.config["signalEval"]    = SignalEval
        self.config["bkgdShift"]     = ("TT", TT_2016)
        self.config["dataSet"]       = "2016_NN_inputs_20210802/"
        self.config["doBgWeight"]    = True
        self.config["doSgWeight"]    = True
        self.config["class_weight"]  = None
        self.config["sample_weight"] = None
        self.config["metrics"]       = ['accuracy']
        print("Using "+self.config["dataSet"]+" data set\n")
        print("Training on signal: ",       self.config["signal"], "\n")
        print("Training on background: ",   self.config["bkgd"][1], "\n")
        print("Validating on signal: ",     self.config["signalEval"], "\n")
        print("Validating on background: ", self.config["bkgdEval"][1], "\n")

        # Define ouputDir based on input config
        self.makeOutputDir(hyperconfig, replay)
        self.config.update(hyperconfig)

        if not os.path.exists(self.logdir): os.makedirs(self.logdir)

    # Define loss functions
    def loss_mass_reg(self, c):
        def regLoss(y_true, y_pred):
            return c * K.losses.mean_squared_error(y_true, y_pred)
        return regLoss

    def loss_disco(self, c1, c2, c3, g):
        def discoLoss(y_mask, y_pred):            
            val_1 = tf.reshape(y_pred[:,  :1], [-1])
            val_2 = tf.reshape(y_pred[:, 2:3], [-1])
            normedweight = tf.ones_like(val_1)

            #Mask all signal events
            mask_sg = tf.reshape(y_mask[:,  1:2], [-1])
            val_1_bg = tf.boolean_mask(val_1, mask_sg)
            val_2_bg = tf.boolean_mask(val_2, mask_sg)
            temp1 = tf.boolean_mask(val_1, mask_sg)
            temp2 = tf.boolean_mask(val_2, mask_sg)

            normedweight_bg = tf.boolean_mask(normedweight, mask_sg)

            mask_bg = tf.reshape(y_mask[:, 0:1], [-1])
            val_1_sg = tf.boolean_mask(val_1, mask_bg)
            val_2_sg = tf.boolean_mask(val_2, mask_bg)
            normedweight_sg = tf.boolean_mask(normedweight, mask_bg)

            d1 = g.uniform(shape=(), minval=0.0, maxval=1.0)
            d2 = g.uniform(shape=(), minval=0.0, maxval=1.0)

            nbA = tf.reduce_sum(tf.sigmoid(10e1*(val_1_bg-d1))*tf.sigmoid(10e1*(val_2_bg-d2)))
            nbB = tf.reduce_sum(tf.sigmoid(10e1*(val_2_bg-d2))) - nbA
            nbC = tf.reduce_sum(tf.sigmoid(10e1*(val_1_bg-d1))) - nbA
            nbD = 2.0*tf.reduce_sum(tf.sigmoid(0.0*val_1_bg)) - nbA - nbB - nbC

            nbApred = tf.cond(nbD == tf.constant(0.0, dtype=tf.float32), lambda: nbA, lambda: nbB*nbC/nbD)
            frac = tf.cond(nbApred*nbA == tf.constant(0.0, dtype=tf.float32), lambda: tf.constant(0.0, dtype=tf.float32), lambda: abs(nbApred - nbA)/nbApred)

            return c3 * frac + c1 * cor.distance_corr(temp1, temp2, normedweight_bg, 1) + c2 * cor.distance_corr(val_1_sg, val_2_sg, normedweight_sg, 1)
        return discoLoss

    def loss_disc(self, c):
        def loss_model_disc(y_true, y_pred):            
            # Decat truth and predicted
            val_1_disco_true = tf.reshape(y_true[:,  :2], [-1])
            val_2_disco_true = tf.reshape(y_true[:, 2:4], [-1])

            val_1_disco_pred = tf.reshape(y_pred[:,  :2], [-1])
            val_2_disco_pred = tf.reshape(y_pred[:, 2:4], [-1])

            # Calculate loss function
            val_1_disco_loss = K.losses.binary_crossentropy(val_1_disco_true, val_1_disco_pred)
            val_2_disco_loss = K.losses.binary_crossentropy(val_2_disco_true, val_2_disco_pred)

            return c * (val_1_disco_loss + val_2_disco_loss)

        return loss_model_disc

    def make_model(self, scales, means, regShape, discoShape, inputShape):
        model, optimizer = main_model(self.config, scales, means, regShape, discoShape, inputShape)
        g = tf.random.Generator.from_seed(self.config["seed"]) 
        model.compile(loss=[self.loss_disc(c=self.config["disc_lambda"]), self.loss_disco(c1=self.config["bkg_disco_lambda"], c2=self.config["sig_disco_lambda"], c3=self.config["abcd_close_lambda"], g=g), self.loss_mass_reg(c=self.config["mass_reg_lambda"])], optimizer=optimizer, metrics=self.config["metrics"])
        return model

    def getSamplesToRun(self, names):
        s = glob(names)
        if len(s) == 0:
            raise Exception("No files find that correspond to: "+names)
        return s

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
        outputDir = "Output/%s__"%(d["atag"])
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
        
        outputDir += hyperStr + "_" + nodesStr + "_" + layersStr + "_" + learningStr + "_" + otherStr
        d["outputDir"] = outputDir[:-1]
        if os.path.exists(d["outputDir"]) and not replay:
            print("Removing old training files: ", d["outputDir"] + "\n")
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
        dRbjets         = ["dR_bjets"]
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
            if not self.config["dRbjets"]:
                theVars += dRbjets

        else:
            nJets = 7
            label = "_1l"
            theVars = j4Vec + jFlavVec + htVec_1l + fwmVec_1l + jmtVec_1l + stop1OldSeed + stop2OldSeed

        newVars = []; auxVars = []
        for var in theVars:

            if "Jet_" in var:
                start = 0

                if "phi" in var:
                    start = 1
        
                for nJet in range(start,nJets):
                    newVars.append(var + str(nJet+1) + str(label))

            else: newVars.append(var)

        self.config["trainVars"] = newVars
        
        # We load auxiliary variables that are not to be used as direct inputs
        # DataLoader handles these separately
        auxVars.append(self.config["weightLabel"])
        auxVars.append(self.config["modelLabel"])
        auxVars.append(self.config["regressionLabel"])
        auxVars.append(self.config["massLabel"])
        auxVars.append(self.config["domainLabel"])

        if "0l" in self.config["tree"]:
            auxVars.append(self.config["ntopsLabel"])
            auxVars.append(self.config["dRbjetsLabel"])
       
        self.config["auxVars"] = auxVars

    def importData(self):
        # Import data
        print("\n----------------Preparing data------------------")
        temp = "*"

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
       
        self.loader     = DataLoader(self.config, sgTrainSet, bgTrainSet)
        self.valLoader  = DataLoader(self.config, sgValSet,   bgValSet)
        self.testLoader = DataLoader(self.config, sgTestSet,  bgTestSet)

    def train(self):
        # Define vars for training
        self.defineVars()
        print("Training variables:")
        print(len(self.config["trainVars"]), self.config["trainVars"])

        # Get stuff from input ROOT files
        self.importData()

        self.config["nBkgTrainEvents"] = self.loader.getNumBkgEvents()
        self.config["nSigTrainEvents"] = self.loader.getNumSigEvents()

        scales = self.loader.getDataScales()
        means  = self.loader.getDataMeans()

        regShape, domainShape, discoShape, inputShape = self.loader.getShapes()

        # Make model
        print("\n----------------Preparing training model------------------")
        # Kelvin says no
        self.gpu_allow_mem_grow()
        model     = self.make_model(scales, means, regShape, discoShape, inputShape)
        callbacks = self.get_callbacks()

        # Training model
        print("\n----------------Training model------------------")
        result_log = model.fit(self.loader, epochs=self.config["epochs"], callbacks=callbacks, validation_data=self.testLoader)

        if self.saveAndPrint:
            # Model Visualization
            print("\n----------------Printed model layout------------------")
            try:
                self.plot_model(model)
            except Exception as e:
                print("WARNING: Could not print model !", e)

            # Save trainig model as a protocol buffers file
            print("\n----------------Saving model------------------")
            try:
                self.save_model_pb(model)
            except Exception as e:
                print("ERROR: Could not save model pb !", e)
       
        #Plot results
        print("\n----------------Validation of training------------------")
        val = Validation(model, self.config, self.loader, self.valLoader, self.evalLoader, self.testLoader, result_log)

        metric = val.makePlots(self.doQuickVal, self.config["evalMass"], self.config["evalModel"])
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
        metric = val.makePlots(doQuickVal, evalMass)
        del val

if __name__ == '__main__':
    usage = "usage: %prog [options]"
    parser = argparse.ArgumentParser(usage)
    parser.add_argument("--quickVal",     dest="quickVal",     help="Do quick validation",            action="store_true", default=False) 
    parser.add_argument("--reweight",     dest="reweight",     help="Do event reweighting",           action="store_true", default=False) 
    parser.add_argument("--json",         dest="json",         help="JSON config file",               default="NULL"                    ) 
    parser.add_argument("--minMass",      dest="minMass",      help="Minimum stop mass to train on",  default=300                       )
    parser.add_argument("--maxMass",      dest="maxMass",      help="Maximum stop mass to train on",  default=1400                      ) 
    parser.add_argument("--evalMass",     dest="evalMass",     help="Stop mass to evaluate on",       default=500                       ) 
    parser.add_argument("--evalModel",    dest="evalModel",    help="Signal model to evaluate on",    default="RPV_SYY_SHH"             ) 
    parser.add_argument("--model",        dest="model",        help="Signal model to train on",       type=str, default="RPV_SYY_SHH"   ) 
    parser.add_argument("--replay",       dest="replay",       help="Replay saved model",             action="store_true", default=False) 
    parser.add_argument("--year",         dest="year",         help="Year(s) to train on",            type=str, default="2016_2017_2018") 
    parser.add_argument("--tree",         dest="tree",         help="myMiniTree to train on",         default="myMiniTree"              )
    parser.add_argument("--saveAndPrint", dest="saveAndPrint", help="Save pb and print model",        action="store_true", default=False)
    parser.add_argument("--seed",         dest="seed",         help="Use specific seed",              type=int, default=-1              )
    parser.add_argument("--nTops",        dest="nTops",        help="Number of tops for 0L",          type=int, default=2               )
    parser.add_argument("--dRbjets",      dest="dRbjets",      help="Cut on dR for bjets",            action="store_true", default=False)
    parser.add_argument("--debug",        dest="debug",        help="Do some debugging",              action="store_true", default=False)
    parser.add_argument("--useJECs",      dest="useJECs",      help="Use JEC/JER variations",         action="store_true", default=False)
    parser.add_argument("--maskNjet",     dest="maskNjet",     help="mask Njet bin/bins in training", default=[-1], nargs="+", type=int )
    parser.add_argument("--procCats",     dest="procCats",     help="Balance batches bkg/sig",        default=False, action="store_true")
    parser.add_argument("--massCats",     dest="massCats",     help="Balance batches among masses",   default=False, action="store_true")
    parser.add_argument("--njetsCats",    dest="njetsCats",    help="Balance batches among njets",    default=False, action="store_true")

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
        hyperconfig = {"atag" : "Perfect_vpow", "disc_lambda": 30.0, "bkg_disco_lambda": 2000.0, "sig_disco_lambda" : 0.0, "mass_reg_lambda": 0.0001, "abcd_close_lambda" : 30.0, "disc_nodes":300, "mass_reg_nodes":100, "disc_layers":1, "mass_reg_layers":1, "dropout":0.3, "batch":10000, "epochs":10, "other_lr" : 0.001, "disc_lr":0.001, "mass_reg_lr" : 0.5}

    t = Train(USER, args.nTops, args.dRbjets, args.useJECs, args.debug, masterSeed, replay, args.saveAndPrint, hyperconfig, args.quickVal, args.reweight, minStopMass=args.minMass, maxStopMass=args.maxMass, trainModel=args.model, evalMass=args.evalMass, evalModel=args.evalModel, year=args.year, tree=args.tree, maskNjet=args.maskNjet, procCats=args.procCats, massCats=args.massCats, njetsCats=args.njetsCats)

    if replay: t.replay()

    elif args.json != "NULL": 

        metric = t.train()

        metricName = args.json.replace(".json", "_metric.json")
        with open(str(metricName), 'w') as f:
            json.dump(metric, f)
    else:
        t.train()
        print("----------------Ran with default config------------------")
