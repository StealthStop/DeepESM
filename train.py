#!/bin/env python
import sys, ast, os
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras as K
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
from DataGetter import get_data,getSamplesToRun
import shutil
from Validation import Validation
from Correlation import Correlation as cor
import json
import argparse
from Models import main_model_alt, main_model_alt2, main_model, doubleDisco_model, model_reg
import os
import time

class Train:
    def __init__(self, USER, seed, replay, saveAndPrint, hyperconfig, doQuickVal=False, doReweight=False, minStopMass=300, maxStopMass=1400, model="*", valMass=500, year = "2016_2017_2018", tree = "myMiniTree"):
        self.user = USER
        self.logdir = "/storage/local/data1/gpuscratch/%s"%(self.user)
        self.config = {}
        self.config["seed"] = seed
        self.config["minStopMass"] = int(minStopMass)
        self.config["maxStopMass"] = int(maxStopMass)
        self.config["doReweight"] = doReweight

        self.doQuickVal = doQuickVal
        self.saveAndPrint = saveAndPrint
        self.model = model
        self.valMass = valMass
        self.config["year"] = year
        self.config["tree"] = tree

        self.config["minNJetBin"] = 7
        self.config["maxNJetBin"] = 11
        self.config["verbose"] = 1
        self.config["Mask"] = False
        self.config["Mask_nJet"] = 7

        TTJets_2016 = ["2016_TTJets_Incl", "2016_TTJets_SingleLeptFromT", "2016_TTJets_SingleLeptFromTbar", "2016_TTJets_DiLept", 
                       "2016_TTJets_HT-600to800", "2016_TTJets_HT-800to1200", "2016_TTJets_HT-1200to2500", "2016_TTJets_HT-2500toInf"]
        TTJets_2017 = ["2017_TTJets_Incl", "2017_TTJets_SingleLeptFromT", "2017_TTJets_SingleLeptFromTbar", "2017_TTJets_DiLept", 
                       "2017_TTJets_HT-600to800", "2017_TTJets_HT-800to1200", "2017_TTJets_HT-1200to2500", "2017_TTJets_HT-2500toInf"]

        TT_2016 = ["2016_TT"]; TT_2017 = None; TT_2018 = None
        if "0l" in self.config["tree"]:
            TT_2017 = ["2017_TTToHadronic"]
            TT_2018 = ["2018pre_TTToHadronic"]
        elif "2l" in self.config["tree"]:
            TT_2017 = ["2017_TTTo2L2Nu"]
            TT_2018 = ["2018pre_TTTo2L2Nu"]
        else:
            TT_2017 = ["2017_TTToSemiLeptonic"]
            TT_2018 = ["2018pre_TTToSemiLeptonic"]

        Signal_2016 = list("2016%smStop*"%(self.model)+str(m) for m in range(self.config["minStopMass"],self.config["maxStopMass"]+50,50))
        Signal_2017 = list("2017%smStop*"%(self.model)+str(m) for m in range(self.config["minStopMass"],self.config["maxStopMass"]+50,50))
        Signal_2018 = list("2018%smStop*"%(self.model)+str(m) for m in range(self.config["minStopMass"],self.config["maxStopMass"]+50,50))

        TT = []; Signal = []; self.config["lumi"] = 0
        if "2016" in self.config["year"]: 
            TT += TT_2016
            Signal += Signal_2016
            self.config["lumi"] += 35900
        if "2017" in self.config["year"]:
            TT += TT_2017
            Signal += Signal_2017
            self.config["lumi"] += 41500
        if "2018" in self.config["year"]:
            TT += TT_2018
            Signal += Signal_2018
            self.config["lumi"] += 59800

        self.config["ttbarMC"] = ("TT", TT)
        self.config["massModels"] = Signal
        self.config["ttbarMCShift"] = ("TT", TT_2016)
        self.config["dataSet"] = "MVA_Training_Files_FullRun2_V3/"
        self.config["doBgWeight"] = True
        self.config["doSgWeight"] = True
        self.config["class_weight"] = None
        self.config["sample_weight"] = None
        self.config["metrics"]=['accuracy']
        print("Using "+self.config["dataSet"]+" data set")
        print("Training on mass models: ", self.config["massModels"])
        print("Training on ttbarMC: ", self.config["ttbarMC"][1])

        # Define ouputDir based on input config
        self.makeOutputDir(hyperconfig, replay)
        self.config.update(hyperconfig)

        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        
    # Makes a fully connected DNN 
    def DNN_model(self, n_var, n_first_layer, n_hidden_layers, n_last_layer, drop_out):
        inputs = K.layers.Input(shape=(n_var,))
        M_layer = K.layers.Dense(n_first_layer, activation='relu')(inputs)
        for n in n_hidden_layers:
            M_layer = K.layers.Dense(n, activation='relu')(M_layer)
        M_layer = K.layers.Dropout(drop_out)(M_layer)
        M_layer = K.layers.Dense(n_last_layer, activation='relu')(M_layer)
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
            y1 = y_pred[:,  :1]
            y2 = y_pred[:, 2:3]
            y1_mean = K.backend.mean(y1, axis=0)
            y1_centered = K.backend.abs(y1 - y1_mean)
            y2_mean = K.backend.mean(y2, axis=0)
            y2_centered = K.backend.abs(y2 - y2_mean)
            corr_nr = K.backend.sum(y1_centered * y2_centered, axis=0) 
            corr_dr1 = K.backend.sqrt(K.backend.sum(y1_centered * y1_centered, axis=0) + 1e-8)
            corr_dr2 = K.backend.sqrt(K.backend.sum(y2_centered * y2_centered, axis=0) + 1e-8)
            corr_dr = corr_dr1 * corr_dr2
            corr = corr_nr / corr_dr 
            return c * K.backend.sum(corr)
        return correlationLoss

    def loss_disco(self, c1, c2):
        def discoLoss(y_mask, y_pred):            
            val_1 = tf.reshape(y_pred[:,  :1], [-1])
            val_2 = tf.reshape(y_pred[:, 2:3], [-1])
            normedweight = tf.ones_like(val_1)

            #Mask all signal events
            mask_sg = tf.reshape(y_mask[:,  1:2], [-1])
            val_1_bg = tf.boolean_mask(val_1, mask_sg)
            val_2_bg = tf.boolean_mask(val_2, mask_sg)
            normedweight_bg = tf.boolean_mask(normedweight, mask_sg)

            mask_bg = tf.reshape(y_mask[:, 0:1], [-1])
            val_1_sg = tf.boolean_mask(val_1, mask_bg)
            val_2_sg = tf.boolean_mask(val_2, mask_bg)
            normedweight_sg = tf.boolean_mask(normedweight, mask_bg)

            return c1 * cor.distance_corr(val_1_bg, val_2_bg, normedweight_bg, 1) + c2 * cor.distance_corr(val_1_sg, val_2_sg, normedweight_sg, 1)
        return discoLoss

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

        return loss_model_comb

    def make_model(self, trainData, trainDataTT):
        model, optimizer = main_model(self.config, trainData, trainDataTT)
        model.compile(loss=[self.loss_crossentropy(c=self.config["disc1_lambda"]), self.loss_crossentropy(c=self.config["disc2_lambda"]), self.make_loss_adversary(c=self.config["gr_lambda"]), self.loss_disco(c=self.config["cor_lambda"]), 
                            self.make_loss_MSE(c=self.config["reg_lambda"])], optimizer=optimizer, metrics=self.config["metrics"])
        return model

    def make_model_alt(self, trainData, trainDataTT):
        model, optimizer = main_model_alt(self.config, trainData, trainDataTT)
        model.compile(loss=[self.make_loss_adversary(c=self.config["gr_lambda"]), self.loss_disco_alt(c=self.config["cor_lambda"]), 
                            self.make_loss_MSE(c=self.config["reg_lambda"])], optimizer=optimizer, metrics=self.config["metrics"])
        return model

    def make_model_alt2(self, trainData, trainDataTT):
        model, optimizer = main_model_alt2(self.config, trainData, trainDataTT)
        model.compile(loss=[self.loss_crossentropy_comb(c1=self.config["disc_comb_lambda"], c2=self.config["disc_lambda"]), self.make_loss_adversary(c=self.config["gr_lambda"]), self.loss_disco(c1=self.config["bg_cor_lambda"], c2=self.config["sg_cor_lambda"]), 
                            self.make_loss_MSE(c=self.config["reg_lambda"])], optimizer=optimizer, metrics=self.config["metrics"])
        return model

    def make_doubleDisco_model(self, trainData, trainDataTT):
        model, optimizer = doubleDisco_model(self.config, trainData, trainDataTT)
        model.compile(loss=[self.loss_crossentropy(c=1.0), self.loss_crossentropy(c=1.0), self.make_loss_adversary(c=self.config["gr_lambda"]), self.loss_disco(c=self.config["cor_lambda"])], 
                      optimizer=optimizer, metrics=self.config["metrics"])
        return model

    def make_model_reg(self, trainData, trainDataTT):
        model, optimizer = model_reg(self.config, trainData, trainDataTT)
        model.compile(loss=K.losses.MeanSquaredError(), optimizer=optimizer)
        #model.compile(loss=[self.make_loss_MSE(c=1.0)], optimizer=optimizer)
        #model.compile(loss=[self.make_loss_MAPE(c=1.0)], optimizer=optimizer)
        return model

    def get_callbacks(self):
        tbCallBack = K.callbacks.TensorBoard(log_dir=self.logdir+"/log_graph", histogram_freq=0, write_graph=True, write_images=True)
        log_model = K.callbacks.ModelCheckpoint(self.config["outputDir"]+"/BestNN.hdf5", monitor='val_loss', verbose=self.config["verbose"], save_best_only=True)
        earlyStop = K.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=0, mode="auto", baseline=None)
        callbacks = []
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
        model.save(self.config["outputDir"]+"/keras_model")

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
            outputDir += key+"_"+str(d[key])+"_"
        d["outputDir"] = outputDir
        if os.path.exists(d["outputDir"]) and not replay:
            print("Removing old training files: ", d["outputDir"])
            shutil.rmtree(d["outputDir"])

        if not replay: os.makedirs(d["outputDir"]+"/log_graph")    

    def defineVars(self):

        if "0l" in self.config["tree"]:
            numJets        = ["HT_trigger_pt45", "NGoodJets_pt45_double", "NGoodBJets_pt45_double"]
            eventShapeVars = ["fwm2_top6",       "fwm3_top6",             "fwm4_top6", "fwm5_top6", 
                              "jmt_ev0_top6",    "jmt_ev1_top6",          "jmt_ev2_top6"]
            jVec1          = ["Jet_dcsv_",       "Jet_pt_",               "Jet_eta_",  "Jet_m_"]
            jVec2          = ["Jet_phi_"]
            extra          = ["Mass_stop1_PtRank_cm_TopSeed_maskedISR", "Mass_stop2_PtRank_cm_TopSeed_maskedISR", 
                              "Pt_stop1_PtRank_cm_TopSeed_maskedISR",   "Pt_stop2_PtRank_cm_TopSeed_maskedISR",  
                              "Phi_stop1_PtRank_cm_TopSeed_maskedISR",  "Phi_stop2_PtRank_cm_TopSeed_maskedISR", 
                              "Eta_stop1_PtRank_cm_TopSeed_maskedISR",  "Eta_stop2_PtRank_cm_TopSeed_maskedISR", ]
            
            nJets  = 7 
            jVecs =  list(y+str(x+1) for y in jVec1 for x in range(nJets)) 
            jVecs += list(y+str(x+1) for y in jVec2 for x in range(1,nJets)) 

            self.config["allVars"] = numJets + eventShapeVars + jVecs + extra

        else:
            jVec1 = ["Jet_pt_", "Jet_eta_", "Jet_m_", "Jet_ptD_", "Jet_axismajor_", "Jet_axisminor_", "Jet_multiplicity_", "Jet_dcsv_"]
            jVec2 = ["Jet_phi_"]
            lepton = ["GoodLeptons_pt_1", "GoodLeptons_eta_1", "GoodLeptons_phi_1"]

            MET = ["lvMET_cm_pt", "lvMET_cm_eta", "lvMET_cm_phi"]
            eventShapeVars = ["fwm2_top6", "fwm3_top6", "fwm4_top6", "fwm5_top6", "jmt_ev0_top6", "jmt_ev1_top6", "jmt_ev2_top6"]
            numJets = ["NGoodJets_pt30_double", "NGoodBJets_pt30_double"]

            #extra = ["HT_trigger_pt30", "stop1_PtRank_1l_mass", "stop2_PtRank_1l_mass"]
            extra  = ["HT_trigger_pt30", "Mass_stop1_PtRank_cm_OldSeed", "Mass_stop2_PtRank_cm_OldSeed", 
                      "Pt_stop1_PtRank_cm_OldSeed",   "Pt_stop2_PtRank_cm_OldSeed",  
                      "Phi_stop1_PtRank_cm_OldSeed",  "Phi_stop2_PtRank_cm_OldSeed", 
                      "Eta_stop1_PtRank_cm_OldSeed",  "Eta_stop2_PtRank_cm_OldSeed", ]

            nJets = 7
            jVecs =  list(y+str(x+1) for y in jVec1 for x in range(nJets))
            jVecs += list(y+str(x+1) for y in jVec2 for x in range(1,nJets))

            self.config["allVars"] = jVec + lepton + MET + eventShapeVars + extra

    def importData(self):
        # Import data
        print("----------------Preparing data------------------")
        #Get Data set used in training and validation
        sgTrainSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+"*Train.root") for mass in self.config["massModels"]) , [])
        bgTrainSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+ttbar+"*Train.root") for ttbar in self.config["ttbarMC"][1]), [])
        sgTestSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+"*Test.root") for mass in self.config["massModels"]) , [])
        bgTestSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+ttbar+"*Test.root") for ttbar in self.config["ttbarMC"][1]), [])

        trainData, trainSg, trainBg = get_data(sgTrainSet, bgTrainSet, self.config)
        testData, testSg, testBg = get_data(sgTestSet, bgTestSet, self.config)
        
        return sgTrainSet, bgTrainSet, sgTestSet, bgTestSet, trainData, trainSg, trainBg, testData, testSg, testBg
       
    def train(self):
   
        # Define vars for training
        self.defineVars()
        print("Training variables:")
        print(len(self.config["allVars"]), self.config["allVars"])

        #Get stuff from input ROOT files
        sgTrainSet, bgTrainSet, sgTestSet, bgTestSet, trainData, trainSg, trainBg, testData, testSg, testBg = self.importData()

        #Data set used to shift and scale the mean and std to 0 and 1 for all input variales into the network 
        trainDataTT = trainData

        # Make model
        print("----------------Preparing training model------------------")
        # Kelvin says no
        self.gpu_allow_mem_grow()
        model = self.make_model_alt2(trainData, trainDataTT)
        callbacks = self.get_callbacks()
        maskTrain = np.concatenate((trainData["labels"],trainData["labels"]), axis=1)
        maskTest = np.concatenate((testData["labels"],testData["labels"]), axis=1)
        
        # Training model
        print("----------------Training model------------------")
        result_log = model.fit(trainData["data"], [maskTrain, trainData["domain"], maskTrain, trainData["masses"]], 
                               batch_size=self.config["batch_size"], epochs=self.config["epochs"], callbacks=callbacks,
                               validation_data=(testData["data"], [maskTest, testData["domain"], maskTest, testData["masses"]], testData["sample_weight"]), 
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
        val = Validation(model, self.config, sgTrainSet, trainData, trainSg, trainBg, result_log)

        metric = val.makePlots(self.doQuickVal, self.valMass)
        del val
        
        #Clean up training
        del model
    
        return metric

    def replay(self):

        model = K.models.load_model("./"+self.config["outputDir"]+"/keras_model")

        sgTrainSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+"*Train.root") for mass in self.config["massModels"]) , [])
        bgTrainSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+ttbar+"*Train.root") for ttbar in self.config["ttbarMC"][1]), [])
        sgTestSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+mass+"*Test.root") for mass in self.config["massModels"]) , [])
        bgTestSet = sum( (getSamplesToRun(self.config["dataSet"]+"MyAnalysis_"+ttbar+"*Test.root") for ttbar in self.config["ttbarMC"][1]), [])

        trainData, trainSg, trainBg = get_data(sgTrainSet, bgTrainSet, self.config)

        val = Validation(model, self.config, sgTrainSet, trainData, trainSg, trainBg)
        metric = val.makePlots(doQuickVal, valMass)
        del val

if __name__ == '__main__':
    usage = "usage: %prog [options]"
    parser = argparse.ArgumentParser(usage)
    parser.add_argument("--quickVal",     dest="quickVal",     help="Do quick validation", action="store_true", default=False) 
    parser.add_argument("--reweight",     dest="reweight",     help="Do event reweighting", action="store_true", default=False) 
    parser.add_argument("--json",         dest="json",         help="JSON config file", default="NULL") 
    parser.add_argument("--minMass",      dest="minMass",      help="Minimum stop mass to train on", default=300)
    parser.add_argument("--maxMass",      dest="maxMass",      help="Maximum stop mass to train on", default=1400) 
    parser.add_argument("--valMass",      dest="valMass",      help="Stop mass to validate on", default=500) 
    parser.add_argument("--model",        dest="model",        help="Signal model to train on", type=str, default="*") 
    parser.add_argument("--replay",       dest="replay",       help="Replay saved model", action="store_true", default=False) 
    parser.add_argument("--year",         dest="year",         help="Year(s) to train on", type=str, default="2016_2017_2018") 
    parser.add_argument("--tree",         dest="tree",         help="myMiniTree to train on", default="myMiniTree")
    parser.add_argument("--saveAndPrint", dest="saveAndPrint", help="Save pb and print model", action="store_true", default=False)
    parser.add_argument("--seed",         dest="seed",         help="Use specific seed", type=int, default=-1)

    args = parser.parse_args()

    # Get seed from time, but allow user to reseed with their own number
    masterSeed = int(time.time())
    if args.seed != -1:
        masterSeed = args.seed

    # Supposedly using 0 makes things predictable
    os.environ["PYTHONHASHSEED"] = "0"

    # Supposedly makes calculations on GPU deterministic
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    # Seed the tensorflow here, seed numpy in datagetter
    tf.random.set_seed(masterSeed)

    # _After_ setting seed, for reproduceability, try these resetting/clearing commands
    # Enforce 64 bit precision
    K.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    K.backend.set_floatx('float64')

    USER = os.getenv("USER")

    model = "*"
    if args.model != "*": model = "*%s*"%(args.model)

    replay = args.replay

    hyperconfig = {}
    if args.json != "NULL": 
        with open(str(args.json), "r") as f:
            hyperconfig = json.load(f)
    else: 
        hyperconfig = {"atag" : "GoldenTEST", "disc_comb_lambda": 0.1, "gr_lambda": 2.2, "disc_lambda": 0.5, "bg_cor_lambda": 1000.0, "sg_cor_lambda" : 0.0, "reg_lambda": 0.01, "nNodes":100, "nNodesD":1, "nNodesM":100, "nHLayers":1, "nHLayersD":1, "nHLayersM":1, "drop_out":0.3, "batch_size":10000, "epochs":60, "lr":0.001}

    t = Train(USER, masterSeed, replay, args.saveAndPrint, hyperconfig, args.quickVal, args.reweight, minStopMass=args.minMass, maxStopMass=args.maxMass, model=model, valMass=args.valMass, year=args.year, tree=args.tree)

    if replay: t.replay()

    elif args.json != "NULL": 

        metric = t.train()

        with open(str(args.json), 'w') as f:
            json.dump(metric, f)
    else:
        t.train()
        print("----------------Ran with default config------------------")
