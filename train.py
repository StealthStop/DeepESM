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
from Models import main_model, doubleDisco_model, model_reg

class Train:
    #def __init__(self):
    #    print("Constructor")
        
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
            #return c * K.losses.mean_squared_error(y_true, y_pred)
            return c * K.losses.mean_squared_logarithmic_error(y_true, y_pred)
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

    def loss_disco(self, c):
        def discoLoss(y_mask, y_pred):            
            val_1 = tf.reshape(y_pred[:,  :1], [-1])
            val_2 = tf.reshape(y_pred[:, 2:3], [-1])
            normedweight = tf.ones_like(val_1)

            ##Mask all signal events
            #mask = tf.reshape(y_mask[:,  1:2], [-1])
            #val_1 = tf.boolean_mask(val_1, mask)
            #val_2 = tf.boolean_mask(val_2, mask)
            #normedweight = tf.boolean_mask(normedweight, mask)
            return c * cor.distance_corr(val_1, val_2, normedweight, 1)
        return discoLoss

    def make_model(self, config, trainData, trainDataTT):
        model, optimizer = main_model(config, trainData, trainDataTT)
        model.compile(loss=[self.loss_crossentropy(c=config["disc1_lambda"]), self.loss_crossentropy(c=config["disc2_lambda"]), self.make_loss_adversary(c=config["gr_lambda"]), self.loss_disco(c=config["cor_lambda"]), 
                            self.make_loss_MSE(c=config["reg_lambda"])], optimizer=optimizer, metrics=config["metrics"])
        return model

    def make_doubleDisco_model(self, config, trainData, trainDataTT):
        model, optimizer = doubleDisco_model(config, trainData, trainDataTT)
        model.compile(loss=[self.loss_crossentropy(c=1.0), self.loss_crossentropy(c=1.0), self.make_loss_adversary(c=config["gr_lambda"]), self.loss_disco(c=config["cor_lambda"])], 
                      optimizer=optimizer, metrics=config["metrics"])
        return model

    def make_model_reg(self, config, trainData, trainDataTT):
        model, optimizer = model_reg(config, trainData, trainDataTT)
        model.compile(loss=K.losses.MeanSquaredError(), optimizer=optimizer)
        #model.compile(loss=[self.make_loss_MSE(c=1.0)], optimizer=optimizer)
        #model.compile(loss=[self.make_loss_MAPE(c=1.0)], optimizer=optimizer)
        return model

    def get_callbacks(self, config):
        tbCallBack = K.callbacks.TensorBoard(log_dir="/storage/local/data1/gpuscratch/jhiltbra"+"/log_graph", histogram_freq=0, write_graph=True, write_images=True)
        log_model = K.callbacks.ModelCheckpoint(config["outputDir"]+"/BestNN.hdf5", monitor='val_loss', verbose=config["verbose"], save_best_only=True)
        earlyStop = K.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=0, mode="auto", baseline=None)
        callbacks = []
        if config["verbose"] == 1: 
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

    def save_model_pb(self, model, config):
        #https://github.com/leimao/Frozen_Graph_TensorFlow/tree/master/TensorFlow_v2

        # Save model as hdf5 format
        #K.models.save_model(model, config["outputDir"]+"/keras_model.h5")

        # Save model to SavedModel format
        #tf.saved_model.save(model, config["outputDir"]+"/keras_model")

        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        config["input_output"] = list(x.name.split(':')[0] for x in frozen_func.inputs + frozen_func.outputs)
        
        # Save frozen graph from frozen ConcreteFunction to hard drive
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=config["outputDir"], name="keras_frozen.pb", as_text=False)

    def plot_model(self, model, config):
        try:
            K.utils.plot_model(model, to_file=config["outputDir"]+"/model.png", show_shapes=True)
        except AttributeError as e:
            print("Error: plot_model failed: ",e)

    def makeOutputDir(self,config):
        outputDir = "Output/"
        for key in sorted(config.keys()):
            outputDir += key+"_"+str(config[key])+"_"
        config["outputDir"] = outputDir
        if os.path.exists(config["outputDir"]):
            print("Removing old training files: ", config["outputDir"])
            shutil.rmtree(config["outputDir"])
        os.makedirs(config["outputDir"]+"/log_graph")    
        return config

    def defineVars(self,config):
        #jVec1 = ["Jet_pt_", "Jet_eta_", "Jet_m_", "Jet_dcsv_", "Jet_ptD_", "Jet_axismajor_", "Jet_axisminor_", "Jet_multiplicity_" ]
        jVec1 = ["Jet_pt_", "Jet_eta_", "Jet_m_", "Jet_dcsv_"]
        jVec2 = ["Jet_phi_"]
        #jVec1AK8 = ["JetsAK8Cands_pt_", "JetsAK8Cands_eta_", "JetsAK8Cands_m_", "JetsAK8Cands_SDM_", "JetsAK8Cands_Pruned_", "JetsAK8Cands_T21_"]
        #jVec2AK8 = ["JetsAK8Cands_phi_"]
        lepton = ["GoodLeptons_pt_1", "GoodLeptons_eta_1", "GoodLeptons_phi_1", "GoodLeptons_m_1"]
        #lepton = ["GoodLeptons_pt_1", "GoodLeptons_eta_1", "GoodLeptons_phi_1"]

        MET = ["lvMET_cm_pt", "lvMET_cm_eta", "lvMET_cm_phi", "lvMET_cm_m"]
        eventShapeVars = ["fwm2_top6", "fwm3_top6", "fwm4_top6", "fwm5_top6", "jmt_ev0_top6", "jmt_ev1_top6", "jmt_ev2_top6"]
        #numJets = ["NGoodJets_pt30_double"]
        numJets = []

        #extra = ["deepESM_val", "HT_trigger_pt30", "stop1_PtRank_1l_mass", "stop2_PtRank_1l_mass"]
        extra = ["HT_trigger_pt30", "stop1_PtRank_1l_mass", "stop2_PtRank_1l_mass"]

        nJets = 8 
        nJetsAK8 = 4
        jVecs =  list(y+str(x+1) for y in jVec1 for x in range(nJets)) 
        jVecs += list(y+str(x+1) for y in jVec2 for x in range(1,nJets)) 
        #jVecsAK8 =  list(y+str(x+1) for y in jVec1AK8 for x in range(nJetsAK8))
        #jVecsAK8 += list(y+str(x+1) for y in jVec2AK8 for x in range(1,nJetsAK8))
        #config["allVars"] = jVecs + lepton + eventShapeVars + MET + numJets + extra
        #config["allVars"] = jVecs + lepton + eventShapeVars + ["HT_trigger_pt30", "stop1_PtRank_1l_mass", "stop2_PtRank_1l_mass"] + jVecsAK8
        config["allVars"] = jVecs + lepton + eventShapeVars + MET + numJets + extra# + jVecsAK8
        #config["allVars"] = jVecs + lepton + eventShapeVars + numJets + extra# + jVecsAK8
        return config
        
    def train(self, config = {"gr_lambda": 1.0, "disc1_lambda": 1.0, "disc2_lambda": 1.0, "cor_lambda": 45.0, "reg_lambda": 1.0, "nNodes":300, "nNodesD":40, "nNodesM":500,
                              "nHLayers":1, "nHLayersD":1, "nHLayersM":1, "drop_out":0.3,
                              "batch_size":16384, "epochs":2, "lr":0.001}, doQuickVal=False, minStopMass=300, maxStopMass=1400, model="*", valMass=500, year = "2016_2017_2018"):
   
        # Define ouputDir based on input config
        config = self.makeOutputDir(config)
        config["minNJetBin"] = 7
        config["maxNJetBin"] = 11
        config["verbose"] = 1
        config["Mask"] = True
        config["Mask_nJet"] = 7

        # Define vars for training
        self.defineVars(config)

        # Import data
        print("----------------Preparing data------------------")
        TTJets_2016 = ["2016_TTJets_Incl", "2016_TTJets_SingleLeptFromT", "2016_TTJets_SingleLeptFromTbar", "2016_TTJets_DiLept", 
                       "2016_TTJets_HT-600to800", "2016_TTJets_HT-800to1200", "2016_TTJets_HT-1200to2500", "2016_TTJets_HT-2500toInf"]
        TTJets_2017 = ["2017_TTJets_Incl", "2017_TTJets_SingleLeptFromT", "2017_TTJets_SingleLeptFromTbar", "2017_TTJets_DiLept", 
                       "2017_TTJets_HT-600to800", "2017_TTJets_HT-800to1200", "2017_TTJets_HT-1200to2500", "2017_TTJets_HT-2500toInf"]
        TT_2016 = ["2016_TT"]
        #TT_2017 = ["2017_TTToSemiLeptonic","2017_TTTo2L2Nu","2017_TTToHadronic"]
        TT_2017 = ["2017_TTToSemiLeptonic"]
        #TT_2018 = ["2018pre_TTToSemiLeptonic","2018pre_TTTo2L2Nu","2018pre_TTToHadronic"]
        TT_2018 = ["2018pre_TTToSemiLeptonic"]
        config["minStopMass"] = int(minStopMass)
        config["maxStopMass"] = int(maxStopMass)
        Signal_2016 = list("2016%smStop*"%(model)+str(m) for m in range(config["minStopMass"],config["maxStopMass"]+50,50))
        Signal_2017 = list("2017%smStop*"%(model)+str(m) for m in range(config["minStopMass"],config["maxStopMass"]+50,50))
        Signal_2018 = list("2018%smStop*"%(model)+str(m) for m in range(config["minStopMass"],config["maxStopMass"]+50,50))

        TT = []; Signal = []; config["lumi"] = 0
        if "2016" in year: 
            TT += TT_2016
            Signal += Signal_2016
            config["lumi"] += 35900
        if "2017" in year:
            TT += TT_2017
            Signal += Signal_2017
            config["lumi"] += 41500
        if "2018" in year:
            TT += TT_2018
            Signal += Signal_2018
            config["lumi"] += 59800

        config["ttbarMC"] = ("TT", TT)
        config["massModels"] = Signal
        config["otherttbarMC"] = ("TT 2017", TT_2017)
        config["othermassModels"] = Signal_2017
        config["ttbarMCShift"] = ("TT", TT_2016)
        config["dataSet"] = "MVA_Training_Files_FullRun2_V3/"
        config["doBgWeight"] = True
        config["doSgWeight"] = True
        config["class_weight"] = None #{0: {0: 2.0, 1: 1.0}, 1: {0: 2.0, 1: 1.0}, 2: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}}
        config["sample_weight"] = None
        config["metrics"]=['accuracy']
        config["year"] = year 
        print("Using "+config["dataSet"]+" data set")
        print("Training variables:")
        print(len(config["allVars"]), config["allVars"])
        print("Training on mass models: ", config["massModels"])
        print("Training on ttbarMC: ", config["ttbarMC"][1])
        
        #Get Data set used in training and validation
        sgTrainSet = sum( (getSamplesToRun(config["dataSet"]+"MyAnalysis_"+mass+"*Train.root") for mass in config["massModels"]) , [])
        bgTrainSet = sum( (getSamplesToRun(config["dataSet"]+"MyAnalysis_"+ttbar+"*Train.root") for ttbar in config["ttbarMC"][1]), [])
        sgTestSet = sum( (getSamplesToRun(config["dataSet"]+"MyAnalysis_"+mass+"*Test.root") for mass in config["massModels"]) , [])
        bgTestSet = sum( (getSamplesToRun(config["dataSet"]+"MyAnalysis_"+ttbar+"*Test.root") for ttbar in config["ttbarMC"][1]), [])

        trainData, trainSg, trainBg = get_data(sgTrainSet, bgTrainSet, config)
        testData, testSg, testBg = get_data(sgTestSet, bgTestSet, config)
        
        #Data set used to shift and scale the mean and std to 0 and 1 for all input variales into the network 
        #bgTrainTT = sum( (getSamplesToRun(config["dataSet"]+"MyAnalysis_"+ttbar+"*Train.root") for ttbar in config["ttbarMCShift"][1]), [])
        #trainDataTT, trainSgTT, trainBgTT = get_data(sgTrainSet, bgTrainTT, config)
        trainDataTT = trainData

        # Make model
        print("----------------Preparing training model------------------")
        # Kelvin says no
        self.gpu_allow_mem_grow()
        model = self.make_model(config, trainData, trainDataTT)
        #model = self.make_doubleDisco_model(config, trainData, trainDataTT)
        #model = self.make_model_reg(config, trainData, trainDataTT)
        callbacks = self.get_callbacks(config)
        maskTrain = np.concatenate((trainData["labels"],trainData["labels"]), axis=1)
        maskTest = np.concatenate((testData["labels"],testData["labels"]), axis=1)
        
        # Training model
        print("----------------Training model------------------")
        result_log = model.fit(trainData["data"], [trainData["labels"], trainData["labels"], trainData["domain"], maskTrain, trainData["masses"]], 
                               batch_size=config["batch_size"], epochs=config["epochs"], callbacks=callbacks,
                               validation_data=(testData["data"], [testData["labels"], testData["labels"], testData["domain"], maskTest, testData["masses"]], testData["sample_weight"]), 
                               sample_weight=config["sample_weight"])
        #result_log = model.fit(trainData["data"], [trainData["labels"], trainData["labels"], trainData["domain"], maskTrain], 
        #                       batch_size=config["batch_size"], epochs=config["epochs"], callbacks=callbacks,
        #                       validation_data=(testData["data"], [testData["labels"], testData["labels"], testData["domain"], maskTest], testData["sample_weight"]), 
        #                       sample_weight=trainData["sample_weight"])
        #result_log = model.fit(trainData["data"], trainData["masses"], epochs=config["epochs"], sample_weight=trainData["sample_weight"],
        #                       validation_data=(testData["data"], testData["masses"], testData["sample_weight"]), callbacks=callbacks)

        # Model Visualization
        print("----------------Printed model layout------------------")
        self.plot_model(model, config)
        
        # Save trainig model as a protocol buffers file
        print("----------------Saving model------------------")
        self.save_model_pb(model, config)
        
        #Plot results
        print("----------------Validation of training------------------")
        val = Validation(model, config, sgTrainSet, trainData, trainSg, trainBg, result_log)
        metric = val.makePlots(doQuickVal, valMass)
        del val
        
        #Clean up training
        del model
    
        return metric
        
if __name__ == '__main__':
    usage = "usage: %prog [options]"
    parser = argparse.ArgumentParser(usage)
    parser.add_argument("--quickVal", dest="quickVal", help="Do quick validation", action="store_true", default=False) 
    parser.add_argument("--json",    dest="json",    help="JSON config file", default="NULL") 
    parser.add_argument("--minMass", dest="minMass", help="Minimum stop mass to train on", default=300)
    parser.add_argument("--maxMass", dest="maxMass", help="Maximum stop mass to train on", default=1400) 
    parser.add_argument("--valMass", dest="valMass", help="Stop mass to validate on", default=500) 
    parser.add_argument("--model",   dest="model",   help="Signal model to train on", default="*") 
    parser.add_argument("--year",    dest="year",    help="Year(s) to train on", default="2016_2017_2018") 

    args = parser.parse_args()

    model = "*"
    if args.model != "*": model = "*%s*"%(args.model)

    t = Train()
    if args.json != "NULL": 
        config = None
        with open(str(args.json), "r") as f:
            config = json.load(f)
        print(config)

        metric = t.train(config, args.quickVal)

        with open(str(args.json), 'w') as f:
          json.dump(metric, f)
    else:
        t.train(doQuickVal=args.quickVal, minStopMass=args.minMass, maxStopMass=args.maxMass, model=model, valMass=args.valMass, year=args.year)
        print("----------------Ran with default config------------------")

