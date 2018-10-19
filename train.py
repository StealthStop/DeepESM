import os
import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
import pandas as pd
from DataGetter import get_data
from flipGradientTF import GradientReversal
from glob import glob
import shutil
from Validation import Validation

class Train:
    def __init__(self):
        print "Constructor"
        
    # Makes a fully connected DNN 
    def create_main_model(self, n_var, n_first_layer, n_hidden_layers, n_last_layer, drop_out):
        inputs = keras.layers.Input(shape=(n_var,))
        M_layer = keras.layers.Dense(n_first_layer, activation='relu')(inputs)
        for n in n_hidden_layers:
            M_layer = keras.layers.Dense(n, activation='relu')(M_layer)
        M_layer = keras.layers.Dropout(drop_out)(M_layer)
        M_layer = keras.layers.Dense(n_last_layer, activation='relu')(M_layer)
        mainModel = keras.models.Model(inputs=inputs, outputs=M_layer)
        return mainModel
    
    # Define loss functions
    def make_loss_model(self, c):
        def loss_model(y_true, y_pred):
            return c * keras.backend.binary_crossentropy(y_true, y_pred)
        return loss_model
    
    def make_loss_adversary(self, c):
        def loss_adversary(y_true, y_pred):
            return c * keras.backend.categorical_crossentropy(y_true, y_pred)
            #return c * keras.backend.binary_crossentropy(y_true, y_pred)
        return loss_adversary
        
    def train(self, config = {"minNJetBin": 7, "maxNJetBin": 11, "gr_lambda": 4, "nNodes":70, "nNodesD":10,
                              "nHLayers":1, "nHLayersD":1, "drop_out":0.7, "batch_size":2048, "epochs":100,
                              "lr":0.001, "verbose":1, "Mask":False, "Mask_nJet":7}):
    
        # Define ouputDir based on input config
        outputDir = "Output/"
        for key in sorted(config.keys()):
            outputDir += key+"_"+str(config[key])+"_"
        config["outputDir"] = outputDir
            
        # Define vars for training
        jVec = ["Jet_pt_", "Jet_eta_", "Jet_phi_", "Jet_m_"]
        lepton = ["GoodLeptons_pt_1", "GoodLeptons_eta_1", "GoodLeptons_phi_1", "GoodLeptons_m_1"]
        MET = ["lvMET_cm_pt", "lvMET_cm_eta", "lvMET_cm_phi", "lvMET_cm_m"]
        eventShapeVars = ["fwm2_top6", "fwm3_top6", "fwm4_top6", "fwm5_top6", "jmt_ev0_top6", "jmt_ev1_top6", "jmt_ev2_top6"]
        numJets = ["NGoodJets_double"]
        nJets = 7
        if config["Mask"]: nJets = config["Mask_nJet"]
        jVecs = list(y+str(x+1) for y in jVec for x in range(nJets))
        config["allVars"] = jVecs + lepton
        
        # Import data
        print("----------------Preparing data------------------")
        TTJets = ("TTJets", ["TTJets_SingleLeptFromT", "TTJets_SingleLeptFromTbar", "TTJets_DiLept", "TTJets_HT-600to800", "TTJets_HT-800to1200", "TTJets_HT-1200to2500", "TTJets_HT-2500toInf"])
        TT = ("TT", ["TT"])                  
        TT_TTJets = ("TT+TTJets", ["TT", "TTJets_SingleLeptFromT", "TTJets_SingleLeptFromTbar", "TTJets_DiLept", "TTJets_HT-600to800", "TTJets_HT-800to1200", "TTJets_HT-1200to2500", "TTJets_HT-2500toInf"])
        TTJets_SingleLep = ("TTJets_SingleLep", ["TTJets_SingleLeptFromT_Train", "TTJets_SingleLeptFromTbar_Train"])
        config["dataSet"] = "BackGroundMVA_V10_CM_All_GoodJets_AllTTbarSamples/"
        config["massModels"] = ["350","450","550","650","750","850"]
        config["ttbarMC"] = TT_TTJets
        config["otherttbarMC"] = TTJets_SingleLep
        config["doBgWeight"] = True
        config["doSgWeight"] = False
        print "Using "+config["dataSet"]+" data set"
        print "Training variables:"
        print config["allVars"]
        print "Training on mass models: ", config["massModels"]
        print "Training on ttbarMC: ", config["ttbarMC"][0]
        if os.path.exists(config["outputDir"]):
            print "Removing old training files: ", config["outputDir"]
            shutil.rmtree(config["outputDir"])
        os.makedirs(config["outputDir"]+"/log_graph")    
        
        sgTrainSet = sum( (glob(config["dataSet"]+"trainingTuple_*_division_0_*_"+mass+"*_training_0.h5") for mass in config["massModels"]) , [])
        bgTrainSet = sum( (glob(config["dataSet"]+"trainingTuple_*_division_0_"+ttbar+"_training_0.h5") for ttbar in config["ttbarMC"][1]) , [])
    
        sgTestSet = sum( (glob(config["dataSet"]+"trainingTuple_*_division_2_*_"+mass+"*_test_0.h5") for mass in config["massModels"]) , [])
        bgTestSet = sum( (glob(config["dataSet"]+"trainingTuple_*_division_2_"+ttbar+"_test_0.h5") for ttbar in config["ttbarMC"][1]) , [])
        
        trainData, trainSg, trainBg = get_data(sgTrainSet, bgTrainSet, config)
        testData, testSg, testBg = get_data(sgTestSet, bgTestSet, config)
    
        bgTrainTT = glob(config["dataSet"]+"trainingTuple_*_division_0_TT_training_0.h5")
        trainDataTT, trainSgTT, trainBgTT = get_data(sgTrainSet, bgTrainTT, config)
    
        # Make and train model
        print("----------------Preparing training model------------------")
        class_weight = None#{0: {0: 1.0, 1: 1.0}, 1: {0: 1.0, 1: 5.0, 2: 25.0, 3: 125.0, 4: 625.0}}    
        sample_weight = None#{0: trainData["Weight"][:,0].tolist(), 1: trainData["Weight"][:,0].tolist()}
        #optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        optimizer = keras.optimizers.Adam(lr=config["lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        n_hidden_layers = list(config["nNodes"] for x in range(config["nHLayers"]))
        n_hidden_layers_D = list(config["nNodesD"] for x in range(config["nHLayersD"]))
        Flip = GradientReversal(config["gr_lambda"])    
    
        # Setting inter_op_parallelism_threads=1 fixes a memory leak when calling model.predict()
        cfg = K.tf.ConfigProto(inter_op_parallelism_threads=1)
        cfg.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=cfg))
        main_input = keras.layers.Input(shape=(trainData["data"].shape[1],), name='main_input')
        # Set the rescale inputs to have unit variance centered at 0 between -1 and 1
        layer = keras.layers.Lambda(lambda x: (x - K.constant(trainDataTT["mean"])) * K.constant(trainDataTT["scale"]), name='normalizeData')(main_input)
        layer = keras.layers.Dense(config["nNodes"], activation='relu')(layer)
        for n in n_hidden_layers:
            layer = keras.layers.BatchNormalization()(layer)
            layer = keras.layers.Dense(n, activation='relu')(layer)
        layer = keras.layers.Dropout(config["drop_out"])(layer)
        first_output = keras.layers.Dense(trainData["labels"].shape[1], activation='softmax', name='first_output')(layer)
        
        layer = Flip(first_output)
        #layer = keras.layers.Dense(nNodesD, activation='relu')(first_output)
        for n in n_hidden_layers_D:
            layer = keras.layers.BatchNormalization()(layer)
            layer = keras.layers.Dense(n, activation='relu')(layer)
        layer = keras.layers.Dropout(config["drop_out"])(layer)
        second_output = keras.layers.Dense(trainData["domain"].shape[1], activation='softmax', name='second_output')(layer)
        
        model = keras.models.Model(inputs=main_input, outputs=[first_output, second_output], name='model')
        model.compile(loss=[self.make_loss_model(c=1.0) , self.make_loss_adversary(c=1.0)], optimizer=optimizer, metrics=['accuracy'])
        tbCallBack = keras.callbacks.TensorBoard(log_dir="./"+outputDir+"/log_graph", histogram_freq=0, write_graph=True, write_images=True)
        log_model = keras.callbacks.ModelCheckpoint(outputDir+"/BestNN.hdf5", monitor='val_loss', verbose=config["verbose"], save_best_only=True)
        callbacks = []
        if config["verbose"] == 1:
            callbacks = [log_model, tbCallBack]
        result_log = model.fit(trainData["data"], [trainData["labels"], trainData["domain"]], batch_size=config["batch_size"], epochs=config["epochs"], class_weight=class_weight,
                               validation_data=(testData["data"], [testData["labels"], testData["domain"]]), callbacks=callbacks, sample_weight=sample_weight)
    
        # Model Visualization
        keras.utils.plot_model(model, to_file=outputDir+"/model.png", show_shapes=True)
        
        # Save trainig model as a protocol buffers file
        inputName = model.input.op.name.split(':')[0]
        outputName = model.output[0].op.name.split(':')[0]
        print "Input name:", inputName
        print "Output name:", outputName
        config["input_output"] = [inputName, outputName]
        saver = tf.train.Saver()
        saver.save(keras.backend.get_session(), outputDir+"/keras_model.ckpt")
        export_path="./"+outputDir+"/"
        freeze_graph_binary = "python freeze_graph.py"
        graph_file=export_path+"keras_model.ckpt.meta"
        ckpt_file=export_path+"keras_model.ckpt"
        output_file=export_path+"keras_frozen.pb"
        command = freeze_graph_binary+" --input_meta_graph="+graph_file+" --input_checkpoint="+ckpt_file+" --output_graph="+output_file+" --output_node_names="+outputName+" --input_binary=true"
        os.system(command)
        
        #Plot results
        print("----------------Validation of training------------------")
        val = Validation(model, config, sgTrainSet, trainData, trainSg, trainBg, result_log)
        config, metric = val.plot()
        del val
        
        #Clean up training
        K.clear_session()
        tf.reset_default_graph()
        del model
        
        return config, metric

if __name__ == '__main__':
    t = Train()
    config, metric = t.train()
    print config
    
    for key in metric:
        print key, metric[key]
