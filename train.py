import os
import tensorflow as tf
import numpy as np
import pandas as pd
from DataGetter import DataGetter as dg
from glob import glob

# Makes a fully connected DNN 
def create_main_model(n_var, n_first_layer, n_hidden_layers, n_last_layer, drop_out):
    inputs = tf.keras.layers.Input(shape=(n_var,))
    M_layer = tf.keras.layers.Dense(n_first_layer, activation='relu')(inputs)
    for n in n_hidden_layers:
        M_layer = tf.keras.layers.Dense(n, activation='relu')(M_layer)
    M_layer = tf.keras.layers.Dropout(drop_out)(M_layer)
    M_layer = tf.keras.layers.Dense(n_last_layer, activation='relu')(M_layer)
    mainModel = tf.keras.models.Model(inputs=inputs, outputs=M_layer)
    return mainModel

# Define loss functions
def make_loss_model(c):
    def loss_model(y_true, y_pred):
        #return c * tf.keras.backend.binary_crossentropy(y_pred, y_true)
        return c * tf.keras.backend.binary_crossentropy(y_true, y_pred)
    return loss_model

def make_loss_adversary(c):
    def loss_R(z_true, z_pred):
        #return c * tf.keras.backend.categorical_crossentropy(z_pred, z_true)
        return c * tf.keras.backend.categorical_crossentropy(z_true, z_pred)
    return loss_R
    
# Takes training vars, signal and background files and returns training data
def get_data(allVars, signalDataSet, backgroundDataSet):
    dgSig = dg.DefinedVariables(allVars, signal = True,  background = False)
    dgBg = dg.DefinedVariables(allVars,  signal = False, background = True)
    
    dataSig = dgSig.importData(samplesToRun = tuple(signalDataSet), prescale=True, ptReweight=False)
    dataBg = dgBg.importData(samplesToRun = tuple(backgroundDataSet), prescale=True, ptReweight=False)
    minLen = min(len(dataSig["data"]),len(dataBg["data"]))

    # Put signal and background data together in trainData dictionary
    trainDataArray = [dataSig,dataBg]
    trainData = {}
    for data in trainDataArray:
        for key in data:
            if key in trainData:
                trainData[key] = np.vstack([trainData[key], data[key][:minLen]])
            else:
                trainData[key] = data[key][:minLen]

    # Randomly shuffle the signal and background 
    perms = np.random.permutation(trainData["data"].shape[0])
    for key in trainData:
        trainData[key] = trainData[key][perms]

    # Rescale inputs to have unit variance centered at 0
    from sklearn.preprocessing import StandardScaler
    def scale(data):
        scaler = StandardScaler()
        scaler.fit(data)
        return scaler.transform(data)
    trainData["data"] = scale(trainData["data"])
    dataSig["data"] = scale(dataSig["data"])
    dataBg["data"] = scale(dataBg["data"])
    return trainData, dataSig, dataBg    

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

#def saveModel(sess, outputDirectory = ""):
#    from tensorflow.python.framework import graph_io
#    from tensorflow.python.tools import freeze_graph
#
#    #Save training checkpoint (contains a copy of the model and the weights)
#    os.makedirs(outputDirectory + "models")
#    saver = tf.train.Saver()
#    checkpoint_path = outputDirectory + "models/model.ckpt"
#    save_path = saver.save(sess, checkpoint_path)
#    
#    print("Model checkpoint saved in file: %s" % save_path)
#    
#    input_graph_path = outputDirectory + "tfModel.pb"
#    graph_io.write_graph(sess.graph, "./", input_graph_path)
#    
#    #create frozen version of graph for distribution
#    input_saver_def_path = ""
#    input_binary = False
#    checkpoint_path = outputDirectory + "models/model.ckpt"
#    output_node_names = "y_ph"
#    restore_op_name = "save/restore_all"
#    filename_tensor_name = "save/Const:0"
#    output_graph_path = outputDirectory + "tfModel_frozen.pb"
#    clear_devices = False
#    
#    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
#                              input_binary, checkpoint_path, output_node_names,
#                              restore_op_name, filename_tensor_name,
#                              output_graph_path, clear_devices, "")
#    
#    print("Frozen model (model and weights) saved in file: %s" % output_graph_path)

if __name__ == '__main__':

    # Define vars
    allVars = ["Jet_pt_1", "Jet_pt_2", "Jet_pt_3", "Jet_pt_4", "Jet_pt_5", "Jet_pt_6", "Jet_pt_7",
               "Jet_eta_1","Jet_eta_2","Jet_eta_3","Jet_eta_4","Jet_eta_5","Jet_eta_6", "Jet_eta_7",
               "Jet_phi_1","Jet_phi_2","Jet_phi_3","Jet_phi_4","Jet_phi_5","Jet_phi_6", "Jet_phi_7",
               "Jet_m_1", "Jet_m_2", "Jet_m_3", "Jet_m_4", "Jet_m_5", "Jet_m_6", "Jet_m_7"]
    
    # Import data
    sgTrainSet = glob("EventShapeTrainingData_V3/trainingTuple_*_division_0_rpv_stop_*_training_0.h5")
    #sgTrainSet = glob("EventShapeTrainingData_V3/trainingTuple_*_division_0_rpv_stop_850_training_0.h5")
    bgTrainSet = glob("EventShapeTrainingData_V3/trainingTuple_*_division_0_TT_training_0.h5")

    sgTestSet = glob("EventShapeTrainingData_V3/trainingTuple_*_division_2_rpv_stop_*_test_0.h5")
    #sgTestSet = glob("EventShapeTrainingData_V3/trainingTuple_*_division_2_rpv_stop_850_test_0.h5")
    bgTestSet = glob("EventShapeTrainingData_V3/trainingTuple_*_division_2_TT_test_0.h5")

    trainData, trainSg, trainBg = get_data(allVars, sgTrainSet, bgTrainSet)
    testData, testSg, testBg = get_data(allVars, sgTestSet, bgTestSet)

    ## Make and train model
    #model = create_main_model(n_var=trainData["data"].shape[1], n_first_layer=70, n_hidden_layers=[70], n_last_layer=1, drop_out=0.5)
    #adagrad = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    #model.compile(loss=make_loss_model(c=1.0), optimizer=adagrad, metrics=['accuracy'])
    #os.makedirs("TEST")
    #log_model = tf.keras.callbacks.ModelCheckpoint('TEST/BestNN.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    #result_log = model.fit(trainData["data"], trainData["labels"][:,0], batch_size=2048, epochs=100, validation_data=(testData["data"], testData["labels"][:,0]), callbacks=[log_model])
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------------
    lam = 0
    nNodes = 70
    n_hidden_layers = list(nNodes for x in range(1))
    drop_out = 0.5
    
    main_input = tf.keras.layers.Input(shape=(trainData["data"].shape[1],), name='main_input')
    layer = tf.keras.layers.Dense(nNodes, activation='relu')(main_input)
    for n in n_hidden_layers:
        layer = tf.keras.layers.Dense(n, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(drop_out)(layer)
    first_output = tf.keras.layers.Dense(1, activation='relu', name='first_output')(layer)
    
    layer = tf.keras.layers.Dense(nNodes, activation='relu')(first_output)
    for n in n_hidden_layers:
        layer = tf.keras.layers.Dense(n, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(drop_out)(layer)
    second_output = tf.keras.layers.Dense(1, activation='relu', name='second_output')(layer)
    
    model = tf.keras.models.Model(inputs=main_input, outputs=[first_output, second_output])
    adagrad = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    model.compile(loss={'first_output' : make_loss_model(c=1.0) , 'second_output' : make_loss_model(c=-lam)}, optimizer=adagrad, metrics=['accuracy'])
    os.makedirs("TEST")
    log_model = tf.keras.callbacks.ModelCheckpoint('TEST/BestNN.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    result_log = model.fit({'main_input' : trainData["data"]}, {'first_output' : trainData["labels"][:,0], 'second_output' : trainData["domain"][:,0]}, batch_size=2048, epochs=100,
                           validation_data=({'main_input' : testData["data"]}, {'first_output' : testData["labels"][:,0], 'second_output' : testData["domain"][:,0]}), callbacks=[log_model])
    #-------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Save trainig model as a protocol buffers file
    frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "TEST/", "tfModel_frozen.pb", as_text=False)
    #sess = tf.keras.backend.get_session()
    #saveModel(sess, "TEST/")
    
    # Plot results
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    #sgValSet = glob("EventShapeTrainingData_V3/trainingTuple_*_division_1_rpv_stop_*_validation_0.h5")
    sgValSet = glob("EventShapeTrainingData_V3/trainingTuple_*_division_1_rpv_stop_850_validation_0.h5")
    bgValSet = glob("EventShapeTrainingData_V3/trainingTuple_*_division_1_TT_validation_0.h5")
    valData, valSg, valBg = get_data(allVars, sgValSet, bgValSet)
    #y_Val = model.predict(valData["data"]).ravel()
    #y_Val_Sg = model.predict(valSg["data"]).ravel()
    #y_Val_Bg = model.predict(valBg["data"]).ravel()    
    #y_Train = model.predict(trainData["data"]).ravel()
    #y_Train_Sg = model.predict(trainSg["data"]).ravel()
    #y_Train_Bg = model.predict(trainBg["data"]).ravel()

    #-------------------------------------------------------------------------------------------------------------------------------------------------------
    y_Val = model.predict(valData["data"])[0].ravel()
    y_Val_Sg = model.predict(valSg["data"])[0].ravel()
    y_Val_Bg = model.predict(valBg["data"])[0].ravel()    
    y_Train = model.predict(trainData["data"])[0].ravel()
    y_Train_Sg = model.predict(trainSg["data"])[0].ravel()
    y_Train_Bg = model.predict(trainBg["data"])[0].ravel()
    #-------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Plot loss of training vs test
    #print(result_log.history.keys())
    plt.plot(result_log.history['loss'])
    plt.plot(result_log.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # Plot discriminator distribution
    bins = np.linspace(0, 1, 30)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('')
    ax.set_ylabel('Events')
    ax.set_xlabel('Discriminator')
    plt.hist(y_Train_Sg, bins, color='xkcd:red', alpha=0.9, histtype='step', lw=2, label='Sg Train', normed=1)
    plt.hist(y_Val_Sg, bins, color='xkcd:green', alpha=0.9, histtype='step', lw=2, label='Sg Val', normed=1)
    plt.hist(y_Train_Bg, bins, color='xkcd:blue', alpha=0.9, histtype='step', lw=2, label='Bg Train', normed=1)
    plt.hist(y_Val_Bg, bins, color='xkcd:magenta', alpha=0.9, histtype='step', lw=2, label='Bg Val', normed=1)
    ax.legend(loc='upper right', frameon=False)
    plt.show()
    
    # Plot validation roc curve
    fpr_Val, tpr_Val, thresholds_Val = roc_curve(valData["labels"][:,0], y_Val)
    fpr_Train, tpr_Train, thresholds_Train = roc_curve(trainData["labels"][:,0], y_Train)
    from sklearn.metrics import auc
    auc_Val = auc(fpr_Val, tpr_Val)
    auc_Train = auc(fpr_Train, tpr_Train)
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_Val, tpr_Val, color='xkcd:black', label='Val (area = {:.3f})'.format(auc_Val))
    plt.plot(fpr_Train, tpr_Train, color='xkcd:red', label='Train (area = {:.3f})'.format(auc_Train))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    
    # Plot NJet dependance
    x = trainBg["nJet"][:,0]
    y = y_Train_Bg
    df = pd.DataFrame({'x': x, 'y': y})
    binxl = 7
    binxh = 11
    numbin = binxh - binxl
    
    from matplotlib.colors import LogNorm
    plt.hist2d(trainBg["nJet"][:,0], y_Train_Bg, bins=[numbin, 30], range=[[binxl, binxh], [0, 1]], norm=LogNorm())
    plt.colorbar()
    plt.show()

    bins = np.linspace(binxl, binxh, numbin)
    df['bin'] = np.digitize(x, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]
    binned = df.groupby('bin')
    result = binned['y'].agg(['mean', 'sem'])
    result['x'] = bin_centers
    result['xerr'] = bin_width / 2
    result.plot(x='x', y='mean', xerr='xerr', yerr='sem', linestyle='none', capsize=0, color='black', ylim=(0,1))
    plt.show()
