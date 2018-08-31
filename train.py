import os
import tensorflow as tf
import numpy
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

# Makes the adversary DNN for a given DNN (still need to specify loss function)
def create_adversary_model(mainModel, main_var, n_hidden_layers, n_last_layer, drop_out):
    inputs = tf.keras.layers.Input(shape=(main_var,))
    A_layer = mainModel(inputs)
    for n in n_hidden_layers:        
        A_layer = tf.keras.layers.Dense(n, activation='relu')(A_layer)
    A_layer = tf.keras.layers.Dropout(drop_out)(A_layer)
    A_layer = tf.keras.layers.Dense(n_last_layer, activation='relu')(A_layer)
    adversaryModel = tf.keras.models.Model(inputs=inputs, outputs=A_layer)
    return adversaryModel

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
                trainData[key] = numpy.vstack([trainData[key], data[key][:minLen]])
            else:
                trainData[key] = data[key][:minLen]

    # Randomly shuffle the signal and background 
    perms = numpy.random.permutation(trainData["data"].shape[0])
    for key in trainData:
        trainData[key] = trainData[key][perms]

    # Rescale inputs to have unit variance centered at 0
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(trainData["data"])
    trainData["data"] = scaler.transform(trainData["data"])

    return trainData    

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
    allVars = ["Jet_pt_1", "Jet_pt_2", "Jet_pt_3", "Jet_pt_4", "Jet_pt_5", "Jet_pt_6",
               "Jet_eta_1","Jet_eta_2","Jet_eta_3","Jet_eta_4","Jet_eta_5","Jet_eta_6",
               "Jet_phi_1","Jet_phi_2","Jet_phi_3","Jet_phi_4","Jet_phi_5","Jet_phi_6",
               "Jet_m_1", "Jet_m_2", "Jet_m_3", "Jet_m_4", "Jet_m_5", "Jet_m_6"]

    # Import data
    sgTrainSet = glob("EventShapeTrainingData_V2/trainingTuple_*_division_0_rpv_stop_*_training_0.h5")
    bgTrainSet = glob("EventShapeTrainingData_V2/trainingTuple_*_division_0_TT_training_0.h5")
    sgTestSet = glob("EventShapeTrainingData_V2/trainingTuple_*_division_2_rpv_stop_*_test_0.h5")
    bgTestSet = glob("EventShapeTrainingData_V2/trainingTuple_*_division_2_TT_test_0.h5")
    trainData = get_data(allVars, sgTrainSet, bgTrainSet)
    testData = get_data(allVars, sgTestSet, bgTestSet)

    # Make and train model
    model = create_main_model(n_var=trainData["data"].shape[1], n_first_layer=60, n_hidden_layers=[60,60,60], n_last_layer=1, drop_out=0.4)
    adversay = create_adversary_model(mainModel=model, main_var=trainData["data"].shape[1], n_hidden_layers=[60,60,60], n_last_layer=1, drop_out=0.4)
    adagrad = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adagrad)
    os.makedirs("TEST")
    log_model = tf.keras.callbacks.ModelCheckpoint('TEST/BestNN.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    result_log = model.fit(trainData["data"], trainData["labels"][:,0], batch_size=2048, epochs=50, validation_data=(testData["data"], testData["labels"][:,0]), callbacks=[log_model])
    
    # Save trainig model as a protocol buffers file
    frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "TEST/", "tfModel_frozen.pb", as_text=False)
    #sess = tf.keras.backend.get_session()
    #saveModel(sess, "TEST/")
    
    # Plot results
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    
    # Plot loss of training vs test
    #print(result_log.history.keys())
    plt.plot(result_log.history['loss'])
    plt.plot(result_log.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # Plot validation roc curve
    sgValSet = glob("EventShapeTrainingData_V2/trainingTuple_*_division_1_rpv_stop_*_validation_0.h5")
    bgValSet = glob("EventShapeTrainingData_V2/trainingTuple_*_division_1_TT_validation_0.h5")
    valData = get_data(allVars, sgValSet, bgValSet)
    y_pred_keras = model.predict(valData["data"]).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(valData["labels"][:,0], y_pred_keras)
    
    from sklearn.metrics import auc
    auc_keras = auc(fpr_keras, tpr_keras)
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()    
