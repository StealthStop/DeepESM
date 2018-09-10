import os
import tensorflow as tf
import numpy as np
import pandas as pd
from DataGetter import DataGetter as dg
from flipGradientTF import GradientReversal
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
        return c * tf.keras.backend.binary_crossentropy(y_true, y_pred)
    return loss_model

def make_loss_adversary(c):
    def loss_adversary(y_true, y_pred):
        #return c * tf.keras.backend.categorical_crossentropy(y_true, y_pred)
        return c * tf.keras.backend.binary_crossentropy(y_true, y_pred)
    return loss_adversary
    
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

    # Get the rescale inputs to have unit variance centered at 0 between -1 and 1
    def scale(data):
        data["mean"] = np.mean(data["data"], 0)
        data["std"] = np.std(data["data"], 0)
        data["scale"] = 1.0 / np.std(data["data"], 0)
    scale(trainData)
    scale(dataSig)
    scale(dataBg)
    return trainData, dataSig, dataBg    

if __name__ == '__main__':
    
    # Define vars
    allVars = ["Jet_pt_1", "Jet_pt_2", "Jet_pt_3", "Jet_pt_4", "Jet_pt_5", "Jet_pt_6", "Jet_pt_7",
               "Jet_eta_1","Jet_eta_2","Jet_eta_3","Jet_eta_4","Jet_eta_5","Jet_eta_6", "Jet_eta_7",
               "Jet_phi_1","Jet_phi_2","Jet_phi_3","Jet_phi_4","Jet_phi_5","Jet_phi_6", "Jet_phi_7",
               "Jet_m_1", "Jet_m_2", "Jet_m_3", "Jet_m_4", "Jet_m_5", "Jet_m_6", "Jet_m_7"]
    #allVars = ["Jet_pt_1", "Jet_pt_2", "Jet_pt_3", "Jet_pt_4", "Jet_pt_5", "Jet_pt_6",
    #           "Jet_eta_1","Jet_eta_2","Jet_eta_3","Jet_eta_4","Jet_eta_5","Jet_eta_6",
    #           "Jet_phi_1","Jet_phi_2","Jet_phi_3","Jet_phi_4","Jet_phi_5","Jet_phi_6",
    #           "Jet_m_1", "Jet_m_2", "Jet_m_3", "Jet_m_4", "Jet_m_5", "Jet_m_6"]
    
    # Import data
    print("----------------Preparing data------------------")
    dataSet = "EventShapeTrainingData_V3/"
    massModel = "*"
    print "Using "+dataSet+" data set"
    print "Training variables:"
    print allVars
    print "Training on mass model: ", massModel
    
    sgTrainSet = glob(dataSet+"trainingTuple_*_division_0_rpv_stop_"+massModel+"_training_0.h5")
    bgTrainSet = glob(dataSet+"trainingTuple_*_division_0_TT_training_0.h5")

    sgTestSet = glob(dataSet+"trainingTuple_*_division_2_rpv_stop_"+massModel+"_test_0.h5")
    bgTestSet = glob(dataSet+"trainingTuple_*_division_2_TT_test_0.h5")

    trainData, trainSg, trainBg = get_data(allVars, sgTrainSet, bgTrainSet)
    testData, testSg, testBg = get_data(allVars, sgTestSet, bgTestSet)

    # Make and train model
    #model = create_main_model(n_var=trainData["data"].shape[1], n_first_layer=70, n_hidden_layers=[70, 70, 70], n_last_layer=1, drop_out=0.5)
    #adagrad = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    #model.compile(loss=make_loss_model(c=1.0), optimizer=adagrad, metrics=['accuracy'])
    #os.makedirs("TEST")
    #log_model = tf.keras.callbacks.ModelCheckpoint('TEST/BestNN.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    #result_log = model.fit(trainData["data"], trainData["labels"][:,0], batch_size=2048, epochs=200, validation_data=(testData["data"], testData["labels"][:,0]), callbacks=[log_model])
    
    print("----------------Preparing training model------------------")
    lam = gr_lambda = 0.0
    nNodes = 70
    n_hidden_layers = list(nNodes for x in range(3))
    drop_out = 0.5
    adagrad = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    epochs=100
    
    main_input = tf.keras.layers.Input(shape=(trainData["data"].shape[1],), name='main_input')
    mean = tf.keras.backend.constant(value=trainData["mean"], dtype=np.float32)
    scale = tf.keras.backend.constant(value=trainData["scale"], dtype=np.float32)
    # Get the rescale inputs to have unit variance centered at 0 between -1 and 1
    layer = tf.keras.layers.Lambda(lambda x: (x - mean) * scale, name='normalizeData')(main_input)
    #layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Dense(nNodes, activation='relu')(layer)
    for n in n_hidden_layers:
        layer = tf.keras.layers.Dense(n, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(drop_out)(layer)
    first_output = tf.keras.layers.Dense(1, activation='relu', name='first_output')(layer)
    
    #layer = tf.keras.layers.BatchNormalization()(first_output) #Might not need this
    #Flip = GradientReversal(gr_lambda)
    #layer = Flip(layer)
    layer = tf.keras.layers.Dense(nNodes, activation='relu')(first_output)
    for n in n_hidden_layers:
        layer = tf.keras.layers.Dense(n, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(drop_out)(layer)
    second_output = tf.keras.layers.Dense(1, activation='relu', name='second_output')(layer)
    
    model = tf.keras.models.Model(inputs=main_input, outputs=[first_output, second_output], name='model')
    model.compile(loss=[make_loss_model(c=1.0) , make_loss_adversary(c=-lam)], optimizer=adagrad, metrics=['accuracy'], loss_weights=[1.0, 1.0])
    os.makedirs("TEST/log_graph")
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./TEST/log_graph', histogram_freq=0, write_graph=True, write_images=True)
    log_model = tf.keras.callbacks.ModelCheckpoint('TEST/BestNN.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    result_log = model.fit(trainData["data"], [trainData["labels"][:,0], trainData["domain"][:,0]], batch_size=2048, epochs=epochs,
                           validation_data=(testData["data"], [testData["labels"][:,0], testData["domain"][:,0]]), callbacks=[log_model, tbCallBack])

    # Model Visualization
    tf.keras.utils.plot_model(model, to_file='TEST/model.png', show_shapes=True)
    
    # Save trainig model as a protocol buffers file
    print "Input name:", model.input.op.name.split(':')[0]
    print "Output name:", model.output[0].op.name.split(':')[0]
    saver = tf.train.Saver()
    saver.save(tf.keras.backend.get_session(), 'TEST/keras_model.ckpt')
    export_path="./TEST/"
    freeze_graph_binary = "python ~/Desktop/Research/SUSY/trainingTopTagger/ENV/lib/python2.7/site-packages/tensorflow/python/tools/freeze_graph.py"
    graph_file=export_path+"keras_model.ckpt.meta"
    ckpt_file=export_path+"keras_model.ckpt"
    output_file=export_path+"keras_frozen.pb"
    output_name=model.output[0].name.split(':')[0]
    command = freeze_graph_binary+" --input_meta_graph="+graph_file+" --input_checkpoint="+ckpt_file+" --output_graph="+output_file+" --output_node_names="+output_name+" --input_binary=true"
    os.system(command)

    # Plot results
    print("----------------Validation of training------------------")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    sgValSet = glob(dataSet+"trainingTuple_*_division_1_rpv_stop_"+massModel+"_validation_0.h5")
    bgValSet = glob(dataSet+"trainingTuple_*_division_1_TT_validation_0.h5")
    valData, valSg, valBg = get_data(allVars, sgValSet, bgValSet)
    y_Val = model.predict(valData["data"])[0].ravel()
    y_Val_Sg = model.predict(valSg["data"])[0].ravel()
    y_Val_Bg = model.predict(valBg["data"])[0].ravel()
    y_Train = model.predict(trainData["data"])[0].ravel()
    y_Train_Sg = model.predict(trainSg["data"])[0].ravel()
    y_Train_Bg = model.predict(trainBg["data"])[0].ravel()
    
    # Plot loss of training vs test
    #print(result_log.history.keys())
    fig = plt.figure()
    plt.plot(result_log.history['loss'])
    plt.plot(result_log.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig('TEST/loss_train_val.png', dpi=fig.dpi)
    
    # Plot discriminator distribution
    bins = np.linspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('')
    ax.set_ylabel('Events')
    ax.set_xlabel('Discriminator')
    plt.hist(y_Train_Sg, bins, color='xkcd:red', alpha=0.9, histtype='step', lw=2, label='Sg Train', density=True)
    plt.hist(y_Val_Sg, bins, color='xkcd:green', alpha=0.9, histtype='step', lw=2, label='Sg Val', density=True)
    plt.hist(y_Train_Bg, bins, color='xkcd:blue', alpha=0.9, histtype='step', lw=2, label='Bg Train', density=True)
    plt.hist(y_Val_Bg, bins, color='xkcd:magenta', alpha=0.9, histtype='step', lw=2, label='Bg Val', density=True)
    ax.legend(loc='best', frameon=False)
    plt.show()
    fig.savefig('TEST/discriminator.png', dpi=fig.dpi)
    
    # Plot validation roc curve
    fpr_Val, tpr_Val, thresholds_Val = roc_curve(valData["labels"][:,0], y_Val)
    fpr_Train, tpr_Train, thresholds_Train = roc_curve(trainData["labels"][:,0], y_Train)
    from sklearn.metrics import auc
    auc_Val = auc(fpr_Val, tpr_Val)
    auc_Train = auc(fpr_Train, tpr_Train)
    
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_Val, tpr_Val, color='xkcd:black', label='Val (area = {:.3f})'.format(auc_Val))
    plt.plot(fpr_Train, tpr_Train, color='xkcd:red', label='Train (area = {:.3f})'.format(auc_Train))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    fig.savefig('TEST/roc_plot.png', dpi=fig.dpi)
    
    # Plot NJet dependance
    binxl = 7
    binxh = 12
    numbin = binxh - binxl
    
    from matplotlib.colors import LogNorm
    fig = plt.figure()
    #h, xedges, yedges, image = plt.hist2d(trainBg["nJet"][:,0], y_Train_Bg, bins=[numbin, 50], range=[[binxl, binxh], [0, 1]], norm=LogNorm())
    h, xedges, yedges, image = plt.hist2d(trainBg["nJet"][:,0], y_Train_Bg, bins=[numbin, 50], range=[[binxl, binxh], [0, 1]], cmap=plt.cm.binary)
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
    plt.show()
    fig.savefig('TEST/nJet_discriminator.png', dpi=fig.dpi)
    
    # Make njet distribution for 4 different bins
    inds = y_Train_Bg.argsort()
    sortednJet = trainBg["nJet"][:,0][inds[::-1]]
    sorted_y = y_Train_Bg[inds[::-1]]
    nJetDeepESMBins = np.array_split(sortednJet, 4)
    sorted_y_split = np.array_split(sorted_y, 4)
    index=0
    fig = plt.figure()
    for a in nJetDeepESMBins:
        print "DeepESM bin ", len(nJetDeepESMBins) - index, ": ", " NEvents: ", len(a)," bin cuts: ", sorted_y_split[index][0], " ", sorted_y_split[index][-1]
        plt.hist(a, bins=numbin, range=(binxl, binxh), histtype='step', density=True, log=True, label='Bin {}'.format(len(nJetDeepESMBins) - index))
        index += 1
    plt.legend(loc='upper right')
    plt.show()
    fig.savefig('TEST/nJet.png', dpi=fig.dpi)
