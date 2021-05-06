import tensorflow.keras as K
from flipGradientTF2 import GradientReversal 

def main_model(config, trainData, trainDataTT):
    optimizer = K.optimizers.Adam(lr=config["lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    n_hidden_layers = list(config["nNodes"] for x in range(config["nHLayers"]))
    n_hidden_layers_M = list(config["nNodesM"] for x in range(config["nHLayersM"]))
    n_hidden_layers_D = list(config["nNodesD"] for x in range(config["nHLayersD"]))

    main_input = K.layers.Input(shape=(trainData["data"].shape[1],), name='main_input')
    # Set the rescale inputs to have unit variance centered at 0 between -1 and 1
    layer = K.layers.Lambda(lambda x: (x - K.backend.constant(trainDataTT["mean"])) * K.backend.constant(trainDataTT["scale"]), name='normalizeData')(main_input)
    for n in n_hidden_layers:
        layer = K.layers.Dense(n, activation='relu')(layer)
    layerSplit = K.layers.Dense(config["nNodes"], activation='relu')(layer)

    layer = K.layers.BatchNormalization()(layerSplit)        
    for n in n_hidden_layers_M:
        layer = K.layers.Dense(n, activation='relu')(layer)
    layer = K.layers.Dropout(config["drop_out"],seed=config["seed"])(layer)
    fourth_output = K.layers.Dense(trainData["massesReco"].shape[1], activation=None, name='fourth_output')(layer)

    layerSplit = K.layers.concatenate([layerSplit, fourth_output], name='concat_mass_layer')
        
    layer = K.layers.BatchNormalization()(layerSplit)
    for n in n_hidden_layers:
        layer = K.layers.Dense(n, activation='relu')(layer)
    layer = K.layers.Dropout(config["drop_out"],seed=config["seed"])(layer)
    first_output = K.layers.Dense(trainData["labels"].shape[1], activation='softmax', name='first_output')(layer)

    layer = K.layers.BatchNormalization()(layerSplit)        
    for n in n_hidden_layers:
        layer = K.layers.Dense(n, activation='relu')(layer)
    layer = K.layers.Dropout(config["drop_out"],seed=config["seed"])(layer)
    second_output = K.layers.Dense(trainData["labels"].shape[1], activation='softmax', name='second_output')(layer)

    disc_comb = K.layers.concatenate([first_output, second_output], name='disc_comb_layer')

    corr = K.layers.concatenate([first_output, second_output], name='correlation_layer')

    layer = GradientReversal()(corr)
    layer = K.layers.BatchNormalization()(layer)
    for n in n_hidden_layers_D:
        layer = K.layers.Dense(n, activation='relu')(layer)
    layer = K.layers.Dropout(config["drop_out"],seed=config["seed"])(layer)
    third_output = K.layers.Dense(trainData["domain"].shape[1], activation='softmax', name='third_output')(layer)
            
    model = K.models.Model(inputs=main_input, outputs=[disc_comb, third_output, corr, fourth_output], name='model')
    return model, optimizer

def model_doubleDisco(config, trainData, trainDataTT):
    optimizer = K.optimizers.Adam(lr=config["lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    n_hidden_layers = list(config["nNodes"] for x in range(config["nHLayers"]))
    n_hidden_layers_D = list(config["nNodesD"] for x in range(config["nHLayersD"]))

    main_input = K.layers.Input(shape=(trainData["data"].shape[1],), name='main_input')
    # Set the rescale inputs to have unit variance centered at 0 between -1 and 1
    layer = K.layers.Lambda(lambda x: (x - K.backend.constant(trainDataTT["mean"])) * K.backend.constant(trainDataTT["scale"]), name='normalizeData')(main_input)
    for n in n_hidden_layers:
        layer = K.layers.Dense(n, activation='relu')(layer)
    layerSplit = K.layers.Dense(config["nNodes"], activation='relu')(layer)
            
    layer = K.layers.BatchNormalization()(layerSplit)
    for n in n_hidden_layers:
        layer = K.layers.Dense(n, activation='relu')(layer)
    layer = K.layers.Dropout(config["drop_out"],seed=config["seed"])(layer)
    first_output = K.layers.Dense(trainData["labels"].shape[1], activation='softmax', name='first_output')(layer)

    layer = K.layers.BatchNormalization()(layerSplit)        
    for n in n_hidden_layers:
        layer = K.layers.Dense(n, activation='relu')(layer)
    layer = K.layers.Dropout(config["drop_out"],seed=config["seed"])(layer)
    second_output = K.layers.Dense(trainData["labels"].shape[1], activation='softmax', name='second_output')(layer)

    corr = K.layers.concatenate([first_output, second_output], name='correlation_layer')
    layer = GradientReversal()(corr)
    layer = K.layers.BatchNormalization()(layer)
    for n in n_hidden_layers_D:
        layer = K.layers.Dense(n, activation='relu')(layer)
    layer = K.layers.Dropout(config["drop_out"],seed=config["seed"])(layer)
    third_output = K.layers.Dense(trainData["domain"].shape[1], activation='softmax', name='third_output')(layer)
            
    model = K.models.Model(inputs=main_input, outputs=[first_output, second_output, third_output, corr], name='model')
    #model.summary()
    return model, optimizer

def model_reg(config, trainData, trainDataTT):
    optimizer = K.optimizers.Adam(lr=config["lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    n_hidden_layers_M = list(config["nNodesM"] for x in range(config["nHLayersM"]))

    main_input = K.layers.Input(shape=(trainData["data"].shape[1],), name='main_input')
    # Set the rescale inputs to have unit variance centered at 0 between -1 and 1
    layer = K.layers.Lambda(lambda x: (x - K.backend.constant(trainDataTT["mean"])) * K.backend.constant(trainDataTT["scale"]), name='normalizeData')(main_input)
    for n in n_hidden_layers_M:
        layer = K.layers.Dense(n, activation='relu')(layer)
    layer = K.layers.Dropout(config["drop_out"],seed=config["seed"])(layer)
    first_output = K.layers.Dense(trainData["masses"].shape[1], activation=None, name='first_output')(layer)

    model = K.models.Model(inputs=main_input, outputs=first_output, name='model')
    return model, optimizer

