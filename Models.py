import tensorflow.keras as K
from CustomOptimizer import CustomAdam

def main_model(config, scales, means, regShape, discoShape, inputShape):
    theOptimizer = CustomAdam(mass_reg_lr=config["mass_reg_lr"], disc_lr=config["disc_lr"], other_lr=config["other_lr"],
                             mass_reg_beta_1=0.9, mass_reg_beta_2=0.999, disc_beta_1=0.9, disc_beta_2=0.999, 
                             other_beta_1=0.9, other_beta_2=0.999, epsilon=1e-8, amsgrad=True)

    n_hidden_layers_in = list(config["input_nodes"] for x in range(config["input_layers"]))
    n_hidden_layers = list(config["disc_nodes"] for x in range(config["disc_layers"]))
    n_hidden_layers_M = list(config["mass_reg_nodes"] for x in range(config["mass_reg_layers"]))

    main_input = K.layers.Input(shape=(inputShape,), name='main_input')
    # Set the rescale inputs to have unit variance centered at 0 between -1 and 1
    norm_layer = K.layers.Lambda(lambda x: (x - K.backend.constant(means)) * K.backend.constant(scales), name='normalizeData')(main_input)
    iD = 0
    layer = norm_layer
    #for n in n_hidden_layers_in:
    #    layer = K.layers.Dense(n, activation='relu', name="disc_%d"%(iD))(layer)
    #    iD += 1
    #layerSplit = K.layers.Dense(config["input_nodes"], activation='relu', name="disc_%d"%(iD))(layer)

    #iD += 1
    #layer = K.layers.BatchNormalization(name="disc_%d"%(iD))(main_input)        
    '''
    layer = norm_layer
    iM = 0
    for n in n_hidden_layers_M:
        layer = K.layers.Dense(n, activation='relu', name="mass_reg_%d"%(iM))(layer)
        iM += 1
    layer = K.layers.Dropout(config["dropout"],seed=config["seed"])(layer)
    mass_reg = K.layers.Dense(regShape, activation=None, name='mass_reg')(layer)
    

    layerSplit = K.layers.concatenate([norm_layer, mass_reg], name='concat_mass_layer')
           
    layer = layerSplit
    '''
    iPre = 0
    for n in n_hidden_layers_in:
        layer = K.layers.Dense(n, activation='relu',name="pre_disc_%d"%(iPre))(layer)
        iPre += 1
    layerPre = layer

    iD += 1
    layer = K.layers.BatchNormalization(name="disc_%d"%(iD))(layerPre)
    for n in n_hidden_layers:
        iD += 1
        layer = K.layers.Dense(n, activation='relu', name="disc_%d"%(iD))(layer)
    layer = K.layers.Dropout(config["dropout"],seed=config["seed"])(layer)

    iD += 1
    first_output = K.layers.Dense(discoShape, activation='softmax', name='disc_%d'%(iD))(layer)

    iD += 1
    layer = K.layers.BatchNormalization(name="disc_%d"%(iD))(layerPre)        
    for n in n_hidden_layers:
        iD += 1
        layer = K.layers.Dense(n, activation='relu', name="disc_%d"%(iD))(layer)
    layer = K.layers.Dropout(config["dropout"],seed=config["seed"])(layer)

    iD += 1
    second_output = K.layers.Dense(discoShape, activation='softmax', name='disc_%d'%(iD))(layer)

    disc = K.layers.concatenate([first_output, second_output], name='disc')
    disco = K.layers.concatenate([first_output, second_output], name='disco')
    closure = K.layers.concatenate([first_output, second_output], name='closure')

    #model = K.models.Model(inputs=main_input, outputs=[disc, disco, mass_reg], name='model')
    model = K.models.Model(inputs=main_input, outputs=[disc, disco, closure], name='model')
    return model, theOptimizer
