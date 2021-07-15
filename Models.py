import tensorflow.keras as K
from CustomOptimizer import CustomAdam

def main_model(config, scales, means, regShape, discoShape, inputShape):
   theOptimizer = CustomAdam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=True, config=config)

   n_hidden_layers = list(config["disc_nodes"] for x in range(config["disc_layers"]))
   n_hidden_layers_M = list(config["mass_reg_nodes"] for x in range(config["mass_reg_layers"]))

   main_input = K.layers.Input(shape=(inputShape,), name='main_input')
   # Set the rescale inputs to have unit variance centered at 0 between -1 and 1
   layer = K.layers.Lambda(lambda x: (x - K.backend.constant(means)) * K.backend.constant(scales), name='normalizeData')(main_input)
   iD = 0
   for n in n_hidden_layers:
       layer = K.layers.Dense(n, activation='relu', name="disc_%d"%(iD))(layer)
       iD += 1
   layerSplit = K.layers.Dense(config["disc_nodes"], activation='relu', name="disc_%d"%(iD))(layer)

   iD += 1
   layer = K.layers.BatchNormalization(name="disc_%d"%(iD))(layerSplit)        
   iM = 0
   for n in n_hidden_layers_M:
       layer = K.layers.Dense(n, activation='relu', name="mass_reg_%d"%(iM))(layer)
       iM += 1
   layer = K.layers.Dropout(config["dropout"],seed=config["seed"])(layer)
   mass_reg = K.layers.Dense(regShape, activation=None, name='mass_reg')(layer)

   layerSplit = K.layers.concatenate([layerSplit, mass_reg], name='concat_mass_layer')
       
   iD += 1
   layer = K.layers.BatchNormalization(name="disc_%d"%(iD))(layerSplit)
   for n in n_hidden_layers:
       iD += 1
       layer = K.layers.Dense(n, activation='relu', name="disc_%d"%(iD))(layer)
   layer = K.layers.Dropout(config["dropout"],seed=config["seed"])(layer)

   iD += 1
   first_output = K.layers.Dense(discoShape, activation='softmax', name='disc_%d'%(iD))(layer)

   layer = K.layers.BatchNormalization()(layerSplit)        
   for n in n_hidden_layers:
       iD += 1
       layer = K.layers.Dense(n, activation='relu', name="disc_%d"%(iD))(layer)
   layer = K.layers.Dropout(config["dropout"],seed=config["seed"])(layer)

   iD += 1
   second_output = K.layers.Dense(discoShape, activation='softmax', name='disc_%d'%(iD))(layer)

   disc = K.layers.concatenate([first_output, second_output], name='disc')
   disco = K.layers.concatenate([first_output, second_output], name='disco')

   model = K.models.Model(inputs=main_input, outputs=[disc, disco, mass_reg], name='model')
   return model, theOptimizer
