import tensorflow.keras as K
from keras.engine.topology import Layer

class FreeParamLayer(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self._freeParam = self.add_weight(name="freeParam",
                                  shape=(1,),
                                  initializer = K.initializers.RandomUniform(minval=0.0, maxval=1.0),
                                  trainable=True)

        super().build(input_shape)

    def call(self, y):
        return y
