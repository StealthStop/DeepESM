import tensorflow as tf

#@tf.custom_gradient
#def GradientReversalOperator(x, hp_lambda):
#        def grad(dy):
#                return -hp_lambda * dy
#        return x, grad
#
#class GradientReversal(tf.keras.layers.Layer):
#        def __init__(self, hp_lambda=1.0):
#                super(GradientReversal, self).__init__()
#                self.hp_lambda = hp_lambda
#
#        def call(self, x):
#                return GradientReversalOperator(x, self.hp_lambda)


@tf.custom_gradient
def GradientReversalOperator(x):
        def grad(dy):
                return -1 * dy
        return x, grad

class GradientReversal(tf.keras.layers.Layer):
        def __init__(self):
                super(GradientReversal, self).__init__()
                self.hp_lambda = 1.0

        def setLambda(self, hp_lambda):
                self.hp_lambda = hp_lambda

        def call(self, inputs):
                return GradientReversalOperator(inputs)






