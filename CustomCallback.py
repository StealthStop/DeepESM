import tensorflow as tf
from tensorflow import keras as K

class CustomCallback(K.callbacks.Callback):
    '''
    Class to handle the callback for producing random bin edges and getting event counts in the
    A, B, C, and D regions

    Inherets from Keras callbacks class and overrides functions called on training/testing/validating begin.
    Allows us to use the same bin edges for training and validation and runs multiple bin edge calculations
    for averaging purposes.
    '''    

    def __init__(self, current_epoch=None, patience=100):

        super(CustomCallback, self).__init__()
        self.patience = patience

        self.best = 9999.0 
        self.wait = 0

        # Same generator used for all other functions with the same random seed (for repeatability of training)
        self.current_epoch = current_epoch

    def on_epoch_end(self, epoch, logs):
        #Monitor epoch value and update for loss functions
        K.backend.set_value(self.current_epoch, epoch+1)

        #Early stopping check using disc loss
        current = logs.get("disc_loss")

        if tf.less(epoch, 10):
            pass
        elif tf.less(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if tf.greater(self.wait, self.patience-1):
                self.model.stop_training = True
                tf.print("Early stop at epoch {}".format(self.current_epoch))


