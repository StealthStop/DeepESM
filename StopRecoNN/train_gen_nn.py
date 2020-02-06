import tensorflow
from keras import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import uproot
import math
from matplotlib import pyplot as plt

#class for making the input/output variable numpy arrays using uproot. jets are zero padded based on the maximum number of jets in all events

class input_output_var:
    def __init__(self,path,sample):
            file = uproot.open(path+sample+'.root')
    
            events = file['myMiniTree;1']

            nJets                              = events.array("NGoodJets_pt30")
            nBJets                             = events.array("NGoodBJets_pt30")
            FirstStopMass                      = events.array("FirstStopMassSums")
            SecStopMass                        = events.array("SecStopMassSums")
            StopMassDiff                       = events.array("StopMassDiffs")
            Mbl1                               = events.array("TwoLep_Mbl1")
            Mbl2                               = events.array("TwoLep_Mbl2")
            StopMT2                            = events.array("GM_StopMT2")
            Stop1                              = events.array("GM_Stop1")
            Stop1Mass                          = Stop1[:].mass
            Stop2                              = events.array("GM_Stop2")
            Stop2Mass                          = Stop2[:].mass
            lepton_mass                        = self.zero_pad(events.array("GoodLeptonsMass"), 2)
            lepton_pt                          = self.zero_pad(events.array("GoodLeptonsPt"), 2)
            lepton_phi                         = self.zero_pad(events.array("GoodLeptonsPhi"), 2)
            lepton_eta                         = self.zero_pad(events.array("GoodLeptonsEta"), 2)
            jet_mass                           = self.zero_pad(events.array("GoodJetsMass"), 15)
            jet_pt                             = self.zero_pad(events.array("GoodJetsPt"), 15)
            jet_eta                            = self.zero_pad(events.array("GoodJetsEta"), 15)
            jet_phi                            = self.zero_pad(events.array("GoodJetsPhi"), 15)
            

            prelim_sig_input = np.transpose(np.array([nJets, nBJets, Mbl1, Mbl2, StopMT2]))
            self.input = np.concatenate([prelim_sig_input,lepton_mass, lepton_pt, lepton_eta, lepton_phi, jet_mass, jet_pt, jet_eta, jet_phi], axis=1)
            self.output  = np.transpose(np.array([Stop1Mass, Stop2Mass]))
            
#this will zeropad any jagged array given by uproot, or convert it to a regular numpy array
    def zero_pad(self, jagged_array, nparticle):
        np_array  = np.zeros((np.size(jagged_array,0), nparticle))
        for event in range (0,np.size(jagged_array,0)):
            for part in range(0,len(jagged_array[event])):
                np_array[event,part] = jagged_array[event,part]
                return np_array





RPV_signal_2016 = ['2016_RPV_2t6j_mStop-300','2016_RPV_2t6j_mStop-350','2016_RPV_2t6j_mStop-400','2016_RPV_2t6j_mStop-450','2016_RPV_2t6j_mStop-500','2016_RPV_2t6j_mStop-550','2016_RPV_2t6j_mStop-600','2016_RPV_2t6j_mStop-650','2016_RPV_2t6j_mStop-700','2016_RPV_2t6j_mStop-750','2016_RPV_2t6j_mStop-800','2016_RPV_2t6j_mStop-900']

stealth_signal_2016=['2016_StealthSYY_2t6j_mStop-300','2016_StealthSYY_2t6j_mStop-350','2016_StealthSYY_2t6j_mStop-400','2016_StealthSYY_2t6j_mStop-450','2016_StealthSYY_2t6j_mStop-500','2016_StealthSYY_2t6j_mStop-550','2016_StealthSYY_2t6j_mStop-600','2016_StealthSYY_2t6j_mStop-650','2016_StealthSYY_2t6j_mStop-700','2016_StealthSYY_2t6j_mStop-750','2016_StealthSYY_2t6j_mStop-800','2016_StealthSYY_2t6j_mStop-850','2016_StealthSYY_2t6j_mStop-900']

RPV_signal_2017 = ['2017_RPV_2t6j_mStop-300','2017_RPV_2t6j_mStop-350','2017_RPV_2t6j_mStop-400','2017_RPV_2t6j_mStop-450','2017_RPV_2t6j_mStop-500','2017_RPV_2t6j_mStop-550','2017_RPV_2t6j_mStop-600','2017_RPV_2t6j_mStop-650','2017_RPV_2t6j_mStop-700','2017_RPV_2t6j_mStop-750','2017_RPV_2t6j_mStop-800','2017_RPV_2t6j_mStop-900']

signal_path_2016 = '~/nobackup/SUSY3/CMSSW_10_2_9/src/Analyzer/Analyzer/test/condor/MC_Training_Input_2016_v3/'
signal_path_2017 = '~/nobackup/SUSY3/CMSSW_10_2_9/src/Analyzer/Analyzer/test/condor/MC_Training_Input_2017_v3/'

print("<-------Setting up input variables------->")
for sig in range(0,len(RPV_signal_2016)):

    IO_vars_2016 = input_output_var(signal_path_2016, RPV_signal_2016[sig])
    IO_vars_2017 = input_output_var(signal_path_2017, RPV_signal_2017[sig])
    sig_input = np.concatenate([IO_vars_2016.input,IO_vars_2017.input])
    sig_output = np.concatenate([IO_vars_2016.output,IO_vars_2017.output])
    if sig == 0:
        input_variables = sig_input
        output_variables = sig_output
    else:
        input_variables = np.concatenate([input_variables,sig_input])
        output_variables = np.concatenate([output_variables,sig_output])


background = input_output_var(signal_path_2016, '2016_TT')
bg_input = background.input
bg_output = background.output


#there are some NaNs in the output, this removes it but need to find source of issue
check = np.argwhere(np.isnan(output_variables))
add = 0
for x in check[:,0]:
    output_variables = np.delete(output_variables, x-add, 0)
    input_variables = np.delete(input_variables, x-add, 0)
    add += 1 


assert not np.any(np.isnan(input_variables))
assert not np.any(np.isnan(output_variables))


in_train_val, in_test, out_train_val, out_test = train_test_split(input_variables, output_variables, test_size=0.3, random_state=7)


print("Done setting up input variables. Will now begin training")

inputs = Input(shape=(np.size(in_train_val,1),), name = 'input')
outputs = Dense(2, name = 'output', kernel_initializer='normal', activation = 'relu')(inputs)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='mse',optimizer='rmsprop')

callbacks = [ EarlyStopping(verbose = True, patience = 20, monitor = 'val_loss'),
              ModelCheckpoint('test_model.h5',monitor='val_loss',verbose=True,save_best_only=True)]

history = model.fit(in_train_val,out_train_val,epochs=200,validation_split=0.2,callbacks=callbacks)
#print(history.history)
#plt.plot(history.history['val_loss'], label='val_loss')
#plt.plot(history.history['loss'], label = 'loss')
#plt.legend()
#plt.show()
pred_mass = np.transpose( model.predict(in_test))
actual_mass = np.transpose(out_test)


background_pred = np.transpose( model.predict(bg_input))


diff_mass =  np.absolute(np.subtract(pred_mass, actual_mass))
fig = plt.figure()
#plt.hist(pred_mass[0],histtype='step')


plt.hist((pred_mass[0],actual_mass[0],background_pred[0]),label=('Predicted','Actual','Background'),histtype='step',range=(0,1500),bins=300)
#plt.hist(diff_mass[0],histtype='step')
plt.xlabel('Stop Mass [GeV]')
plt.ylabel('Fraction of Events')
plt.legend()
plt.savefig('sig_bg_comp_RPV.png')



model.save('test_model.h5')
