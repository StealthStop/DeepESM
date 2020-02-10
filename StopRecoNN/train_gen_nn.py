import tensorflow
from keras import Sequential, Model, models
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import uproot
import math
from matplotlib import pyplot as plt
from scipy import stats
import ROOT



RPV_signal_2016 = ['2016_RPV_2t6j_mStop-300','2016_RPV_2t6j_mStop-350','2016_RPV_2t6j_mStop-400','2016_RPV_2t6j_mStop-450','2016_RPV_2t6j_mStop-500','2016_RPV_2t6j_mStop-550','2016_RPV_2t6j_mStop-600','2016_RPV_2t6j_mStop-650','2016_RPV_2t6j_mStop-700','2016_RPV_2t6j_mStop-750','2016_RPV_2t6j_mStop-800','2016_RPV_2t6j_mStop-900']

stealth_signal_2016=['2016_StealthSYY_2t6j_mStop-300','2016_StealthSYY_2t6j_mStop-350','2016_StealthSYY_2t6j_mStop-400','2016_StealthSYY_2t6j_mStop-450','2016_StealthSYY_2t6j_mStop-500','2016_StealthSYY_2t6j_mStop-550','2016_StealthSYY_2t6j_mStop-600','2016_StealthSYY_2t6j_mStop-650','2016_StealthSYY_2t6j_mStop-700','2016_StealthSYY_2t6j_mStop-750','2016_StealthSYY_2t6j_mStop-800','2016_StealthSYY_2t6j_mStop-850','2016_StealthSYY_2t6j_mStop-900']

RPV_signal_2017 = ['2017_RPV_2t6j_mStop-300','2017_RPV_2t6j_mStop-350','2017_RPV_2t6j_mStop-400','2017_RPV_2t6j_mStop-450','2017_RPV_2t6j_mStop-500','2017_RPV_2t6j_mStop-550','2017_RPV_2t6j_mStop-600','2017_RPV_2t6j_mStop-650','2017_RPV_2t6j_mStop-700','2017_RPV_2t6j_mStop-750','2017_RPV_2t6j_mStop-800','2017_RPV_2t6j_mStop-900']

stealth_signal_2017 =['2017_StealthSYY_2t6j_mStop-300','2017_StealthSYY_2t6j_mStop-350','2017_StealthSYY_2t6j_mStop-400','2017_StealthSYY_2t6j_mStop-450','2017_StealthSYY_2t6j_mStop-500','2017_StealthSYY_2t6j_mStop-550','2017_StealthSYY_2t6j_mStop-600','2017_StealthSYY_2t6j_mStop-650','2017_StealthSYY_2t6j_mStop-700','2017_StealthSYY_2t6j_mStop-750','2017_StealthSYY_2t6j_mStop-800','2017_StealthSYY_2t6j_mStop-850','2017_StealthSYY_2t6j_mStop-900']

signal_path_2016 = '~/nobackup/SUSY3/CMSSW_10_2_9/src/Analyzer/Analyzer/test/condor/MC_Training_Input_2016_v3/'
signal_path_2017 = '~/nobackup/SUSY3/CMSSW_10_2_9/src/Analyzer/Analyzer/test/condor/MC_Training_Input_2017_v3/'

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


class Train:

    def __init__(self):


        print("<-------Setting up input variables------->")
        for sig in range(0,len(stealth_signal_2016)):

            IO_vars_2016 = input_output_var(signal_path_2016, stealth_signal_2016[sig])
            IO_vars_2017 = input_output_var(signal_path_2017, stealth_signal_2017[sig])
            sig_input = np.concatenate([IO_vars_2016.input,IO_vars_2017.input])
            sig_output = np.concatenate([IO_vars_2016.output,IO_vars_2017.output])
            if sig == 0:
                input_variables = sig_input
                output_variables = sig_output
            else:
                input_variables = np.concatenate([input_variables,sig_input])
                output_variables = np.concatenate([output_variables,sig_output])
            
            
        background = input_output_var(signal_path_2016, '2016_TT')
        self.bg_input = background.input
        self.bg_output = background.output


#there are some NaNs in the output, this removes it but need to find source of issue
        check = np.argwhere(np.isnan(output_variables))
        add = 0
        for x in check[:,0]:
            output_variables = np.delete(output_variables, x-add, 0)
            input_variables = np.delete(input_variables, x-add, 0)
            add += 1 


        assert not np.any(np.isnan(input_variables))
        assert not np.any(np.isnan(output_variables))


        self.in_train_val, self.in_test, self.out_train_val, self.out_test = train_test_split(input_variables, output_variables, test_size=0.3, random_state=7)


        print("Done setting up input variables. Will now begin training")

        inputs = Input(shape=(np.size(self.in_train_val,1),), name = 'input')
        outputs = Dense(2, name = 'output', kernel_initializer='normal', activation = 'relu')(inputs)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse',optimizer='rmsprop')

        callbacks = [ EarlyStopping(verbose = True, patience = 20, monitor = 'val_loss'),
              ModelCheckpoint('test_model.h5',monitor='val_loss',verbose=True,save_best_only=True)]

        self.history = self.model.fit(self.in_train_val,self.out_train_val,epochs=200,validation_split=0.2,callbacks=callbacks)
        


        np.save('input_test',self.in_test)
        np.save('output_test',self.out_test)
        self.model.save('test_model.h5')

if __name__ == '__main__':
    tr = Train()
    model = tr.model
    in_test = tr.in_test
    out_test = tr.out_test
    history = tr.history
#    model = models.load_model('test_model.h5')
#    in_test = np.load('input_test.npy')
#    out_test = np.load('output_test.npy')

    pred_mass = np.transpose( model.predict(in_test))
    actual_mass = np.transpose(out_test)

    background = input_output_var(signal_path_2016, '2016_TT')
    bg_input = background.input
    bg_output = background.output

    background_pred = np.transpose( model.predict(bg_input))
    scaled_bg_pred = np.random.choice( background_pred[0], len(pred_mass[0]))

    diff_mass =  np.absolute(np.subtract(pred_mass, actual_mass))




    fig = plt.figure()
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('stealth_val_plot.png')
    plt.show()
    
    fig = plt.figure()
    plt.hist((diff_mass[0],diff_mass[1]), label = ('Stop 1', 'Stop 2'), histtype = 'step', bins = 50, range=(0,500))
    plt.xlabel('Mass Difference [GeV]')
    plt.ylabel('Events')
    plt.legend()
    plt.savefig('stealth_mass_diff.png')
    plt.show()

    pred_act_weight = 1 / len(pred_mass[0])
    bg_weight = 1 / len(scaled_bg_pred)
    c = ROOT.TCanvas( "c", "c", 0, 0, 1200, 1200)

    ROOT.TH1.SetDefaultSumw2()
    h_pred_S1 = ROOT.TH1D("Stop mass prediction", "Stop mass prediction", 500, 0, 1500)
    h_pred_S1.GetXaxis().SetTitle("Stop mass [GeV]")
    h_pred_S1.GetYaxis().SetTitle("Events")
    h_pred_S1.GetYaxis().SetRangeUser(0,1000)
    h_pred_S1.SetStats(0)
    h_act_S1 = ROOT.TH1D("Actual mass prediction", "Actual  mass prediction", 500, 0, 1500)
    h_bg_S1 = ROOT.TH1D("Background Stop mass prediction", "Background Stop mass prediction", 500, 0, 1500)
    for event in range(0,len(pred_mass[0])):
        h_pred_S1.Fill(pred_mass[0][event])
        h_act_S1.Fill(actual_mass[0][event])    
        h_bg_S1.Fill(scaled_bg_pred[event])
    h_act_S1.SetLineColor(ROOT.kRed)
    h_bg_S1.SetLineColor(ROOT.kGreen+1)    
    legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
    legend.AddEntry(h_pred_S1, "Predicted", "l")
    legend.AddEntry(h_act_S1, "Gen Matched", "l")
    legend.AddEntry(h_bg_S1, "TTBar", "l")
    h_pred_S1.Draw("h e")
    h_act_S1.Draw("same h e")
    h_bg_S1.Draw("same h e")
    legend.Draw("same")
    c.SaveAs("stealth_pred_mass_S1.pdf")
    del c

    c = ROOT.TCanvas( "c", "c", 0, 0, 1200, 1200)
    h_pred_S2 = ROOT.TH1D("Stop mass prediction", "Stop mass prediction", 500, 0, 1500)
    h_pred_S2.GetXaxis().SetTitle("Stop mass [GeV]")
    h_pred_S2.GetYaxis().SetTitle("Events")
    h_pred_S2.GetYaxis().SetRangeUser(0,1000)
    h_pred_S2.SetStats(0)
    h_act_S2 = ROOT.TH1D("Actual mass prediction", "Actual  mass prediction", 500, 0, 1500)
    h_bg_S2 = ROOT.TH1D("Background Stop mass prediction", "Background Stop mass prediction", 500, 0, 1500)
    for event in range(0,len(pred_mass[1])):
        h_pred_S2.Fill(pred_mass[1][event])
        h_act_S2.Fill(actual_mass[1][event])    
        h_bg_S2.Fill(scaled_bg_pred[event])
    h_act_S2.SetLineColor(ROOT.kRed)
    h_bg_S2.SetLineColor(ROOT.kGreen+1)    
    legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
    legend.AddEntry(h_pred_S2, "Predicted", "l")
    legend.AddEntry(h_act_S2, "Gen Matched", "l")
    legend.AddEntry(h_bg_S2, "TTBar", "l")
    h_pred_S2.Draw("h e")
    h_act_S2.Draw("same h e")
    h_bg_S2.Draw("same h e")
    legend.Draw("same")
    c.SaveAs("stealth_pred_mass_S2.pdf")
    del c


