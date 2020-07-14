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



masses = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]

RPV_f_2016  = '2016_RPV_2t6j_mStop-' 
RPV_f_2017  = '2017_RPV_2t6j_mStop-' 

stealth_f_2016 = '2016_StealthSYY_2t6j_mStop-'
stealth_f_2017 = '2017_StealthSYY_2t6j_mStop-'
RPV_signal_2016 = ['2016_RPV_2t6j_mStop-300','2016_RPV_2t6j_mStop-350','2016_RPV_2t6j_mStop-400','2016_RPV_2t6j_mStop-450','2016_RPV_2t6j_mStop-500','2016_RPV_2t6j_mStop-550','2016_RPV_2t6j_mStop-600','2016_RPV_2t6j_mStop-650','2016_RPV_2t6j_mStop-700','2016_RPV_2t6j_mStop-750','2016_RPV_2t6j_mStop-800','2016_RPV_2t6j_mStop-850', '2016_RPV_2t6j_mStop-900']

stealth_signal_2016=['2016_StealthSYY_2t6j_mStop-300','2016_StealthSYY_2t6j_mStop-350','2016_StealthSYY_2t6j_mStop-400','2016_StealthSYY_2t6j_mStop-500','2016_StealthSYY_2t6j_mStop-550','2016_StealthSYY_2t6j_mStop-600','2016_StealthSYY_2t6j_mStop-650','2016_StealthSYY_2t6j_mStop-700','2016_StealthSYY_2t6j_mStop-750','2016_StealthSYY_2t6j_mStop-800','2016_StealthSYY_2t6j_mStop-850','2016_StealthSYY_2t6j_mStop-900']

RPV_signal_2017 = ['2017_RPV_2t6j_mStop-300','2017_RPV_2t6j_mStop-350','2017_RPV_2t6j_mStop-400','2017_RPV_2t6j_mStop-450','2017_RPV_2t6j_mStop-500','2017_RPV_2t6j_mStop-550','2017_RPV_2t6j_mStop-600','2017_RPV_2t6j_mStop-650','2017_RPV_2t6j_mStop-700','2017_RPV_2t6j_mStop-750','2017_RPV_2t6j_mStop-800','2017_RPV_2t6j_mStop-850','2017_RPV_2t6j_mStop-900']

stealth_signal_2017 =['2017_StealthSYY_2t6j_mStop-300','2017_StealthSYY_2t6j_mStop-350','2017_StealthSYY_2t6j_mStop-400','2017_StealthSYY_2t6j_mStop-500','2017_StealthSYY_2t6j_mStop-550','2017_StealthSYY_2t6j_mStop-600','2017_StealthSYY_2t6j_mStop-650','2017_StealthSYY_2t6j_mStop-700','2017_StealthSYY_2t6j_mStop-750','2017_StealthSYY_2t6j_mStop-800','2017_StealthSYY_2t6j_mStop-850','2017_StealthSYY_2t6j_mStop-900']

signal_path_2016 = '~/nobackup/SUSY3/CMSSW_10_2_9/src/Analyzer/Analyzer/test/condor/MC_Training_Input_2016_v4/'
signal_path_2017 = '~/nobackup/SUSY3/CMSSW_10_2_9/src/Analyzer/Analyzer/test/condor/MC_Training_Input_2017_v4/'

#class for making the input/output variable numpy arrays using uproot. jets are zero padded based on the maximum number of jets in all events


class input_output_var:
    def __init__(self,path,sample, mass):
            file = uproot.open(path+sample+'.root')
    
            events = file['myMiniTree;1']

            nJets                              = events.array("NGoodJets_pt30")
            nBJets                             = events.array("NGoodBJets_pt30")
#            FirstStopMass                      = events.array("FirstStopMassSums")
#            SecStopMass                        = events.array("SecStopMassSums")
#            StopMassDiff                       = events.array("StopMassDiffs")
            Mbl1                               = events.array("TwoLep_Mbl1")
            Mbl2                               = events.array("TwoLep_Mbl2")
#            StopMT2                            = events.array("GM_StopMT2")
#            Stop1                              = events.array("GM_Stop1")
#            Stop1Mass                          = Stop1[:].mass
#            Stop2                              = events.array("GM_Stop2")
#            Stop2Mass                          = Stop2[:].mass
            lepton_mass                        = self.zero_pad(events.array("GoodLeptonsMass"), 2)
            lepton_pt                          = self.zero_pad(events.array("GoodLeptonsPt"), 2)
            lepton_phi                         = self.zero_pad(events.array("GoodLeptonsPhi"), 2)
            lepton_eta                         = self.zero_pad(events.array("GoodLeptonsEta"), 2)
            lepton_px                          = self.zero_pad(events.array("GoodLeptonsPx"), 2)
            lepton_py                          = self.zero_pad(events.array("GoodLeptonsPy"), 2)
            lepton_pz                          = self.zero_pad(events.array("GoodLeptonsPz"), 2)
#            lepton_E                           = self.zero_pad(events.array("GoodLeptonsE"), 2)


            jet_mass                           = self.zero_pad(events.array("GoodJetsMass"), 15)
            jet_pt                             = self.zero_pad(events.array("GoodJetsPt"), 15)
            jet_eta                            = self.zero_pad(events.array("GoodJetsEta"), 15)
            jet_phi                            = self.zero_pad(events.array("GoodJetsPhi"), 15)
            jet_px                             = self.zero_pad(events.array("GoodJetsPx"), 15)
            jet_py                             = self.zero_pad(events.array("GoodJetsPy"), 15)
            jet_pz                             = self.zero_pad(events.array("GoodJetsPz"), 15)
            jet_E                              = self.zero_pad(events.array("GoodJetsE"), 15)
            

            prelim_sig_input = np.transpose(np.array([nJets, nBJets, Mbl1, Mbl2]))
            self.input = np.concatenate([prelim_sig_input,lepton_mass, lepton_pt, lepton_eta, lepton_phi, lepton_px, lepton_py, lepton_pz, jet_mass, jet_pt, jet_eta, jet_phi, jet_px, jet_py, jet_pz, jet_E], axis=1)
#            self.output  = np.transpose(np.array([Stop1Mass, Stop2Mass]))
            self.output = np.full((np.size(self.input,0),2), mass)
            
            
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
        input_list = []
        output_list = []
        max_events = 0
        for sig in range(0,len(masses)):
            mass = masses[sig]
            IO_vars_2016 = input_output_var(signal_path_2016, RPV_signal_2016[sig], mass)
            IO_vars_2017 = input_output_var(signal_path_2017, RPV_signal_2017[sig], mass)
            sig_input = np.concatenate([IO_vars_2016.input,IO_vars_2017.input])
            sig_output = np.concatenate([IO_vars_2016.output,IO_vars_2017.output])
            
            nevents = np.shape(sig_input)[0]
            if nevents > max_events:
                max_events = nevents

            input_list.append(sig_input)
            output_list.append(sig_output)
            
          #  print(mass, np.shape(sig_input))

#            sig_input = IO_vars_2016.input
#            sig_output = IO_vars_2016.output
#            if sig == 0:
#                input_variables = sig_input
#                output_variables = sig_output
#            else:
#                input_variables = np.concatenate([input_variables,sig_input])
#                output_variables = np.concatenate([output_variables,sig_output])
        for m in input_list:
            print(np.shape(m))
        for sig_mass in range(len(input_list)):
            if np.shape(input_list[sig_mass])[0] < max_events:
               input_list[sig_mass] = np.resize(input_list[sig_mass], (max_events, np.shape(input_list[sig_mass])[1]))
        for sig_mass in range(len(output_list)):
            if np.shape(output_list[sig_mass])[0] < max_events:
               output_list[sig_mass] = np.resize(output_list[sig_mass], (max_events, np.shape(output_list[sig_mass])[1]))

        
        input_variables = np.vstack(input_list)
        output_variables = np.vstack(output_list)
        print(np.shape(input_variables))
#there are some NaNs in the output, this removes it but need to find source of issue
        check = np.argwhere(np.isnan(output_variables))
        add = 0
        for x in check[:,0]:
            output_variables = np.delete(output_variables, x-add, 0)
            input_variables = np.delete(input_variables, x-add, 0)
            add += 1 


        assert not np.any(np.isnan(input_variables))
        assert not np.any(np.isnan(output_variables))
#        print(sig_input[0])


        bg_TT_IO = input_output_var(signal_path_2016, '2016_TT', 0)
        bg_QCD_IO = input_output_var(signal_path_2016, '2016_QCD', 0)
        bg_DY_IO = input_output_var(signal_path_2016, '2016_DYJetsToLL_M-50', 0)


        self.in_train_val, self.in_test, self.out_train_val, self.out_test = train_test_split(input_variables, output_variables, test_size=0.2, random_state=7)

        self.TT_in_train_val, self.TT_in_test, self.TT_out_train_val, self.TT_out_test = train_test_split(bg_TT_IO.input, bg_TT_IO.output, test_size=0.2, random_state=7)

        self.QCD_in_train_val, self.QCD_in_test, self.QCD_out_train_val, self.QCD_out_test = train_test_split(bg_QCD_IO.input, bg_QCD_IO.output, test_size=0.2, random_state=7)

        self.DY_in_train_val, self.DY_in_test, self.DY_out_train_val, self.DY_out_test = train_test_split(bg_DY_IO.input, bg_DY_IO.output, test_size=0.2, random_state=7)

        input_train = np.concatenate([self.in_train_val,self.TT_in_train_val,self.QCD_in_train_val,self.DY_in_train_val])
        output_train = np.concatenate([self.out_train_val,self.TT_out_train_val,self.QCD_out_train_val,self.DY_out_train_val])

#        input_train = self.in_train_val
#        output_train = self.out_train_val

        print("Done setting up input variables. Will now begin training")

        inputs = Input(shape=(np.size(input_train,1),), name = 'input')
        h_1 = Dense(2*np.size(input_train,1), name = 'h_1', activation = 'relu')(inputs)
        outputs = Dense(2, name = 'output', kernel_initializer='normal', activation = 'relu')(h_1)


        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse',optimizer='rmsprop')

        callbacks = [ EarlyStopping(verbose = True, patience = 10, monitor = 'val_loss'),
              ModelCheckpoint('test_model.h5',monitor='val_loss',verbose=True,save_best_only=True)]

        self.history = self.model.fit(input_train,output_train,batch_size=64,epochs=2000,validation_split=0.125,callbacks=callbacks)
        


#        np.save('input_test',self.in_test)
#        np.save('output_test',self.out_test)
        self.model.save('test_model.h5')


if __name__ == '__main__':
    tr = Train()
    print('Finished training, now testing')
    model = tr.model
    in_test = tr.in_test
    out_test = tr.out_test
    history = tr.history

    TT_in_test = tr.TT_in_test
    QCD_in_test = tr.QCD_in_test
    DY_in_test = tr.DY_in_test

    TT_pred = np.transpose(model.predict(TT_in_test))
    QCD_pred = np.transpose(model.predict(QCD_in_test))
    DY_pred = np.transpose(model.predict(DY_in_test))

 #   model = models.load_model('test_model.h5')
 #   in_test = np.load('input_test.npy')
 #   out_test = np.load('output_test.npy')

    pred_mass = np.transpose( model.predict(in_test))
    actual_mass = np.transpose(out_test)

    print('Finished testing, making plots')
    

    diff_mass =  np.absolute(np.subtract(pred_mass, actual_mass))



#    fig = plt.figure()
#    plt.plot(history.history['val_loss'], label='val_loss')
#    plt.plot(history.history['loss'], label='loss')
#    plt.xlabel('Epoch')
#    plt.ylabel('Loss')
#    plt.legend()
#    plt.savefig('RPV_val_plot.png')
#    plt.show()
    
#    fig = plt.figure()
#    plt.hist((diff_mass[0],diff_mass[1]), label = ('Stop 1', 'Stop 2'), histtype = 'step', bins = 50, range=(0,500))
#    plt.xlabel('Mass Difference [GeV]')
#    plt.ylabel('Events')
#    plt.legend()
#    plt.savefig('RPV_mass_diff.png')
#    plt.show()

#    pred_act_weight = 1 / len(pred_mass[0])
#    bg_weight = 1 / len(scaled_bg_pred)

    for mass in masses:
        c = ROOT.TCanvas( "c", "c", 0, 0, 1200, 1200)

        ROOT.TH1.SetDefaultSumw2()
        h_pred_S1 = ROOT.TH1D("Stop mass prediction", "Stop mass prediction", 500, 0, 1500)
        h_pred_S1.SetTitle('Stop Mass Prediction-'+str(mass))
        h_pred_S1.GetXaxis().SetTitle("Stop mass [GeV]")
        h_pred_S1.GetYaxis().SetTitle("Normalized Events")
        h_pred_S1.SetStats(0)
        h_TT_S1 = ROOT.TH1D("TTbar Stop mass prediction", "TTbar Stop mass prediction", 500, 0, 1500)
        h_QCD_S1 = ROOT.TH1D("QCD Stop mass prediction", "QCD Stop mass prediction", 500, 0, 1500)
        h_DY_S1 = ROOT.TH1D("DY Stop mass prediction", "DY Stop mass prediction", 500, 0, 1500)

        for event in range(0,len(pred_mass[0])):
            if actual_mass[0][event] == mass and in_test[event,0] >= 7:
                h_pred_S1.Fill(pred_mass[0][event])
        for event in range(0,len(TT_pred[0])):
            h_TT_S1.Fill(TT_pred[0][event])
        
        for event in range(0,len(QCD_pred[0])):
            h_QCD_S1.Fill(QCD_pred[0][event])
        for event in range(0,len(DY_pred[0])):   
            h_DY_S1.Fill(DY_pred[0][event])
        
        h_TT_S1.Scale(1 / h_TT_S1.Integral())
        h_QCD_S1.Scale(1 / h_QCD_S1.Integral())
        h_DY_S1.Scale(1 / h_DY_S1.Integral())
        h_pred_S1.Scale(1 / h_pred_S1.Integral())
        h_TT_S1.SetLineColor(ROOT.kRed)
        h_QCD_S1.SetLineColor(ROOT.kGreen+1)
        h_DY_S1.SetLineColor(ROOT.kOrange)
        legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
        legend.AddEntry(h_pred_S1, "Signal", "l")
    #    legend.AddEntry(h_act_S1, "Gen Matched", "l")
        legend.AddEntry(h_TT_S1, "TTBar", "l")
        legend.AddEntry(h_QCD_S1, "QCD", "l")
        legend.AddEntry(h_DY_S1, "DY", "l")
        h_pred_S1.Draw("h e")
    #    h_act_S1.Draw("same h e")
        h_TT_S1.Draw("same h e")
        h_QCD_S1.Draw("same h e")
        h_DY_S1.Draw("same h e")
        legend.Draw("same")

        h_pred_S1.Draw("h e")
        c.SaveAs("RPV_pred_mass_"+str(mass)+".pdf")
        del c
        del h_pred_S1

    c = ROOT.TCanvas( "c", "c", 0, 0, 1200, 1200)
    h_pred_S1 = ROOT.TH1D("Stop mass prediction", "Stop mass prediction", 500, 0, 1500)
    h_pred_S1.GetXaxis().SetTitle("Stop mass [GeV]")
    h_pred_S1.GetYaxis().SetTitle("Events")
#    h_pred_S1.GetYaxis().SetRangeUser(0,1000)
    h_pred_S1.SetStats(0)
    h_act_S1 = ROOT.TH1D("Actual mass prediction", "Actual  mass prediction", 500, 0, 1500)
    h_TT_S1 = ROOT.TH1D("TTbar Stop mass prediction", "TTbar Stop mass prediction", 500, 0, 1500)
    h_QCD_S1 = ROOT.TH1D("QCD Stop mass prediction", "QCD Stop mass prediction", 500, 0, 1500)
    h_DY_S1 = ROOT.TH1D("DY Stop mass prediction", "DY Stop mass prediction", 500, 0, 1500)
    for event in range(0,len(pred_mass[0])):
        h_pred_S1.Fill(pred_mass[0][event])
#        h_act_S1.Fill(actual_mass[0][event])    
    for event in range(0,len(TT_pred[0])):
        h_TT_S1.Fill(TT_pred[0][event])
        
    for event in range(0,len(QCD_pred[0])):
        h_QCD_S1.Fill(QCD_pred[0][event])
    for event in range(0,len(DY_pred[0])):   
        h_DY_S1.Fill(DY_pred[0][event])
   
    h_TT_S1.SetLineColor(ROOT.kRed)
    h_QCD_S1.SetLineColor(ROOT.kGreen+1)
    h_DY_S1.SetLineColor(ROOT.kOrange)
    legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
    legend.AddEntry(h_pred_S1, "Signal", "l")
#    legend.AddEntry(h_act_S1, "Gen Matched", "l")
    legend.AddEntry(h_TT_S1, "TTBar", "l")
    legend.AddEntry(h_QCD_S1, "QCD", "l")
    legend.AddEntry(h_DY_S1, "DY", "l")
    h_pred_S1.Draw("h e")
#    h_act_S1.Draw("same h e")
    h_TT_S1.Draw("same h e")
    h_QCD_S1.Draw("same h e")
    h_DY_S1.Draw("same h e")
    legend.Draw("same")
    c.SaveAs("RPV_pred_mass_S1.pdf")
    del c

    
    c = ROOT.TCanvas( "c", "c", 0, 0, 1200, 1200)
    h_pred_S2 = ROOT.TH1D("Stop mass prediction", "Stop mass prediction", 500, 0, 1500)
    h_pred_S2.GetXaxis().SetTitle("Stop mass [GeV]")
    h_pred_S2.GetYaxis().SetTitle("Events")
#    h_pred_S2.GetYaxis().SetRangeUser(0,1000)
    h_pred_S2.SetStats(0)
    h_act_S2 = ROOT.TH1D("Actual mass prediction", "Actual  mass prediction", 500, 0, 1500)
    h_TT_S2 = ROOT.TH1D("TTbar Stop mass prediction", "TTbar Stop mass prediction", 500, 0, 1500)
    h_QCD_S2 = ROOT.TH1D("QCD Stop mass prediction", "QCD Stop mass prediction", 500, 0, 1500)
    h_DY_S2 = ROOT.TH1D("DY Stop mass prediction", "DY Stop mass prediction", 500, 0, 1500)
    for event in range(0,len(pred_mass[1])):
        h_pred_S2.Fill(pred_mass[1][event])
#        h_act_S2.Fill(actual_mass[1][event])    
    for event in range(0,len(TT_pred[0])):
        h_TT_S1.Fill(TT_pred[0][event])
        
    for event in range(0,len(QCD_pred[0])):
        h_QCD_S1.Fill(QCD_pred[0][event])
    for event in range(0,len(DY_pred[0])):   
        h_DY_S1.Fill(DY_pred[0][event])

    h_act_S2.SetLineColor(ROOT.kRed)
    h_TT_S2.SetLineColor(ROOT.kBlue)
    h_QCD_S2.SetLineColor(ROOT.kGreen+1)
    h_DY_S2.SetLineColor(ROOT.kOrange)
    legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
    legend.AddEntry(h_pred_S2, "Signal", "l")
#    legend.AddEntry(h_act_S2, "Gen Matched", "l")
    legend.AddEntry(h_TT_S2, "TTBar", "l")
    legend.AddEntry(h_QCD_S2, "QCD", "l")
    legend.AddEntry(h_DY_S2, "DY", "l")
    h_pred_S2.Draw("h e")
#    h_act_S2.Draw("same h e")
    h_TT_S2.Draw("same h e")
    h_QCD_S2.Draw("same h e")
    h_DY_S2.Draw("same h e")
    legend.Draw("same")
    c.SaveAs("RPV_pred_mass_S2.pdf")
    del c



    for njet in range(1,11):
        
        c = ROOT.TCanvas( "c", "c", 0, 0, 1200, 1200)
        h_pred_S1 = ROOT.TH1D("Stop mass prediction", "Stop mass prediction", 500, 0, 1500)
        h_pred_S1.GetXaxis().SetTitle("Stop mass [GeV]")
        h_pred_S1.GetYaxis().SetTitle("Events")
        h_pred_S1.SetTitle('Stop prediction, nJets = '+str(njet))
    #   h_pred_S1.GetYaxis().SetRangeUser(0,1000)
        h_pred_S1.SetStats(0)
        h_act_S1 = ROOT.TH1D("Actual mass prediction", "Actual  mass prediction", 500, 0, 1500)
        h_TT_S1 = ROOT.TH1D("TTbar Stop mass prediction", "TTbar Stop mass prediction", 500, 0, 1500)
        h_QCD_S1 = ROOT.TH1D("QCD Stop mass prediction", "QCD Stop mass prediction", 500, 0, 1500)
        h_DY_S1 = ROOT.TH1D("DY Stop mass prediction", "DY Stop mass prediction", 500, 0, 1500)

        for event in range(0,len(pred_mass[0])):
            if in_test[event,0] == njet:
                h_pred_S1.Fill(pred_mass[0][event])
            #       h_act_S1.Fill(actual_mass[0][event])
        for event in range(0,len(TT_pred[0])):
            if TT_in_test[event,0] == njet:
                h_TT_S1.Fill(TT_pred[0][event])
        
        for event in range(0,len(QCD_pred[0])):
            if QCD_in_test[event,0] == njet:
                h_QCD_S1.Fill(QCD_pred[0][event])
        for event in range(0,len(DY_pred[0])):
            if DY_in_test[event,0] == njet:
                h_DY_S1.Fill(DY_pred[0][event])


       # h_TT_S1.Scale(1 / h_TT_S1.Integral())
  #      h_QCD_S1.Scale(1 / h_QCD_S1.Integral())
  #      h_DY_S1.Scale(1 / h_DY_S1.Integral())
      #  h_pred_S1.Scale(1 / h_pred_S1.Integral())
        h_act_S1.SetLineColor(ROOT.kRed)
        h_TT_S1.SetLineColor(ROOT.kRed)    

        h_QCD_S1.SetLineColor(ROOT.kGreen+1)
        h_DY_S1.SetLineColor(ROOT.kOrange)

        legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
        legend.AddEntry(h_pred_S1, "Signal", "l")
    #    legend.AddEntry(h_act_S1, "Gen Matched", "l")
        legend.AddEntry(h_TT_S1, "TTBar", "l")

        legend.AddEntry(h_QCD_S1, "QCD", "l")
        legend.AddEntry(h_DY_S1, "DY", "l")

        h_pred_S1.Draw("h e")
    #   h_act_S1.Draw("same h e")
        h_TT_S1.Draw("same h e")
 #       h_QCD_S1.Draw("same h e")
 #       h_DY_S1.Draw("same h e")

        legend.Draw("same")
        c.SaveAs("RPV_pred_mass_S1_njet-"+str(njet)+".pdf")
        del c
        del h_TT_S1
        del h_pred_S1 

    
                
