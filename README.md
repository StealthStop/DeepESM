# DeepESM
Produces the DeepESM training model

##Python packages needed (As far as I can tell)

```
python                        2.7
dask                          0.19.0   
h5py                          2.8.0    
Keras                         2.2.2    
matplotlib                    2.2.3    
numpy                         1.14.5   
pandas                        0.23.4   
pip                           18.0     
protobuf                      3.6.1    
scikit-learn                  0.19.2   
tensorboard                   1.10.0   
tensorflow                    1.10.1   
Theano                        1.0.2    
```
Install using pip
```
pip install <Package Name>
```
or
```
sudo pip install <Package Name>
```

This package is not manditory, but is highly recomended to isolate all the packages you will install from your main python installation
```
virtualenv
```
https://virtualenv.pypa.io/en/stable/

## Training files stored on eos

Location of training files (Less than 1G total)
```
eosls -l /store/user/cmadrid/trainingTuples
```

Variables in latest training
```
        {"EventShapeVar", {"EvtNum_double",
                           "sampleWgt",
                           "Weight",
                           "fwm2_top6", 
                           "fwm3_top6", 
                           "fwm4_top6", 
                           "fwm5_top6", 
                           "fwm6_top6", 
                           "fwm7_top6", 
                           "fwm8_top6", 
                           "fwm9_top6", 
                           "fwm10_top6", 
                           "jmt_ev0_top6", 
                           "jmt_ev1_top6", 
                           "jmt_ev2_top6",
                           "NGoodJets_double",
                           "Jet_pt_1",
                           "Jet_pt_2",
                           "Jet_pt_3",
                           "Jet_pt_4",
                           "Jet_pt_5",
                           "Jet_pt_6",
                           "Jet_pt_7",
                           "Jet_eta_1",
                           "Jet_eta_2",
                           "Jet_eta_3",
                           "Jet_eta_4",
                           "Jet_eta_5",
                           "Jet_eta_6",
                           "Jet_eta_7",
                           "Jet_phi_1",
                           "Jet_phi_2",
                           "Jet_phi_3",
                           "Jet_phi_4",
                           "Jet_phi_5",
                           "Jet_phi_6",
                           "Jet_phi_7",
                           "Jet_m_1",
                           "Jet_m_2",
                           "Jet_m_3",
                           "Jet_m_4",
                           "Jet_m_5",
                           "Jet_m_6",
                           "Jet_m_7",
                           "GoodLeptons_pt_1",
                           "GoodLeptons_eta_1",
                           "GoodLeptons_phi_1",
                           "GoodLeptons_m_1",
                           "lvMET_cm_pt",
                           "lvMET_cm_eta",
                           "lvMET_cm_phi",
                           "lvMET_cm_m",
                           "BestComboAvgMass"}}
```

## Running the training

Get the code, make sure you have the training file and run

```
git clone git@github.com:StealthStop/DeepESM.git
cd DeepESM
rm -rf TEST && python train.py
```
