# DeepESM
Produces the DeepESM training model

## Python packages needed (As far as I can tell)

```
python 
dask   
h5py   
Keras  
matplotlib 
numpy      
pandas     
pip        
protobuf   
scikit-learn
tensorboard 
tensorflow  
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

# Running on the LPC

Log into one of the three gpu nodes and setting up..

```
ssh -Y username@cmslpcgpu1.fnal.gov # can be cmslpcgpu 1-3
cd ~/nobackup/
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh <<< $'\nyes\n~/nobackup/anaconda3\nyes\n'
rm Anaconda3-2019.10-Linux-x86_64.sh
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/Setup.sh
source ~/nobackup/anaconda3/bin/activate
conda update -n base -c defaults conda <<< $'y\n'
conda create -n pt python=3.7 anaconda <<< $'y\n'
conda activate pt
conda install -n pt libgcc pandas scikit-learn tensorboard tensorflow-gpu Keras matplotlib numpy=1.16.6 dask h5py protobuf pydot pytorch torchvision cudatoolkit=10.0 <<< $'y\n'
pip install uproot
pip install coffea
pip install mplhep
pip install pypi
pip install matplotlib==3.3.0
cd anaconda3/envs/pt
git clone git@github.com:fizisist/LorentzGroupNetwork.git
cd LorentzGroupNetwork
pip install -e .

cd ~/nobackup/
conda create -n tf python=3.7 anaconda <<< $'y\n'
conda activate tf
conda install -n tf libgcc pandas scikit-learn tensorboard tensorflow=2.2.0 tensorflow-gpu Keras=2.4.3 matplotlib numpy=1.18.5 dask h5py protobuf pydot pytorch torchvision cudatoolkit <<< $'y\n'
pip install uproot
pip install coffea
pip install mplhep==0.1.35
pip install pypi
pip install matplotlib==3.3.0
```

Get the code, make sure you have the training file and run

```
cd WORKINGAREA
git clone git@github.com:StealthStop/DeepESM.git
cd DeepESM
mkdir MVA_Training_Files_FullRun2_V2
xrdcp -r root://cmseos.fnal.gov///store/user/cmadrid/trainingTuples/MVA_Training_Files_FullRun2_V2/ ./MVA_Training_Files_FullRun2_V2/
python train.py
```

### Alternative LPC setup: Python Virtual Environment

To begin the initial setup, run the following commands:
```bash
cd <working_directory>
git clone git@github.com:StealthStop/DeepESM.git
cd DeepESM
./setup.sh
```
Remember to replace `<working_directory>` with the directory where you want your files/folders to appear. You can change the name of the virtual environment by using the `-n` option and you can use the development version of coffea by using the `-d` option. These commands only need to be run during the initial setup. When doing your day-to-day tasks, you can skip these.

To activate the `coffeaenv` environment and set the Jupyter paths, run the command (every time):
```bash
cd <working_directory>/DeepESM
source init.sh
```

When you are done working and would like to ``de-activate'' the `coffeaenv` environment, run the command:
```bash
deactivate
```
This shell function was given to you by the virtual environment.

To remove the virtual environment and the associated files (i.e. inverse of the setup script), you can use the run the following command:
```bash
cd <working_directory>/DeepESM
./clean.sh
```
The `clean.sh` script has the same `-n` and `-d` options as in the `setup.sh` script.

### Plotting Input Variables

A plotting script is provided to make pretty plots of NN inputs from the ntuple files.

Arguments to the script are:

```
--approved : is Plot is approved?
--path     : Path to ntuples files
--tree     : TTree name to use
--year     : which year
--mass1    : mass 1 to show
--mass2    : mass 2 to show
--model1   : model 1 to show
--model2   : model 2 to show
```

An example to run the script could be:

```
python ttVsSigNN_mini.py --year 2016 --path /path/to/ntuples/files --mass1 350 --model1 RPV --mass2 500 --model2 StealthSYY
```
