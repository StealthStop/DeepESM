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

# Running at the Minnesota Supercomputing Institute

Connect to the Mangi (V100 nodes) or Agate (A100 nodes) cluster.
Please visit https://www.msi.umn.edu/ for more information on connecting.

## Setting Up the Working Area
```
mkdir -p Train

cd Train
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh <<< $'\nyes\n~/anaconda3\nyes\n'
rm Anaconda3-2019.10-Linux-x86_64.sh

source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/Setup.sh
source ~/anaconda3/bin/activate

conda update -n base -c defaults conda <<< $'y\n'

conda create -n tf python=3.7 anaconda <<< $'y\n'
conda activate tf
conda install -n tf libgcc pandas scikit-learn tensorboard tensorflow=2.2.0 tensorflow-gpu Keras=2.4.3 matplotlib numpy=1.18.5 dask h5py protobuf pydot pytorch torchvision cudatoolkit <<< $'y\n'
pip install uproot
pip install coffea
pip install mplhep==0.1.35
pip install pypi
pip install matplotlib==3.3.0
```

Get the analysis code from GitHub and rsync over the ROOT file NN inputs from the LPC

```
cd Train

git clone git@github.com:StealthStop/DeepESM.git
cd DeepESM

rsync -r <lpcuser>@cmslpc120.fnal.gov:/uscmst1b_scratch/lpc1/3DayLifetime/<some path> .
source deepenv.sh
```

## Running Interactively

On the MSI system, one can run interactive jobs, which run on GPU nodes and whose output is returned to the user's terminal.
Most use cases are for debugging the code and testing purposes.
This interactive running is performed using the `srun` command.
The command lets the user allocate a custom amount of CPU/GPU/RAM resources as well as time for running their program.

An example call of `srun` would be of the form:
```
srun -u \
     -t 0:40:00 \
     -p interactive-gpu \
     --gres=gpu:k40:1 \
     --mem-per-cpu=30G 
     python train.py --saveAndPrint --procCats --njetsCats --massCats --minMass 350 --maxMass 1150 --evalMass 550 --trainModel RPV --evalModel RPV --year 2016preVFP --seed 527725 --tree myMiniTree_1l --nJets 7 --inputs UL_NN_inputs/
```
where 40 minutes of GPU time is requested on an interactive GPU node in the Mangi cluster (k40, for Agate cluster one would use a40) and 30 GB of RAM for loading events from disk.
Finally, the last argument provided is the entire `python` call to the executable to run, in this case `train.py`.
These `train.py` arguments are detailed below.
```
usage: usage: %prog [options] [-h] [--quickVal] [--json JSON]
                              [--minMass MINMASS] [--maxMass MAXMASS]
                              [--evalMass EVALMASS] [--evalModel EVALMODEL]
                              [--evalYear EVALYEAR] [--trainModel TRAINMODEL]
                              [--replay] [--trainYear TRAINYEAR]
                              [--inputs INPUTS] [--tree TREE] [--saveAndPrint]
                              [--seed SEED] [--nJets NJETS] [--debug]
                              [--scaleJetPt] [--useJECs]
                              [--maskNjet MASKNJET [MASKNJET ...]]
                              [--procCats] [--massCats] [--njetsCats]
                              [--outputDir OUTPUTDIR]

optional arguments:
  -h, --help            show this help message and exit
  --quickVal            Do quick (partial) validation
  --json JSON           JSON config file
  --minMass MINMASS     Minimum stop mass to train on
  --maxMass MAXMASS     Maximum stop mass to train on
  --evalMass EVALMASS   Stop mass to evaluate on
  --evalModel EVALMODEL
                        Signal model to evaluate on
  --evalYear EVALYEAR   Year(s) to eval on
  --trainModel TRAINMODEL
                        Signal model to train on
  --replay              Replay saved model
  --trainYear TRAINYEAR
                        Year(s) to train on
  --inputs INPUTS       Path to input files
  --tree TREE           TTree to load events from
  --saveAndPrint        Save pb and print model
  --seed SEED           Use specific seed for env
  --nJets NJETS         Minimum number of jets
  --debug               Debug with small set of events
  --scaleJetPt          Scale Jet pt by HT
  --useJECs             Use JEC/JER variations
  --maskNjet MASKNJET [MASKNJET ...]
                        mask Njet bin(s) in training
  --procCats            Balance batches bkg/sig
  --massCats            Balance batches among masses
  --njetsCats           Balance batches among njets
  --outputDir OUTPUTDIR
                        Output directory path
```

##

The most powerful use case is submitting NN training jobs in batch to the Mangi or Agate GPU clusters.
This is acheived using the `boboTrain.py` script, whose arguments are detailed:
```
usage: boboTrain.py [-h] [--trainBkgd TRAINBKGD [TRAINBKGD ...]]
                    [--trainModel TRAINMODEL]
                    [--evalBkgd EVALBKGD [EVALBKGD ...]]
                    [--evalModel EVALMODEL]
                    [--trainMass TRAINMASS [TRAINMASS ...]]
                    [--evalMass EVALMASS] [--tag TAG]
                    [--bcorr BCORR [BCORR ...]] [--disc DISC [DISC ...]]
                    [--abcd ABCD [ABCD ...]] [--reg REG [REG ...]]
                    [--nodes NODES [NODES ...]] [--reglr REGLR [REGLR ...]]
                    [--disclr DISCLR [DISCLR ...]]
                    [--factors FACTORS [FACTORS ...]]
                    [--epochs EPOCHS [EPOCHS ...]] [--trainYear TRAINYEAR]
                    [--evalYear EVALYEAR] [--seed SEED] [--channel CHANNEL]
                    [--noSubmit] [--cluster CLUSTER] [--memory MEMORY]
                    [--walltime WALLTIME] [--useJECs] [--nJets NJETS]
                    [--maskNjet MASKNJET [MASKNJET ...]] [--procCats]
                    [--massCats] [--njetsCats] [--saveAndPrint]
                    [--inputs INPUTS]

optional arguments:
  -h, --help            show this help message and exit
  --trainBkgd TRAINBKGD [TRAINBKGD ...]
                        which bkgd to train on
  --trainModel TRAINMODEL
                        which sig to train on
  --evalBkgd EVALBKGD [EVALBKGD ...]
                        which bkgd to validate on
  --evalModel EVALMODEL
                        which model to validate on
  --trainMass TRAINMASS [TRAINMASS ...]
                        lower and upper mass range bounds
  --evalMass EVALMASS   which mass point to validate on
  --tag TAG             tag to use in output
  --bcorr BCORR [BCORR ...]
                        list of bcorr lambda values
  --disc DISC [DISC ...]
                        list of disc lambda values
  --abcd ABCD [ABCD ...]
                        list of abcd lambda values
  --reg REG [REG ...]   list of reg lambda values
  --nodes NODES [NODES ...]
                        list of nodes values
  --reglr REGLR [REGLR ...]
                        regression lr
  --disclr DISCLR [DISCLR ...]
                        disc lr
  --factors FACTORS [FACTORS ...]
                        list of factors to multiply
  --epochs EPOCHS [EPOCHS ...]
                        how many epochs
  --trainYear TRAINYEAR
                        which year(s) to train on
  --evalYear EVALYEAR   which year to eval on
  --seed SEED           which seed to init with
  --channel CHANNEL     which decay channel
  --noSubmit            do not submit to cluster
  --cluster CLUSTER     which cluster to run on
  --memory MEMORY       how much mem to request
  --walltime WALLTIME   how much time to request
  --useJECs             use JEC/JER variation events
  --nJets NJETS         Minimum number of jets
  --maskNjet MASKNJET [MASKNJET ...]
                        mask Njet bin/bins in training
  --procCats            Balance batches bkg/sig
  --massCats            Balance batches among masses
  --njetsCats           Balance batches among njets
  --saveAndPrint        Save model peanut butter
  --inputs INPUTS       which inputs files to use
```

An example call to `boboTrain.py` would be:
```
python boboTrain.py --saveAndPrint \
                    --procCats \
                    --njetsCats \
                    --useJECs \
                    --channel 1l \
                    --epochs 15 20 25 \
                    --bcorr 1000 2000 \
                    --disc 1.0 3.0 2.0 5.0 \
                    --abcd 1.0 2.0 3.0 5.0 \
                    --disclr 0.001 \
                    --reg 0.0001 \
                    --reglr 1.0 \
                    --trainYear Run2 \
                    --trainSig RPV \
                    --evalSig RPV \
                    --evalMass 550 \
                    --evalYear 2016preVFP \
                    --tag Run2_RPV \
                    --inputs UL_NN_inputs/ \
                    --memory 50gb \
                    --walltime 01:30:00 \
                    --cluster a100-4 
                    --noSubmit
```

Running this command will generate a `Run2_RPV_<unique_timestamp>` in `./batch`.
No jobs have been submitted yet, but that is acheived by going to `./batch/Run2_RPV_<unique_timestamp` and running `qsub job_submit.pbs`.
The status of jobs can be checked using the command `qstat -a -f -M -u $USER`.

## Resubmitting Jobs
Occaisionally, some jobs may encounter a segmentation violation and crash.
A `resubmit.py` script has been provided to generate a new `.pbs` submission file with just the jobs that did not complete successfully.
The arguments are detailed below:
```
usage: resubmit.py [-h] [--jobDir JOBDIR] [--cluster CLUSTER]
                   [--memory MEMORY] [--walltime WALLTIME]

optional arguments:
  -h, --help           show this help message and exit
  --jobDir JOBDIR      Directory where jobs submitted from
  --cluster CLUSTER    which cluster to run on
  --memory MEMORY      how much mem to request
  --walltime WALLTIME  how much time to request
```

The user may also request a different cluster, or different amount of memory or RAM when doing the resubmission.

An example call would be of the form
```
python resubmit.py --jobDir Run2_RPV_<unique_timestamp> --memory 75GB
```
where the user is resubmitting jobs for the job dir used above and is requesting a new memory of 35gb.

Again, jobs have not been submitted, so the user can navigate to the respective job dir and call:
```
qsub job_resubmit.pbs
```

# Plotting Input Variables

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
