# The Neural Network Framework

## Running at the Minnesota Supercomputing Institute

All running is done at the Minnesota Supercomputing Institute (MSI).
Connect to the Mangi (V100 nodes) or Agate (A100 nodes) cluster.
Please visit https://www.msi.umn.edu/ for more information on connecting.

### Setting Up the Working Area
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
conda install -c conda-forge shap
pip install uproot
pip install coffea
pip install mplhep==0.1.35
pip install pypi
pip install matplotlib==3.3.0
```

Get the analysis code from GitHub and `rsync` over the ROOT file NN inputs from the LPC

```
cd Train

git clone git@github.com:StealthStop/DeepESM.git
cd DeepESM

rsync -r <lpcuser>@cmslpc120.fnal.gov:/uscmst1b_scratch/lpc1/3DayLifetime/<some path> .

# Sets environment parameters 
source deepenv.sh
```

### Running Interactively

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
where 40 minutes of GPU time is requested on an interactive GPU node in the Mangi cluster (`k40`, for Agate cluster one would use `a40`) and 30 GB of RAM for loading events from disk.
Finally, the last argument provided is the entire `python` call to the executable to run, in this case `train.py`.
The `train.py` arguments are detailed below.
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

### Submitting Jobs to a Cluster

The most powerful use case is submitting NN training jobs in batch to the Mangi or Agate GPU clusters.
This is acheived using the `boboTrain.py` script, whose arguments are detailed as follows:
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

Running this command will generate a `Run2_RPV_<unique_timestamp>` folder in `./batch`.
No jobs have been submitted yet, but that is acheived by going to `./batch/Run2_RPV_<unique_timestamp` and running `qsub job_submit.pbs`.
The status of jobs can be checked using the command `qstat -a -f -M -u $USER`.

### Resubmitting Jobs
Occaisionally, some jobs may encounter a segmentation violation or problem (insufficient resource allocation) and stop running.
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

At this juncture, the user may also request a different cluster, or different amount of memory or RAM when doing the resubmission.

An example call would be of the form:
```
python resubmit.py --jobDir Run2_RPV_<unique_timestamp> --memory 75gb
```
where the user is resubmitting jobs for the job dir used above and is requesting a new memory of 75gb.

Again, jobs have not been submitted, so the user can navigate to the respective job dir and call `qsub job_resubmit.pbs`

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

### Parsing Job Output

A python script is provided `parseNNjobs.py` to grab plots for each neural network job and make a two slide summary, where two slide summaries are concatenated together into one set of LaTeX slides.
Some primitive logic is available to sort the trainings by a metric and currently the metric is a chi2 calculation comparing the ABCD-predicted number of events in A to the actual number of events in A based on fixed ABCD region boundaries.
Thus, the first NN jobs in the slides demonstrate the best ABCD closure.
The script expects a certain folder structure for the NN jobs of the form
```
<main_folder_with_tex_file>/<collection_of_NN_jobs>/<individual_NN_job>
```

An example call to use the script is
```
python parseNNjobs.py --inputDir main_folder_with_tex_file --subdir collection_of_NN_jobs --title Fancy title for slides
```

## Preparing Files for a Release
Once a NN training configuration has been chosen for use in the StealthStop analysis framework, a release needs to be made in `DeepESMCfg` (see that repository for further information on making a release).
A script is provided here to help make the `.cfg` and tar up the `.pb` ready for sending to `DeepESMCfg`.
The script has the arguments:

```
usage: usage: %prog [options] [-h] --year YEAR --path PATH --model MODEL --channel CHANNEL --version VERSION

optional arguments:
  -h, --help         show this help message and exit
  --year YEAR        year that NN is trained for
  --path PATH        Input dir with pb and json from training
  --model MODEL      signal model that NN is trained for
  --channel CHANNEL  channel that NN is trained for
  --version VERSION  versioning tag for local organization
```

where an example call to the script would be

```
python make_DoubleDisCo_cfgFile.py --year Run2 --path Output/atag_1l_MyFavRPV_lots_of_hyperparams/ --channel 1l --model RPV --version v1.2
```

This would make a folder `DoubleDisCo_Reg_1l_RPV_Run2_v1.2` that contains two `.cfg` and a `.tar`.
The `.cfg` are to be pushed to `DeepESMCfg`, while the tar should be uploaded when a new tag is made.
