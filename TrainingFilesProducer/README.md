# Produce DeepESM Training Files

Setting up this area
```
getSamplesCfg.sh
make -j4
```

Test locally before submitting
```
./makeTrainingTuples -D TTJets_SingleLept_Train -E 101 -R 10:1
```

Run on condor
```
python condorSubmit.py -d AllSignal,TT,TTJets,TTJets_Incl_Train,TTJets_SingleLept_Train -n 20 --output trainingFiles_VX
```
