#!/cvmfs/cms.cern.ch/slc6_amd64_gcc491/cms/cmssw/CMSSW_7_4_8/external/slc6_amd64_gcc491/bin/python
####!${SRT_CMSSW_RELEASE_BASE_SCRAMRTDEL}/external/${SCRAM_ARCH}/bin/python

import sys
from os import system, environ
sys.path = [environ["CMSSW_BASE"] + "/src/SusyAnaTools/Tools/condor/",] + sys.path

from samples import SampleCollection
import optparse 
import subprocess
from pyxrootd import client

filestoTransferGTP = [environ["CMSSW_BASE"] + "/src/DeepESM/TrainingFilesProducer/makeTrainingTuples",
                      environ["CMSSW_BASE"] + "/src/TopTagger/TopTagger/test/libTopTagger.so", #Might need to fix this for new working path
                      environ["CMSSW_BASE"] + "/src/TopTagger/Tools/TopTaggerClusterOnly.cfg", #Might need to fix this for new working path
                      "/uscms_data/d3/pastika/zinv/dev/CMSSW_7_4_8/src/opencv/lib/libopencv_core.so.3.1",
                      "/uscms_data/d3/pastika/zinv/dev/CMSSW_7_4_8/src/opencv/lib/libopencv_ml.so.3.1",
                      environ["CMSSW_BASE"] + "/src/DeepESM/TrainingFilesProducer/sampleSets.cfg",
                      environ["CMSSW_BASE"] + "/src/DeepESM/TrainingFilesProducer/sampleCollections.cfg",
                      ]


#go make top plots!
submitFileGTP = """universe = vanilla
Executable = $ENV(CMSSW_BASE)/src/DeepESM/TrainingFilesProducer/condor/goMakeTrainingTuples.sh
request_memory = 5100
Should_Transfer_Files = YES
WhenToTransferOutput = ON_EXIT
Transfer_Input_Files = $ENV(CMSSW_BASE)/src/DeepESM/TrainingFilesProducer/condor/goMakeTrainingTuples.sh,$ENV(CMSSW_BASE)/src/DeepESM/TrainingFilesProducer/condor/gtp.tar.gz,$ENV(CMSSW_BASE)/src/DeepESM/TrainingFilesProducer/condor/$ENV(CMSSW_VERSION).tar.gz 
Output = logs/makePlots_$(Process).stdout
Error = logs/makePlots_$(Process).stderr
Log = logs/makePlots_$(Process).log
notify_user = ${LOGNAME}@FNAL.GOV
x509userproxy = $ENV(X509_USER_PROXY)


"""


parser = optparse.OptionParser("usage: %prog [options]\n")

parser.add_option ('-n',  dest='numfile', type='int', default = 5, help="number of files per job")
parser.add_option ('-d',  dest='datasets', type='string', default = '', help="List of datasets 'ZJetsToNuNu,DYJetsToLL'")
parser.add_option ('-l',  dest='dataCollections', action='store_true', default = False, help="List all datacollections")
parser.add_option ('-L',  dest='dataCollectionslong', action='store_true', default = False, help="List all datacollections and sub collections")
parser.add_option ('-r',  dest='refLumi', type='string', default = None, help="Data collection to define lumi (uses default lumi if no reference data collection is defined)")
parser.add_option ('-c',  dest='noSubmit', action='store_true', default = False, help="Do not submit jobs.  Only create condor_submit.txt.")
parser.add_option ('--output', dest='outputDir', type='string', default = 'defaultPath', help="Name of directory where output of each condor job goes in cmadrid eos area")

options, args = parser.parse_args()

submitFile = ""
exeName = ""

def makeExeAndFriendsTarrball(filestoTransfer, fname):
    if not options.dataCollections and not options.dataCollectionslong:
        #WORLDSWORSESOLUTIONTOAPROBLEM
        system("mkdir -p WORLDSWORSESOLUTIONTOAPROBLEM")
        for fn in filestoTransfer:
            system("cd WORLDSWORSESOLUTIONTOAPROBLEM; ln -s %s"%fn)
        
        tarallinputs = "tar czvf %s.tar.gz WORLDSWORSESOLUTIONTOAPROBLEM --dereference"%fname
        print tarallinputs
        system(tarallinputs)
        system("rm -r WORLDSWORSESOLUTIONTOAPROBLEM")


if not options.dataCollections and not options.dataCollectionslong:
    system("tar --exclude-caches-all --exclude-vcs -zcf ${CMSSW_VERSION}.tar.gz -C ${CMSSW_BASE}/.. ${CMSSW_VERSION} --exclude=src --exclude=tmp")


exeName = "makeTrainingTuples"
submitFile = submitFileGTP
makeExeAndFriendsTarrball(filestoTransferGTP, "gtp")

nFilesPerJob = options.numfile

fileParts = [submitFile]
sc = SampleCollection("../sampleSets.cfg", "../sampleCollections.cfg")
datasets = []

if options.dataCollections or options.dataCollectionslong:
    scl = sc.sampleCollectionList()
    for sampleCollection in scl:
        sl = sc.sampleList(sampleCollection)
        print sampleCollection
        if options.dataCollectionslong:
            sys.stdout.write("\t")
            for sample in sl:
                sys.stdout.write("%s  "%sample[1])
            print ""
            print ""
    exit(0)

if options.datasets:
    datasets = options.datasets.split(',')
else:
    print "No dataset specified"
    exit(0)

lumis = sc.sampleCollectionLumiList()
lumi = sc.getFixedLumi()
if options.refLumi != None:
    lumi = lumis[options.refLumi]
    print "Normalizing to %s pb-1" % (lumi)

for ds in datasets:
    ds = ds.strip()

    print ds
    for s, n in sc.sampleList(ds):
        print "\t%s"%n
        f = open(s)
        if not f == None:
            count = 0
            for l in f:
                if '.root' in l and not 'failed' in l:
                    count = count + 1
            for startFileNum in xrange(0, count, nFilesPerJob):
                fileParts.append("Arguments = %s $ENV(CMSSW_VERSION) %i %i %f %s %s\n"%(n, nFilesPerJob, startFileNum, lumi, s, options.outputDir))
                fileParts.append("Output = logs/%s_%s_%i.stdout\n"%(exeName, n, startFileNum))
                fileParts.append("Error = logs/%s_%s_%i.stderr\n"%(exeName, n, startFileNum))
                fileParts.append("Log = logs/%s_%s_%i.log\n"%(exeName, n, startFileNum))
                fileParts.append("Queue\n\n")

            f.close()

fout = open("condor_submit.txt", "w")
fout.write(''.join(fileParts))
fout.close()

#Make output path on eos
fs = client.FileSystem("root://cmseos.fnal.gov")
eosOutputPath = "/store/user/cmadrid/trainingTuples/%s" % options.outputDir
output = fs.dirlist(eosOutputPath)
if output[0]['ok']:
    print "Output directory \" %s \" already exist on EOS: delete or use a different output directory" % eosOutputPath
    exit(0)
else:
    fs.mkdir(eosOutputPath)
    print "Made directory on EOS \" %s \" " % eosOutputPath 

if not options.noSubmit: 
    system('mkdir -p logs')
    system("echo 'condor_submit condor_submit.txt'")
    system('condor_submit condor_submit.txt')
