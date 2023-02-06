#! /bin/env/python

import os, argparse, subprocess, json, shutil
from time import strftime

def generate_json(taskPath, config, jobid):

    with open("%s/temp%d.json"%(taskPath,jobid),'w') as f:
         json.dump(config, f)

def generate_qsub_config(taskPath, workPath, jobid, com, cluster, walltime, memory, replayAll, replayCom, evalYear):

    pbsPath = "%s/job_submit.pbs"%(taskPath) 
    g = open(pbsPath, "w")
    g.write("#!/bin/bash -l\n\n")
    g.write("#SBATCH --time=%s\n"%(walltime))
    #g.write("#SBATCH --gres=gpu:%s:1\n"%(cluster.partition("-")[0]))
    g.write("#SBATCH --gres=gpu:1\n")
    g.write("#SBATCH --mem=%s\n"%(memory))
    g.write("#SBATCH --tmp=24g\n")
    g.write("#SBATCH -p %s\n"%(cluster))
    g.write("#SBATCH --output=%s/out_%%A_%%a.txt\n"%(taskPath))
    g.write("#SBATCH -e %s/err_%%A_%%a.txt\n"%(taskPath))

    g.write("#SBATCH --array=1-%d\n"%(jobid))

    g.write("#SBATCH -A nstrobbe\n")
    #g.write("#SBATCH --mail-type=All\n")
    #g.write("#SBATCH --mail-user=cros0400@umn.edu\n\n")
    g.write("cd %s/\n\n"%(workPath))
    g.write("conda activate tf\n")
    g.write("source deepenv.sh\n\n")
    g.write("echo $PYTHONHASHSEED\n")
    g.write("echo $TF_CUDNN_DETERMINISTIC\n")
    g.write("echo $TF_DETERMINISTIC_OPS\n\n")
    g.write("cd %s/\n\n"%(taskPath))
    g.write(com + "\n\n")
    #g.write("module load texlive\n")
    g.write("./makeQuickLook.py\n")
    #g.write("pdflatex ./quickLook.tex\n")
    if replayAll:
        runPeriods = ["2016preVFP", "2016postVFP", "2017", "2018", "Run2"]

        for r in runPeriods:
            if evalYear == r: continue
            g.write(replayCom + " --evalYear {}\n".format(r))

    g.close()

    return pbsPath

if __name__ == '__main__':     

    parser = argparse.ArgumentParser()
    parser.add_argument("--trainBkgd",    dest="trainBkgd",    help="which bkgd to train on",                        default=["wsysts"], nargs="+")
    parser.add_argument("--trainModel",   dest="trainModel",   help="which sig to train on",                         default="RPV",      type=str)
    parser.add_argument("--evalBkgd",     dest="evalBkgd",     help="which bkgd to validate on",                     default=["pow"],    nargs="+")
    parser.add_argument("--evalModel",    dest="evalModel",    help="which model to validate on",                    default="RPV",      type=str)
    parser.add_argument("--trainMass",    dest="trainMass",    help="lower and upper mass range bounds",             default=[300,1400], nargs="+")
    parser.add_argument("--evalMass",     dest="evalMass",     help="which mass point to validate on",               default=550,        type=int)
    parser.add_argument("--tag",          dest="tag",          help="tag to use in output",                          default="TEST",     type=str)
    parser.add_argument("--bcorr",        dest="bcorr",        help="list of bcorr lambda values",                   default=[10.0],     nargs="+")
    parser.add_argument("--bcorrStart",   dest="bcorrStart",   help="list of starting epochs for disco",             default=[0],     nargs="+")
    parser.add_argument("--disc",         dest="disc",         help="list of disc lambda values",                    default=[2.0],     nargs="+")
    parser.add_argument("--discStart",    dest="discStart",    help="list of starting epochs for disc",             default=[0],     nargs="+")
    parser.add_argument("--abcd",         dest="abcd",         help="list of abcd lambda values",                    default=[5.0],     nargs="+")
    parser.add_argument("--abcdStart",    dest="abcdStart",    help="list of starting epochs for abcd",              default=[0],     nargs="+")
    parser.add_argument("--reg",          dest="reg",          help="list of reg lambda values",                     default=[0.001],    nargs="+")
    parser.add_argument("--nodes",        dest="nodes",        help="list of nodes values",                          default=[200],      nargs="+")
    parser.add_argument("--lrs",          dest="lrs",          help="learning rate",                                 default=[0.0001],   nargs="+")
    parser.add_argument("--batch",        dest="batch",        help="batch size",                                    default=[4096],   nargs="+")
    parser.add_argument("--factors",      dest="factors",      help="list of factors to multiply",                   default=[1.0],      nargs="+")
    parser.add_argument("--epochs",       dest="epochs",       help="how many epochs",                               default=[50],       nargs="+")
    parser.add_argument("--trainYear",    dest="trainYear",    help="which year(s) to train on",                     default="2016preVFP", type=str)
    parser.add_argument("--evalYear",     dest="evalYear",     help="which year to eval on",                         default="2016preVFP", type=str)
    parser.add_argument("--seed",         dest="seed",         help="which seed to init with",                       default="-1",     type=str)
    parser.add_argument("--channel",      dest="channel",      help="which decay channel",                           default="1l",         type=str)
    parser.add_argument("--noSubmit",     dest="noSubmit",     help="do not submit to cluster",                      default=False,        action="store_true")
    parser.add_argument("--cluster",      dest="cluster",      help="which cluster to run on",                       default="a100-8",     type=str)
    parser.add_argument("--memory",       dest="memory",       help="how much mem to request",                       default="35gb",       type=str)
    parser.add_argument("--walltime",     dest="walltime",     help="how much time to request",                      default="01:30:00",   type=str)
    parser.add_argument("--replayAll",    dest="replayAll",    help="Validation on all years",                       default=False,        action="store_true")
    parser.add_argument("--useJECs",      dest="useJECs",      help="use JEC/JER variation events",                  default=False,        action="store_true")
    parser.add_argument("--nJets",        dest="nJets",        help="Minimum number of jets",                        default=7,            type=int)
    parser.add_argument("--maskNjet",     dest="maskNjet",     help="mask Njet bin/bins in training",                default=[-1],  nargs="+", type=int )
    parser.add_argument("--procCats",     dest="procCats",     help="Balance batches bkg/sig",                       default=False, action="store_true" )
    parser.add_argument("--massCats",     dest="massCats",     help="Balance batches among masses",                  default=False, action="store_true" )
    parser.add_argument("--njetsCats",    dest="njetsCats",    help="Balance batches among njets",                   default=False, action="store_true" )
    parser.add_argument("--saveAndPrint", dest="saveAndPrint", help="Save model peanut butter",                      action="store_true", default=False)
    parser.add_argument("--inputs",       dest="inputs",       help="which inputs files to use",                     default="UL_NN_inputs", type=str)
    parser.add_argument("--output",       dest="output",       help="output directory name",                     default="Output", type=str)
    args = parser.parse_args()


    HOME = os.getenv("HOME")

    workDir = HOME + "/Train/DeepESM"

    taskDir = args.tag + "_" + strftime("%Y%m%d_%H%M%S")
    taskPath = workDir + "/batch/%s"%(taskDir)

    os.makedirs(taskPath)

    jobid = 1
    for bcorr in args.bcorr:
        for bcorrStart in args.bcorrStart:
            for disc in args.disc:
                for discStart in args.discStart:
                    for reg in args.reg:
                        for factor in args.factors:
                            for nodes in args.nodes:
                                for abcd in args.abcd:
                                    for abcdStart in args.abcdStart:
                                        for tBkgd in args.trainBkgd:
                                            for vBkgd in args.evalBkgd:
                                                for epoch in args.epochs:
                                                    for lr in args.lrs:
                                                        for batch in args.batch:

                                                            #if float(bcorr) == 0.0 and float(abcd) == 0.0: continue

                                                            config = {"case" : 0, "atag" : "%s_v%s"%(args.tag,vBkgd), "abcd_close_lambda" : float(factor)*float(abcd), "disc_lambda": float(factor)*float(disc), "mass_reg_lambda": float(reg), "bkg_disco_lambda": float(factor)*float(bcorr), "input_nodes": int(nodes), "disc_nodes": int(nodes), "mass_reg_nodes": int(nodes), "input_layers": 1, "disc_layers":1, "mass_reg_layers":1, "dropout":0.3, "batch": int(batch), "epochs": int(epoch), "disco_start": int(bcorrStart), "abcd_start": int(abcdStart), "disc_start": int(discStart), "lr": float(lr)}

                                                            #Training all at once
                                                            generate_json(taskPath, config, jobid)

                                                            jobid += 1

    # We incremented one too far
    jobid -= 1

    balanceStr = ""
    if args.procCats:
        balanceStr += " --procCats"
    if args.njetsCats:
        balanceStr += " --njetsCats"
    if args.massCats:
        balanceStr += " --massCats"

    maskNjet = ""
    for Njet in args.maskNjet:
        maskNjet += "%d "%(Njet)
    
    jecStr = ""
    if args.useJECs:
        jecStr += " --useJECs"

    saveStr = ""
    if args.saveAndPrint:
        saveStr += " --saveAndPrint"

    jetStr = ""
    if args.channel == "1l":
        jetStr += " --nJets %d"%(args.nJets)

    command = "python train.py --json temp${SLURM_ARRAY_TASK_ID}.json %s %s %s %s --maskNjet %s --minMass %d --maxMass %d --evalMass %d --trainModel %s --evalModel %s --evalYear %s --trainYear %s --seed %s --tree myMiniTree_%s --outputDir %s --scaleJetPt"%(jetStr,saveStr,jecStr,balanceStr,maskNjet,int(args.trainMass[0]),int(args.trainMass[1]),int(args.evalMass),args.trainModel,args.evalModel,args.evalYear,args.trainYear,args.seed,args.channel,args.output)

    replayCommand = "python train.py --replay --evalModel %s --json temp${SLURM_ARRAY_TASK_ID}.json --outputDir %s --tree myMiniTree_%s --evalMass %d --scaleJetPt" % (args.evalModel, args.output, args.channel, args.evalMass)

    pbsPath = generate_qsub_config(taskPath, workDir, jobid, command, args.cluster, args.walltime, args.memory, args.replayAll, replayCommand, args.evalYear) 

    shutil.copy2("%s/train.py"%(workDir),           "%s/train.py"%(taskPath))
    shutil.copy2("%s/DataLoader.py"%(workDir),      "%s/DataLoader.py"%(taskPath))
    shutil.copy2("%s/Models.py"%(workDir),          "%s/Models.py"%(taskPath))
    shutil.copy2("%s/Validation.py"%(workDir),      "%s/Validation.py"%(taskPath))
    shutil.copy2("%s/Correlation.py"%(workDir),     "%s/Correlation.py"%(taskPath))
    shutil.copy2("%s/CustomOptimizer.py"%(workDir), "%s/CustomOptimizer.py"%(taskPath))
    shutil.copy2("%s/MeanShiftTF.py"%(workDir),     "%s/MeanShiftTF.py"%(taskPath))
    shutil.copy2("%s/CustomCallback.py"%(workDir),     "%s/CustomCallback.py"%(taskPath))
    #if args.channel != "2l":
    shutil.copy2("%s/utils/makeQuickLook.py"%(workDir),     "%s/makeQuickLook.py"%(taskPath))
    #else:
    #    shutil.copy2("%s/utils/makeQuickLook_2l.py"%(workDir),     "%s/makeQuickLook.py"%(taskPath))

    USER = os.getenv("USER")

    os.system("ln -s /home/nstrobbe/shared/%s %s/NN_inputs"%(args.inputs, taskPath))

    if not args.noSubmit:
        subprocess.call(["qsub", pbsPath])
