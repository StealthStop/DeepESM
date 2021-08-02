import ROOT
import subprocess as sb
import argparse
import os
import glob

bkgs = ["TTJets_Incl", "TT", "TT_isrUp", "TT_isrDown", "TT_fsrUp", "TT_fsrDown",
        "TT_erdOn", "TT_hdampUp", "TT_hdampDown", "TT_underlyingEvtUp", "TT_underlyingEvtDown"
       ]

sigmods = ["RPV", "StealthSYY"]

sigmasses = list(range(300, 1450, 50))

splits = ["Train", "Val", "Test"]

if __name__ == "__main__":

    usage = "usage: %prog [options]"
    parser = argparse.ArgumentParser(usage)
    parser.add_argument("--outputdir", dest="outputdir", help="Directory to store hadded output",    required=True)
    parser.add_argument("--inputdir",  dest="inputdir",  help="Directory of unhadded input",         required=True)
    parser.add_argument("--year",      dest="year",      help="Which year to hadd",                  required=True)
    parser.add_argument("--dryrun",    dest="dryrun",    help="Print what will happen, don't do it", action="store_true", default=False)
    args = parser.parse_args()

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    inputdir = ""
    if args.inputdir[-1] == "/":
        inputdir = args.inputdir[:-1]
    else:
        inputdir = args.inputdir

    # Hadd the background files together
    for bkg in bkgs:
        for split in splits:
            command = ["hadd", "%s/MyAnalysis_%s_%s_%s.root"%(args.outputdir,args.year,bkg,split)] + glob.glob("%s/%s_%s/*_%s_?_%s.root"%(inputdir,args.year,bkg,bkg,split)) \
                                                                           + glob.glob("%s/%s_%s/*_%s_??_%s.root"%(inputdir,args.year,bkg,bkg,split)) \
                                                                           + glob.glob("%s/%s_%s/*_%s_???_%s.root"%(inputdir,args.year,bkg,bkg,split))

            print("Executing command: \"%s\""%(" ".join(command)))
            if not args.dryrun:
                p = sb.Popen(command)
                p.wait()

    for sigmod in sigmods:
        for sigmass in sigmasses:
            for split in splits:

                command = ["hadd", "%s/MyAnalysis_%s_%s_2t6j_mStop-%s_%s.root"%(args.outputdir,args.year,sigmod,sigmass,split)] + glob.glob("%s/%s_AllSignal/*_%s_2t6j_mStop-%s_*_%s.root"%(inputdir,args.year,sigmod,sigmass,split))

                print("Executing command: \"%s\""%(" ".join(command)))
                if not args.dryrun:
                    p = sb.Popen(command)
                    p.wait()
