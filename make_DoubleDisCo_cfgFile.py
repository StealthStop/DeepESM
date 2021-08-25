import os
import json
import argparse

debug = True

def print_db(input):

    if (debug):
        print(input)

def main():

    # -----------------------------------------
    # how we run this script:
    #   -- python make_DoubleDisCo_cfgFile.py --year 2016 --path Output/atag_0l_NJet6_Rpv550_batch_size_10000_bg_cor_lambda_1000.0_disc_comb_lambda_0.0_disc_lambda_10.0_epochs_15_gr_lambda_1.0_nHLayers_1_nHLayersD_1_nHLayersM_1_nNodes_100_nNodesD_1_nNodesM_100_reg_lambda_0.001_sg_cor_lambda_1000.0_/ --case _0l 
    #  -- python make_DoubleDisCo_cfgFile.py --year 2016 --path Output/atag_1l_Rpv550_twsysts_batch_size_10000_bg_cor_lambda_2000.0_disc_comb_lambda_0.0_disc_lambda_100.0_epochs_15_gr_lambda_2.0_nHLayers_1_nHLayersD_1_nHLayersM_1_nNodes_100_nNodesD_1_nNodesM_100_reg_lambda_0.001_sg_cor_lambda_50.0_/ --case _1l
    # ----------------------------------------- 
    usage  = "usage: %prog [options]"
    parser = argparse.ArgumentParser(usage)
    parser.add_argument("--year",  dest="year",  help="which year",             required=True)
    parser.add_argument("--path",  dest="path",  help="Input dir with histos",  required=True)
    parser.add_argument("--model", dest="model", help="for which signal model", required=True)
    parser.add_argument("--case",  dest="case",  help="0l, 1l, 2l",             required=True)
    parser.add_argument("--qcdcr", dest="qcdcr", help="Make config for QCD CR", action="store_true", default=False)
    args = parser.parse_args()

    # ------------------------------------------
    # create a separate folder for each cfg file
    # ------------------------------------------
    qcdCRtag = ""
    if args.qcdcr:
        qcdCRtag = "_NonIsoMuon_"

    release = "DoubleDisCo_Reg" + "_" + args.case + "_" + args.model + "_" + args.year + "_" + args.path.split("atag_")[-1].split("_batch_size_")[0]
    if not os.path.exists(release):
        os.makedirs(release)

    # --------------------------
    # put pb file to this folder
    # --------------------------
    os.system("cp %s/keras_frozen.pb %s" %(args.path,release))

    # -------------------
    # make tar file of pb
    # -------------------
    os.system("tar -C %s -czf %s/MVAFILES.tar.gz keras_frozen.pb" %(release,release))    
   
    # -------------
    # make cfg file
    # ------------- 
    with open (args.path + "config.json", "r") as c:
        cfg        = json.load(c)
        minNjet    = cfg["minNJetBin"]
        maxNjet    = cfg["maxNJetBin"]
        trainVars  = cfg["trainVars"]
        mask       = cfg["Mask"]
        mask_njet  = cfg["Mask_nJet"]

    #with open ("DoubleDisCo" + args.case + "_" + args.year + ".cfg", "w") as f:
    with open ("%s/DoubleDisCo_Reg%s.cfg"%(release,qcdCRtag[:-1]), "w") as f:   
        f.write("//Comment\n")
        f.write("/*another comment*/\n")
        f.write("\n")
        f.write("# Info for the DoubleDisCo_Reg training\n")
        f.write("Info\n")
        f.write("{\n")
        #f.write("   modelFile = \"DoubleDisCo%s_%s.pb\"\n" %(args.case, args.year))
        f.write("   modelFile = \"keras_frozen.pb\"\n")
        f.write("   inputOp = \"x\"\n")
        f.write("   outputOpVec[0] = \"Identity\"\n")
        f.write("   outputOpVec[1] = \"Identity_2\"\n")
        f.write("   outputCmVec[0] = 4\n")
        f.write("   outputCmVec[1] = 1\n")
        f.write("   year = \"%s\"\n" %(args.year))
        f.write("   name = \"%s%s_%s\"\n" %(qcdCRtag[1:],args.case,args.model))

        goodStr = "Good"
        if args.qcdcr:
            goodStr = ""

        if (args.case == "0l"):
            f.write("   nJetVar = \"N%s%sJets_pt45\"\n"%(qcdCRtag[1:-1],goodStr))
        else:
            f.write("   nJetVar = \"N%s%sJets_pt30\"\n"%(qcdCRtag[1:-1],goodStr))

        # Find the highest Njet bin that was masked from the list
        # If the highest masked Njet is higher than minNjet
        # then change minNjet put in cfg to be 1 greater than the highest masked Njet
        maxMaskedNjet = max(mask_njet)
        adjustedMinNjet = minNjet
        if maxMaskedNjet >= minNjet:
            adjustedMinNjet = maxMaskedNjet+1
            
        f.write("   minNjet = %d \n" %adjustedMinNjet)
        f.write("   maxNjet = %d \n" %maxNjet)

        i = 0
        for njet in range(minNjet,maxNjet+1):

            if (mask == True and njet in mask_njet): continue

            if (njet < 10):
                f.write("   binEdges[%d] = %s \n" %(i, cfg["c1_nJet_0%d"%njet]))
                i += 1 
                f.write("   binEdges[%d] = %s \n" %(i, cfg["c2_nJet_0%d"%njet]))
            else:
                f.write("   binEdges[%d] = %s \n" %(i, cfg["c1_nJet_%d"%njet]))
                i += 1 
                f.write("   binEdges[%d] = %s \n" %(i, cfg["c2_nJet_%d"%njet]))
            i += 1

        i = 0
        for var in trainVars:

            if args.qcdcr:
                var = var.replace("Jet",       "Jet%ss"%(qcdCRtag[1:-1]))
                var = var.replace("fwm",       "%ss_fwm"%(qcdCRtag[1:-1]))
                var = var.replace("jmt",       "%ss_jmt"%(qcdCRtag[1:-1]))
                var = var.replace("cm_",       "cm_%s"%(qcdCRtag[1:]))
                var = var.replace("trigger",   "%s"%(qcdCRtag[1:-1]))

            f.write("   mvaVar[%d] = \"%s\" \n" %(i,var))
            i += 1

        f.write("}\n")
        f.close()
                    
if __name__ == '__main__':
    main()
