import os
import json
import argparse

# Make a release of the Double DisCo NN including configuration and proto buffer files
class ReleaseMaker():

    def __init__(self, year, path, model, channel, version):

        self.year    = year
        self.path    = path
        self.model   = model
        self.channel = channel
        self.version = version

        self.debug = True

        # Create a separate folder for each release to include .pb and .cfg files
        self.release = "DoubleDisCo_Reg" + "_" + self.channel + "_" + self.model + "_" + self.year + "_" + self.version
        if not os.path.exists(self.release):
            os.makedirs(self.release)

        os.system("cp %s/keras_frozen.pb %s" %(self.path,self.release))
        os.system("tar -C %s -czf %s/MVAFILES.tar.gz keras_frozen.pb" %(self.release,self.release))    

    def print_db(self, input):
    
        if (self.debug):
            print(input)
    
    def writeCfg(self, doQCDCR=False):

        # Load the original json generated by train.py
        config = open(self.path + "/config.json", "r")
        cfg        = json.load(config)
        minNjet    = cfg["minNJetBin"]
        maxNjet    = cfg["maxNJetBin"]
        trainVars  = cfg["trainVars"]
        mask       = cfg["Mask"]
        mask_njet  = cfg["Mask_nJet"]
        scaleJetPt = cfg["scaleJetPt"]

        # When making a cfg for the NonIsoMuon we should use as tag
        qcdCRtag    = ""
        prependTag  = ""
        postpendTag = ""
        if doQCDCR:
            qcdCRtag    = "NonIsoMuon"
            prependTag  = "_NonIsoMuon"
            postpendTag = "NonIsoMuon_"

        f = open("%s/DoubleDisCo_Reg%s.cfg"%(self.release, prependTag), "w")

        f.write("# Info for the DoubleDisCo_Reg training\n")
        f.write("Info\n")
        f.write("{\n")
        f.write("    modelFile      = \"keras_frozen.pb\"\n")
        f.write("    inputOp        = \"x\"\n")
        f.write("    outputOpVec[0] = \"Identity\"\n")
        f.write("    outputOpVec[1] = \"Identity_2\"\n")
        f.write("    outputCmVec[0] = 4\n")
        f.write("    outputCmVec[1] = 1\n")
        f.write("    year           = \"%s\"\n" %(self.year))
        f.write("    name           = \"%s%s_%s\"\n" %(postpendTag, self.channel, self.model))

        # Here we use Good jets for 0L in the CR
        # So in that case, revert qcdCRtag to empty for the remainder of the function
        goodStr = "Good"
        if doQCDCR and self.channel != "0l":
            goodStr = qcdCRtag

        # GoodJets_pt30 used for both 0L and 1L
        f.write("    nJetVar        = \"N%sJets_pt30\"\n"%(goodStr))

        # Find the highest Njet bin that was masked from the list
        # If the highest masked Njet is higher than minNjet
        # then change minNjet put in cfg to be 1 greater than the highest masked Njet
        maxMaskedNjet = max(mask_njet)
        adjustedMinNjet = minNjet
        if maxMaskedNjet >= minNjet:
            adjustedMinNjet = maxMaskedNjet+1
            
        f.write("    minNjet        = %d \n" %(adjustedMinNjet))
        f.write("    maxNjet        = %d \n" %(maxNjet))
        f.write("\n")

        # Name the different signal and validation regions
        # Boundaries and disc edges are default values
        # corresponding to fixed ABCD and validation regions
        regions          = ["ABCD", "Val_BD", "Val_CD", "Val_D"]
        topBoundaries    = [1.0,    1.0,      0.4,      0.6]
        bottomBoundaries = [0.0,    0.0,      0.0,      0.0]
        rightBoundaries  = [1.0,    0.4,      1.0,      0.6]
        leftBoundaries   = [0.0,    0.0,      0.0,      0.0]

        disc1Edges       = [0.6,    0.2,      0.6,      0.3]
        disc2Edges       = [0.6,    0.6,      0.2,      0.3]

        iReg = 0
        for region in regions:
            f.write("    regions[%d]   = \"%s\"\n"%(iReg, region))
            iReg += 1
        f.write("\n")

        # Write out default definitions of the signal (ABCD) and validation (Val_BD, Val_CD, Val_D) regions
        # assuming (0.6, 0.6) edges for ABCD.
        # These values would necessarily be updated after running the validation framework and optimizing the edges and boundaries
        # for a given NN configuration
        wroteExample = False
        for region in regions:

            globalCount = 0
            iReg = regions.index(region)
            for njet in range(minNjet,maxNjet+1):

                if (mask == True and njet in mask_njet): continue

                if globalCount == 0 and not wroteExample:
                    f.write("    # An example of region definitions\n")
                    f.write("    # binEdges_aRegion[i]   = disc1edge\n")
                    f.write("    # binEdges_aRegion[i+1] = disc2edge\n")
                    f.write("    # binEdges_aRegion[i+2] = leftBoundary\n")
                    f.write("    # binEdges_aRegion[i+3] = rightBoundary\n")
                    f.write("    # binEdges_aRegion[i+4] = topBoundary\n")
                    f.write("    # binEdges_aRegion[i+5] = bottomBoundary\n\n")

                    wroteExample = True

                f.write("    # region = %s, Njets = %d\n"%(region, njet))
                f.write("    binEdges_%s[%d] = %.2f\n"%(region, globalCount, disc1Edges[iReg]))
                globalCount += 1

                f.write("    binEdges_%s[%d] = %.2f\n"%(region, globalCount, disc2Edges[iReg]))
                globalCount += 1

                f.write("    binEdges_%s[%d] = %.2f\n"%(region, globalCount, leftBoundaries[iReg]))
                globalCount += 1

                f.write("    binEdges_%s[%d] = %.2f\n"%(region, globalCount, rightBoundaries[iReg]))
                globalCount += 1

                f.write("    binEdges_%s[%d] = %.2f\n"%(region, globalCount, topBoundaries[iReg]))
                globalCount += 1

                f.write("    binEdges_%s[%d] = %.2f\n"%(region, globalCount, bottomBoundaries[iReg]))
                globalCount += 1

                f.write("\n")

        # Write out the variables used in the training
        iVar = 0
        for var in trainVars:

            # Need to add the string "NonIsoMuon" into the variable collections
            # for QCD CR config. This is not the case for 0L, which uses the nominal
            # collections 
            # Also, need to add the channel to the end of each variable name except for Stop variables
            if doQCDCR and self.channel != "0l":
                var = var.replace("Jet",         "Jet%ss"%(qcdCRtag)) \
                         .replace("fwm",         "%ss_fwm"%(qcdCRtag)) \
                         .replace("GoodLeptons", "%s"%(qcdCRtag)) \
                         .replace("lvMET",       "%ss_lvMET"%(qcdCRtag)) \
                         .replace("jmt",         "%ss_jmt"%(qcdCRtag)) \
                         .replace("Seed",        "Seed_%s"%(qcdCRtag)) \
                         .replace("trigger",     "%s"%(qcdCRtag))

                if scaleJetPt and ("Jet" in var or "Stop" in var):
                    var = var.replace("pt", "ptrHT")

            f.write("    inputVar[%d] = \"%s\" \n" %(iVar, var))
            iVar += 1

        f.write("}\n")
        f.close()
                    
if __name__ == '__main__':

    # ----------------------------------------------------------------------------------------------------------------------------------------------
    # how we run this script:
    #  -- python make_DoubleDisCo_cfgFile.py --year Run2 --path Output/atag_1l_MyFavRPV_lots_of_hyperparams/ --channel 1l --model RPV --version v1.2
    # ---------------------------------------------------------------------------------------------------------------------------------------------- 
    usage  = "usage: %prog [options]"
    parser = argparse.ArgumentParser(usage)
    parser.add_argument("--year",    dest="year",     help="year that NN is trained for",              required=True)
    parser.add_argument("--path",    dest="path",     help="Input dir with pb and json from training", required=True)
    parser.add_argument("--model",   dest="model",    help="signal model that NN is trained for",      required=True)
    parser.add_argument("--channel", dest="channel",  help="channel that NN is trained for",           required=True)
    parser.add_argument("--version", dest="version",  help="versioning tag for local organization",    required=True)
    args = parser.parse_args()

    theMaker = ReleaseMaker(args.year, args.path, args.model, args.channel, args.version)

    theMaker.writeCfg()
    theMaker.writeCfg(doQCDCR=True)
