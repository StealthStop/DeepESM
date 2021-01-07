#! /bin/env/python

import ROOT, random, os, argparse, glob, string

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat("")
ROOT.gStyle.SetPaintTextFormat("3.2f")
ROOT.gStyle.SetFrameLineWidth(2)
ROOT.gStyle.SetEndErrorSize(0)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

def randomString():
    randStr = ''.join([random.choice(string.ascii_letters + string.digits) for n in xrange(8)])
    return randStr

def makeNDhisto(evtsTree, variables, ranges, labels, cut):
    h = 0; hName = "h_%s_%s"%(evtsTree.GetTitle(), randomString())
    h = ROOT.TH1F(hName, ";%s"%(labels[0]), ranges[0], ranges[1], ranges[2])
    evtsTree.Draw("%s>>%s"%(variables[0],hName), cut)
    h = ROOT.gDirectory.Get(hName)
    h.Sumw2()

    return h

usage = "usage: %prog [options]"
parser = argparse.ArgumentParser(usage)
parser.add_argument("--approved", dest="approved", help="Plot is approved", action="store_true", default=False) 
parser.add_argument("--path",     dest="path",     help="Path to ntuples", default="NULL", required=True)
parser.add_argument("--tree",     dest="tree",     help="TTree name to use", default="myMiniTree")
parser.add_argument("--year",     dest="year",     help="which year", required=True)
parser.add_argument("--mass1",    dest="mass1",    help="mass 1 to show", default="400")
parser.add_argument("--mass2",    dest="mass2",    help="mass 2 to show", default="700")
parser.add_argument("--model1",   dest="model1",   help="model 1 to show", default="RPV")
parser.add_argument("--model2",   dest="model2",   help="model 2 to show", default="StealthSYY")
args = parser.parse_args()

mass1 = args.mass1; mass2 = args.mass2
model1 = args.model1; model2 = args.model2
inputDir = args.path; tree = args.tree

year = args.year
if   year == "Combo": yearStr = "*"
elif year == "2018":  yearStr = year + "pre"
else:                 yearStr = year

model1Aux = ""; model2Aux = ""
if "RPV" in model1 or "SYY" in model1:
    model1Aux = "2t6j"
elif "SHH" in model1:
    model1Aux = "2t4b"

if "RPV" in model2 or "SYY" in model2:
    model2Aux = "SYY"
elif "SHH" in model2:
    model2Aux = "2t4b"

outpath = "./plots/NNinputs/%s/"%(year.replace("pre", ""))
if not os.path.exists(outpath): os.makedirs(outpath)

ttC   = ROOT.TChain(tree)
sig1C = ROOT.TChain(tree)
sig2C = ROOT.TChain(tree)

ttP   = "%s/MyAnalysis_%s_TT*.root"%(inputDir, yearStr)
sig1P = "%s/MyAnalysis_%s_%s_%s_mStop-%s_*.root"%(inputDir, yearStr, model1, model1Aux, args.mass1)
sig2P = "%s/MyAnalysis_%s_%s_%s_mStop-%s_*.root"%(inputDir, yearStr, model2, model1Aux, args.mass2)

proc = 0; ttFs = []; sig1Fs = []; sig2Fs = []

ttFs   = glob.glob(ttP)
sig1Fs = glob.glob(sig1P)
sig2Fs = glob.glob(sig2P)

for item in ttFs:   ttC.AddFile(item)
for item in sig1Fs: sig1C.AddFile(item)
for item in sig2Fs: sig2C.AddFile(item)

cutDict = {"NGoodJets_pt30_double" : {"==" : list(xrange(7,12)) + [-1], ">=" : [12]}}

histDict = {"GoodLeptons_pt_@"      : {"fudge" : 1.3, "logY" : False, "range" : [1,2],              "X" : {"title" : "Lepton @ p_{T} [GeV]",              "bins" : 80, "min" : 0, "max" : 360}},
            "GoodLeptons_phi_@"     : {"fudge" : 1.3, "logY" : False, "range" : [1,2],              "X" : {"title" : "Lepton @ #phi",                     "bins" : 64, "min" : -4, "max" : 4}},
            "GoodLeptons_m_@"       : {"fudge" : 1.3, "logY" : True,  "range" : [1,2],              "X" : {"title" : "Lepton @ Mass [GeV]",               "bins" : 72, "min" : 0, "max" : 0.12}},
            "GoodLeptons_eta_@"     : {"fudge" : 1.3, "logY" : False, "range" : [1,2],              "X" : {"title" : "Lepton @ #eta",                     "bins" : 80, "min" : -6, "max" : 6}},
            "Jet_pt_@"              : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,13)), "X" : {"title" : "Leading Jet p_{T} [GeV]",           "bins" : 72, "min" : 0, "max" : 1500}},
            "Jet_phi_@"             : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,13)), "X" : {"title" : "Jet #phi",                          "bins" : 64, "min" : -4, "max" : 4}},
            "Jet_m_@"               : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,13)), "X" : {"title" : "Jet Mass [GeV]",                    "bins" : 80, "min" : 0, "max" : 40}},
            "Jet_eta_@"             : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,13)), "X" : {"title" : "Jet #eta",                          "bins" : 80, "min" : -6, "max" : 6}},
            "Jet_axismajor_@"       : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,13)), "X" : {"title" : "Jet Axismajor",                     "bins" : 80, "min" : 0, "max" : 0.4}}, 
            "Jet_axisminor_@"       : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,13)), "X" : {"title" : "Jet Axisminor",                     "bins" : 80, "min" : 0, "max" : 0.4}}, 
            "Jet_dcsv_@"            : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,13)), "X" : {"title" : "Jet DeepCSV",                       "bins" : 80, "min" : -2.5, "max" : 1.0}}, 
            "Jet_multiplicity_@"    : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,13)), "X" : {"title" : "Jet Multiplicity",                  "bins" : 41, "min" : -0.5, "max" : 40.5}}, 
            "Jet_ptD_@"             : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,13)), "X" : {"title" : "Jet p_{T,D}",                       "bins" : 80, "min" : 0, "max" : 1}}, 
            "JetsAK8Cands_Pruned_@" : {"fudge" : 1.3, "logY" : True,  "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ Pruned Mass [GeV]",       "bins" : 80, "min" : 0, "max" : 300}}, 
            "JetsAK8Cands_SDM_@"    : {"fudge" : 1.3, "logY" : True,  "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ Softdrop Mass [GeV]",     "bins" : 80, "min" : 0, "max" : 300}}, 
            "JetsAK8Cands_m_@"      : {"fudge" : 1.3, "logY" : True,  "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ Mass [GeV]",              "bins" : 80, "min" : 0, "max" : 300}}, 
            "JetsAK8Cands_phi_@"    : {"fudge" : 1.3, "logY" : True,  "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ #phi",                    "bins" : 64, "min" : -4, "max" : 4}},
            "JetsAK8Cands_eta_@"    : {"fudge" : 1.3, "logY" : True,  "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ #eta",                    "bins" : 80, "min" : -6, "max" : 6}},
            "JetsAK8Cands_T21_@"    : {"fudge" : 1.3, "logY" : True,  "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ #tau_{21}",               "bins" : 80, "min" : 0, "max" : 1}}, 
            "JetsAK8Cands_T32_@"    : {"fudge" : 1.3, "logY" : True,  "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ #tau_{32}",               "bins" : 80, "min" : 0, "max" : 1}}, 
            "JetsAK8Cands_pt_@"     : {"fudge" : 1.3, "logY" : True,  "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ p_{T}",                   "bins" : 80, "min" : 0, "max" : 1440}}, 
            "JetsAK8Cands_Pruned_@" : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ Pruned Mass [GeV]",       "bins" : 80, "min" : 0.1, "max" : 300}}, 
            "JetsAK8Cands_SDM_@"    : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ Softdrop Mass [GeV]",     "bins" : 80, "min" : 0.1, "max" : 300}}, 
            "JetsAK8Cands_m_@"      : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ Mass [GeV]",              "bins" : 80, "min" : 0.1, "max" : 300}}, 
            "JetsAK8Cands_phi_@"    : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ #phi",                    "bins" : 64, "min" : -4, "max" : 4}},
            "JetsAK8Cands_eta_@"    : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ #eta",                    "bins" : 80, "min" : -6, "max" : 6}},
            "JetsAK8Cands_T21_@"    : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ #tau_{21}",               "bins" : 80, "min" : 0.1, "max" : 1}}, 
            "JetsAK8Cands_T32_@"    : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ #tau_{32}",               "bins" : 80, "min" : 0.1, "max" : 1}}, 
            "JetsAK8Cands_pt_@"     : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(1,6)),  "X" : {"title" : "AK8 Jet @ p_{T}",                   "bins" : 80, "min" : 0.1, "max" : 1440}}, 
            "Mbl"                   : {"fudge" : 1.3, "logY" : False, "range" : [-1],               "X" : {"title" : "M_{b,l} [GeV]",                     "bins" : 60, "min" : 30, "max" : 270}}, 
            "NGoodBJets_pt30"       : {"fudge" : 1.3, "logY" : False, "range" : [-1],               "X" : {"title" : "N_{b jets}",                        "bins" : 9,  "min" : -0.5, "max" : 8.5}}, 
            "NGoodJets_pt30_double" : {"fudge" : 20,  "logY" : True,  "range" : [-1],               "X" : {"title" : "N_{jets}",                          "bins" : 6,  "min" : 6.5, "max" : 12.5}}, 
            "deepESM_val"           : {"fudge" : 1.3, "logY" : False, "range" : [-1],               "X" : {"title" : "Legacy S_{NN}",                     "bins" : 80, "min" : 0, "max" : 1}}, 
            "deepESM_valReg"        : {"fudge" : 1.3, "logY" : False, "range" : [-1],               "X" : {"title" : "M_{NN} [GeV]",                      "bins" : 80, "min" : 0, "max" : 1440}}, 
            "fwm@_top6"             : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(2,11)), "X" : {"title" : "Fox-Wolfram Moment @",              "bins" : 80, "min" : 0, "max" : 1}}, 
            "jmt_ev@_top6"          : {"fudge" : 1.3, "logY" : False, "range" : list(xrange(0,3)),  "X" : {"title" : "Jet E-#vec{p} Tensor Eigenvalue @", "bins" : 80, "min" : 0, "max" : 1}},
            "lvMET_cm_pt"           : {"fudge" : 1.3, "logY" : False, "range" : [-1],               "X" : {"title" : "|#vec{E}_{T}^{miss}| [GeV]",        "bins" : 80, "min" : 0, "max" : 500}},
            "lvMET_cm_phi"          : {"fudge" : 1.3, "logY" : False, "range" : [-1],               "X" : {"title" : "#vec{E}_{T}^{miss} #phi",           "bins" : 64, "min" : -4, "max" : 4}},
            "lvMET_cm_m"            : {"fudge" : 1.3, "logY" : False, "range" : [-1],               "X" : {"title" : "#vec{E}_{T}^{miss} Mass [GeV]",     "bins" : 80, "min" : 0, "max" : 1}},
            "lvMET_cm_eta"          : {"fudge" : 1.3, "logY" : False, "range" : [-1],               "X" : {"title" : "#vec{E}_{T}^{miss} #eta",           "bins" : 80, "min" : -6, "max" : 6}},
            "stop@_PtRank_1l_mass"  : {"fudge" : 1.3, "logY" : False, "range" : [1,2],              "X" : {"title" : "Stop@ Mass (p_{T}-Ranked) [GeV]",   "bins" : 80, "min" : 0, "max" : 1500}},
            "HT_trigger_pt30"       : {"fudge" : 70,  "logY" : True,  "range" : [-1],               "X" : {"title" : "H_{T} [GeV]",                       "bins" : 64, "min" : 0, "max" : 5000}},
}

for name, opt in histDict.iteritems():
    for cutvar, operatorDict in cutDict.iteritems():
        for operator, vals in operatorDict.iteritems(): 
            for val in vals:

                if val != -1: cutStr = "%s%s%d"%(cutvar,operator,val)

                plotStr = ""
                if val != -1: plotStr = "_Njets%d"%(val)

                tth = 0; sig1h = 0; sig2h = 0
                for i in opt["range"]:

                    newName = name.replace("@", "%d"%(i))

                    tth   = makeNDhisto(ttC,   [newName], [opt["X"]["bins"], opt["X"]["min"], opt["X"]["max"]], [opt["X"]["title"].replace("@", "%d"%(i)), "A.U."], "%s"%(cutStr))
                    sig1h = makeNDhisto(sig1C, [newName], [opt["X"]["bins"], opt["X"]["min"], opt["X"]["max"]], [opt["X"]["title"].replace("@", "%d"%(i)), "A.U."], "%s"%(cutStr))
                    sig2h = makeNDhisto(sig2C, [newName], [opt["X"]["bins"], opt["X"]["min"], opt["X"]["max"]], [opt["X"]["title"].replace("@", "%d"%(i)), "A.U."], "%s"%(cutStr))

                    c = ROOT.TCanvas("%s%s_c"%(newName,plotStr), "%s%s_c"%(newName,plotStr), 2400, 2400)

                    sig1col = "#D325D3"
                    sig2col = "#FFA851"
                    ttcol   = "#9999FF"
                    ttcol2  = "#010199"

                    if tth.Integral() > 0.0:   tth.Scale(1.0/tth.Integral())
                    if sig1h.Integral() > 0.0: sig1h.Scale(1.0/sig1h.Integral())
                    if sig2h.Integral() > 0.0: sig2h.Scale(1.0/sig2h.Integral())

                    maxBinTT = tth.GetMaximumBin();   theMaxTT = tth.GetBinContent(maxBinTT)
                    maxBinSG2 = sig2h.GetMaximumBin(); theMaxSG2 = sig2h.GetBinContent(maxBinSG2)
                    maxBinSG1 = sig1h.GetMaximumBin(); theMaxSG1 = sig1h.GetBinContent(maxBinSG1)

                    theMax = max([theMaxTT, theMaxSG1, theMaxSG2])
                    tth.SetMaximum(theMax*opt["fudge"])

                    if opt["logY"]: c.SetLogy()

                    tth.SetFillColorAlpha(ROOT.TColor.GetColor(ttcol), 1.0)
                    sig1h.SetFillColor(ROOT.TColor.GetColor(sig1col))
                    sig2h.SetFillColor(ROOT.TColor.GetColor(sig2col))

                    tth.SetLineColor(ROOT.TColor.GetColor(ttcol2))
                    sig1h.SetLineColor(ROOT.TColor.GetColor(sig1col))
                    sig2h.SetLineColor(ROOT.TColor.GetColor(sig2col))

                    tth.SetMarkerColor(ROOT.TColor.GetColor(ttcol))
                    sig1h.SetMarkerColor(ROOT.TColor.GetColor(sig1col))
                    sig2h.SetMarkerColor(ROOT.TColor.GetColor(sig2col))

                    lw = 2
                    tth.SetLineWidth(lw)
                    sig1h.SetLineWidth(lw)
                    sig2h.SetLineWidth(lw)

                    ms = 0
                    tth.SetMarkerSize(ms)
                    sig1h.SetMarkerSize(ms)
                    sig2h.SetMarkerSize(ms)

                    sig1h.SetFillStyle(3004)
                    sig2h.SetFillStyle(3005)

                    ROOT.gPad.SetTopMargin(0.06)
                    ROOT.gPad.SetLeftMargin(0.14)
                    ROOT.gPad.SetBottomMargin(0.12)
                    ROOT.gPad.SetRightMargin(0.04)
                    
                    iamLegend = ROOT.TLegend(ROOT.gPad.GetLeftMargin() + 0.05, 0.77, 1 - ROOT.gPad.GetRightMargin() - 0.05, 0.87)
                    
                    iamLegend.SetTextSize(0.026)
                    iamLegend.SetNColumns(3)
                    iamLegend.SetBorderSize(2)
                    
                    iamLegend.AddEntry(sig1h, "%s m_{ #tilde{t}} = %s GeV"%(args.model1, mass1), "F")
                    iamLegend.AddEntry(sig2h, "%s m_{ #tilde{t}} = %s GeV"%(args.model2, mass2), "F")
                    iamLegend.AddEntry(tth,   "t#bar{t}", "F")

                    tth.SetTitle("")
                    tth.GetYaxis().SetTitle("A.U.")
                    tth.GetYaxis().SetTitleSize(0.05)
                    tth.GetXaxis().SetTitleSize(0.05)
                    tth.GetYaxis().SetLabelSize(0.04)
                    tth.GetXaxis().SetLabelSize(0.04)
                    tth.GetYaxis().SetTitleOffset(1.4)
                    tth.GetXaxis().SetTitleOffset(1.1)

                    tth.Draw("HIST")
                    sig1h.Draw("HIST SAME")
                    sig2h.Draw("HIST SAME")
                    iamLegend.Draw("SAME")

                    mark = ROOT.TLatex()
                    mark.SetNDC(True)

                    mark.SetTextAlign(11);
                    mark.SetTextSize(0.045);
                    mark.SetTextFont(61);
                    mark.DrawLatex(ROOT.gPad.GetLeftMargin() + 0.03, 1 - (ROOT.gPad.GetTopMargin() + 0.05), "CMS")
                    mark.SetTextFont(52);
                    if args.approved:
                        mark.DrawLatex(ROOT.gPad.GetLeftMargin() + 0.132, 1 - (ROOT.gPad.GetTopMargin() + 0.05), "Simulation")
                    else:
                        mark.DrawLatex(ROOT.gPad.GetLeftMargin() + 0.132, 1 - (ROOT.gPad.GetTopMargin() + 0.05), "Simulation Preliminary")

                    mark.SetTextFont(62)
                    if val != -1: mark.DrawLatex(1.0 - ROOT.gPad.GetRightMargin() - 0.2, 1 - (ROOT.gPad.GetTopMargin() + 0.05), "N_{jets} %s %d"%(operator.replace("==", "=").replace(">=", "#geq"), val))

                    mark.SetTextFont(42)
                    mark.SetTextAlign(31)
                    mark.DrawLatex(1 - ROOT.gPad.GetRightMargin(), 1 - (ROOT.gPad.GetTopMargin() - 0.015), "%s (13 TeV)"%(year.replace("Combo", "Run2")))

                    logStr = ""
                    if opt["logY"]: logStr = "_log"
  
                    if args.approved: c.SaveAs("%s/%s%s%s.pdf"%(outpath,newName,plotStr,logStr))
                    else:             c.SaveAs("%s/%s%s%s_prelim.pdf"%(outpath,newName,plotStr,logStr))
