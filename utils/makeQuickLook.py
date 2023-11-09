#!/usr/bin/python


from glob import glob
import os

def get_outdirs():
    
    cwd = os.getcwd()
    
    output = os.path.join(cwd, "Output/Output/*/Run2")

    return glob(output)
    

def beamer_header(f):

    f.write('''\\documentclass{beamer}
\\title{NN Quick Look}
\\author{Bryan Crossman}
\\institute{Overleaf}
\\date{2021}

\\begin{document}
''')

def write_slide(f, o, flist):
    plots = []
    for p in flist:
        plots += glob(os.path.join(o,p))

    if plots != [] and len(plots) > 9:
        f.write('''
\\begin{frame}
\\tiny %s \medskip
\\begin{columns}
\\column{0.2\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\vspace{0.5em}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\vspace{0.5em}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\end{figure}
\\column{0.2\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\vspace{0.5em}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\vspace{0.5em}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\end{figure}
\\column{0.2\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\vspace{0.5em}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\vspace{0.5em}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\end{figure}
\\column{0.2\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\vspace{0.5em}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\vspace{0.5em}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\end{figure}
\\column{0.2\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\vspace{0.5em}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\vspace{0.5em}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\end{figure}
\\end{columns}
\\end{frame}
''' % (o.split("/")[-2].replace("_", " "), plots[0], plots[1], plots[2], plots[3], plots[4], plots[5], plots[6], plots[7], plots[8], plots[9], plots[10], plots[11], plots[12], plots[13], plots[14]))

def write_slide_big(f, o, flist):
    plots = []
    for p in flist:
        plots += glob(os.path.join(o,p))

    print(len(plots))

    if plots != [] and len(plots) > 19:
        f.write('''
\\begin{frame}
\\begin{columns}
\\column{0.20\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\end{figure}
\\column{0.20\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\end{figure}
\\column{0.20\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\end{figure}
\\column{0.20\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\end{figure}
\\column{0.20\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\end{figure}
\\end{columns}
\\end{frame}
''' % (plots[0], plots[1], plots[2], plots[3], plots[4], plots[5], plots[6], plots[7], plots[8], plots[9], plots[10], plots[11], plots[12], plots[13], plots[14], plots[15], plots[16], plots[17], plots[18], plots[19]))

def write_slide_shap(f, o, flist):
    plots = []
    for p in flist:
        plots += glob(os.path.join(o,p))

    print(len(plots))

    if plots != [] and len(plots) > 11:
        f.write('''
\\begin{frame}
\\begin{columns}
\\column{0.25\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\end{figure}
\\column{0.25\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\end{figure}
\\column{0.25\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\end{figure}
\\column{0.25\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\vspace{0.1em}
    \\includegraphics[width = 0.8\\textwidth]{%s}
    \\end{figure}
\\end{columns}
\\end{frame}
''' % (plots[0], plots[1], plots[2], plots[3], plots[4], plots[5], plots[6], plots[7], plots[8], plots[9], plots[10], plots[11]))
if __name__ == "__main__":
    #Collect all training output dirs
    outputs = get_outdirs()
    
    # Plots that we want to see together
    flist = [
        "2D_BG_Disc1VsDisc2.png", 
        "2D_SG550_Disc1VsDisc2.png", 
        "Njets_Region_A_PredVsActual.png", 
        "output_loss_train.png", 
        "output_loss_val.png", 
        "loss_train_val.png", 
        "mass_split.png", 
        "mass_split_val.png", 
        "PandR2D_plot_disc1_mass550.png",
        "roc_plot_TT_nJet_disc1.png", 
        "roc_plot_TT_nJet_disc2.png", 
        "PandR2D_plot_disc2_mass550.png",
        "roc_plot_mass_split_disc1.png", 
        "roc_plot_mass_split_disc2.png", 
        "PandR2D_plot_disc1.png"
    ]

    flist2 = [
        '2D_SG550_Disc1VsDisc2_Njets6.png',
        '2D_BG_Disc1VsDisc2_Njets6.png',
        '2D_valSG550_Disc1VsDisc2_Njets6.png',
        '2D_valBG_Disc1VsDisc2_Njets6.png',
        '2D_SG550_Disc1VsDisc2_Njets7.png',
        '2D_BG_Disc1VsDisc2_Njets7.png',
        '2D_valSG550_Disc1VsDisc2_Njets7.png',
        '2D_valBG_Disc1VsDisc2_Njets7.png',
        '2D_SG550_Disc1VsDisc2_Njets8.png',
        '2D_BG_Disc1VsDisc2_Njets8.png',
        '2D_valSG550_Disc1VsDisc2_Njets8.png',
        '2D_valBG_Disc1VsDisc2_Njets8.png',
        '2D_SG550_Disc1VsDisc2_Njets9.png',
        '2D_BG_Disc1VsDisc2_Njets9.png',
        '2D_valSG550_Disc1VsDisc2_Njets9.png',
        '2D_valBG_Disc1VsDisc2_Njets9.png',
        '2D_SG550_Disc1VsDisc2_Njets10.png',
        '2D_BG_Disc1VsDisc2_Njets10.png',
        '2D_valSG550_Disc1VsDisc2_Njets10.png',
        '2D_valBG_Disc1VsDisc2_Njets10.png',
        '2D_SG550_Disc1VsDisc2_Njets11.png',
        '2D_BG_Disc1VsDisc2_Njets11.png',
        '2D_valSG550_Disc1VsDisc2_Njets11.png',
        '2D_valBG_Disc1VsDisc2_Njets11.png',
        '2D_SG550_Disc1VsDisc2_Njets12.png',
        '2D_BG_Disc1VsDisc2_Njets12.png',
        '2D_valSG550_Disc1VsDisc2_Njets12.png',
        '2D_valBG_Disc1VsDisc2_Njets12.png',
        '2D_SG550_Disc1VsDisc2_Njets13.png',
        '2D_BG_Disc1VsDisc2_Njets13.png',
        '2D_valSG550_Disc1VsDisc2_Njets13.png',
        '2D_valBG_Disc1VsDisc2_Njets13.png',
    ]

    flist3 = [
        'NonClosure_vs_Disc1Disc2_Njets6.png',
        'NonClosureUnc_vs_Disc1Disc2_Njets6.png',
        'Sign_vs_Disc1Disc2_Njets6.png',
        'SignUnc_vs_Disc1Disc2_Njets6.png',
        'NonClosure_vs_Disc1Disc2_Njets7.png',
        'NonClosureUnc_vs_Disc1Disc2_Njets7.png',
        'Sign_vs_Disc1Disc2_Njets7.png',
        'SignUnc_vs_Disc1Disc2_Njets7.png',
        'NonClosure_vs_Disc1Disc2_Njets8.png',
        'NonClosureUnc_vs_Disc1Disc2_Njets8.png',
        'Sign_vs_Disc1Disc2_Njets8.png',
        'SignUnc_vs_Disc1Disc2_Njets8.png',
        'NonClosure_vs_Disc1Disc2_Njets9.png',
        'NonClosureUnc_vs_Disc1Disc2_Njets9.png',
        'Sign_vs_Disc1Disc2_Njets9.png',
        'SignUnc_vs_Disc1Disc2_Njets9.png',
        'NonClosure_vs_Disc1Disc2_Njets10.png',
        'NonClosureUnc_vs_Disc1Disc2_Njets10.png',
        'Sign_vs_Disc1Disc2_Njets10.png',
        'SignUnc_vs_Disc1Disc2_Njets10.png',
        'NonClosure_vs_Disc1Disc2_Njets11.png',
        'NonClosureUnc_vs_Disc1Disc2_Njets11.png',
        'Sign_vs_Disc1Disc2_Njets11.png',
        'SignUnc_vs_Disc1Disc2_Njets11.png',
        'NonClosure_vs_Disc1Disc2_Njets12.png',
        'NonClosureUnc_vs_Disc1Disc2_Njets12.png',
        'Sign_vs_Disc1Disc2_Njets12.png',
        'SignUnc_vs_Disc1Disc2_Njets12.png',
        'NonClosure_vs_Disc1Disc2_Njets13.png',
        'NonClosureUnc_vs_Disc1Disc2_Njets13.png',
        'Sign_vs_Disc1Disc2_Njets13.png',
        'SignUnc_vs_Disc1Disc2_Njets13.png',
    ]

    flist4 = [
        'violin_plot_disc1_plot.png',
        'violin_plot_disc2_plot.png',
        'bar_plot_disc1_plot.png',
        'bar_plot_disc2_plot.png',
        'heatmap_plot_disc1_plot.png',
        'heatmap_plot_disc2_plot.png',
        'layered_violin_plot_disc1_plot.png',
        'layered_violin_plot_disc2_plot.png',
        'violin_plot_disc1_plot.png',
        'violin_plot_disc2_plot.png',
        'summary_plot_disc1_plot.png',
        'summary_plot_disc2_plot.png',
    ]

    with open("./quickLook.tex", "w") as f:
        beamer_header(f)

        for o in outputs:
            write_slide(f, o, flist)
            write_slide_big(f, o, flist2)
            write_slide_big(f, o, flist3)
            write_slide_shap(f, o, flist4)

        f.write("\\end{document}")    

    f.close()
