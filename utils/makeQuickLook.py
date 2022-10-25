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
    \\end{figure}
\\column{0.2\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\vspace{0.5em}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\end{figure}
\\column{0.2\\textwidth}
    \\begin{figure}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\vspace{0.5em}
    \\includegraphics[width = 0.9\\textwidth]{%s}
    \\end{figure}
\\column{0.2\\textwidth}
    \\begin{figure}
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
''' % (o.split("/")[-2].replace("_", " "), plots[0], plots[1], plots[2], plots[3], plots[4], plots[5], plots[6], plots[7], plots[8], plots[9], plots[10]))

def write_slide_big(f, o, flist):
    plots = []
    for p in flist:
        plots += glob(os.path.join(o,p))

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

if __name__ == "__main__":
    #Collect all training output dirs
    outputs = get_outdirs()
    
    # Plots that we want to see together
    flist = ["2D_BG_Disc1VsDisc2.pdf", "2D_SG550_Disc1VsDisc2.pdf", "output_loss_train.pdf", "output_loss_val.pdf", "Njets_Region_A_PredVsActual.pdf", "loss_train_val.pdf", "roc_plot_TT_nJet_disc1.pdf", "roc_plot_TT_nJet_disc2.pdf", "roc_plot_mass_split_disc1.pdf", "roc_plot_mass_split_disc2.pdf", "PandR_plot.pdf"]

    flist2 = [
        '2D_SG550_Disc1VsDisc2_Njets7.pdf',
        '2D_BG_Disc1VsDisc2_Njets7.pdf',
        '2D_valSG550_Disc1VsDisc2_Njets7.pdf',
        '2D_valBG_Disc1VsDisc2_Njets7.pdf',
        '2D_SG550_Disc1VsDisc2_Njets8.pdf',
        '2D_BG_Disc1VsDisc2_Njets8.pdf',
        '2D_valSG550_Disc1VsDisc2_Njets8.pdf',
        '2D_valBG_Disc1VsDisc2_Njets8.pdf',
        '2D_SG550_Disc1VsDisc2_Njets9.pdf',
        '2D_BG_Disc1VsDisc2_Njets9.pdf',
        '2D_valSG550_Disc1VsDisc2_Njets9.pdf',
        '2D_valBG_Disc1VsDisc2_Njets9.pdf',
        '2D_SG550_Disc1VsDisc2_Njets10.pdf',
        '2D_BG_Disc1VsDisc2_Njets10.pdf',
        '2D_valSG550_Disc1VsDisc2_Njets10.pdf',
        '2D_valBG_Disc1VsDisc2_Njets10.pdf',
        '2D_SG550_Disc1VsDisc2_Njets11.pdf',
        '2D_BG_Disc1VsDisc2_Njets11.pdf',
        '2D_valSG550_Disc1VsDisc2_Njets11.pdf',
        '2D_valBG_Disc1VsDisc2_Njets11.pdf',
    ]

    flist3 = [
        'NonClosure_vs_Disc1Disc2_Njets7.pdf',
        'NonClosureUnc_vs_Disc1Disc2_Njets7.pdf',
        'Sign_vs_Disc1Disc2_Njets7.pdf',
        'SignUnc_vs_Disc1Disc2_Njets7.pdf',
        'NonClosure_vs_Disc1Disc2_Njets8.pdf',
        'NonClosureUnc_vs_Disc1Disc2_Njets8.pdf',
        'Sign_vs_Disc1Disc2_Njets8.pdf',
        'SignUnc_vs_Disc1Disc2_Njets8.pdf',
        'NonClosure_vs_Disc1Disc2_Njets9.pdf',
        'NonClosureUnc_vs_Disc1Disc2_Njets9.pdf',
        'Sign_vs_Disc1Disc2_Njets9.pdf',
        'SignUnc_vs_Disc1Disc2_Njets9.pdf',
        'NonClosure_vs_Disc1Disc2_Njets10.pdf',
        'NonClosureUnc_vs_Disc1Disc2_Njets10.pdf',
        'Sign_vs_Disc1Disc2_Njets10.pdf',
        'SignUnc_vs_Disc1Disc2_Njets10.pdf',
        'NonClosure_vs_Disc1Disc2_Njets11.pdf',
        'NonClosureUnc_vs_Disc1Disc2_Njets11.pdf',
        'Sign_vs_Disc1Disc2_Njets11.pdf',
        'SignUnc_vs_Disc1Disc2_Njets11.pdf',
    ]

    with open("./quickLook.tex", "w") as f:
        beamer_header(f)

        for o in outputs:
            write_slide(f, o, flist)
            write_slide_big(f, o, flist2)
            write_slide_big(f, o, flist3)

        f.write("\\end{document}")    

    f.close()
