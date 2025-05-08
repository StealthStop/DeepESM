
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser
from collections import OrderedDict

import mplhep as hep
plt.style.use([hep.style.ROOT,hep.style.CMS]) # For now ROOT defaults to CMS
plt.style.use({'legend.frameon':False,'legend.fontsize':16,'legend.edgecolor':'black'})

class Plotter:

    def __init__(self, basedir, input_name = "ValData*.json"):
        
        self.basedir = [basedir]
        self.input_name = input_name

        input_files = []

        for dir in self.basedir:
            input_files += glob.glob("{}/{}".format(dir, input_name))

        self.plot_dict = {}
        self.metadata = {}
        self.single_trainings = {}

        for i_train,i in enumerate(input_files):
            input_key = i.split("/")[-1].split("_")[2]
            
            self.plot_dict, self.metadata = self.parse_jsons(i, input_key)

        # Specify two metrics for each axis
        x_var = "SignWithNonClosure"
        y_var = "NonClosure"

        self.plot(self.plot_dict, x_var, y_var)
        self.plot(self.plot_dict, x_var, "normSigFracs")
        self.plot(self.plot_dict, y_var, "normSigFracs")

        #self.plot1D(self.plot_dict, x_var)
        #self.plot1D(self.plot_dict, y_var)
        #self.plot1D(self.plot_dict, "normSigFracs")

        print(self.metadata)
        self.plot_meta(self.metadata, "AvgNonClosure", "AvgSignWithNonClosure")
        self.plot_meta(self.metadata, "AvgNonClosure", "CountReasonable")
        #self.plot_meta(self.metadata, "AvgNonClosure", "lossDiff")

        self.plot_SVJ(self.single_trainings, x_var, y_var, z_var="AvgNonClosure", meta=self.metadata)
        self.plot_SVJ(self.single_trainings, x_var, y_var, z_var="VarNonClosure", meta=self.metadata)

    def parse_jsons(self, inpath, input_key):
       
        def filter_data(edge):
            return float(edge.split(",")[0][1:]) > 0.2 and float(edge.split(",")[1][:-1]) > 0.2 and float(edge.split(",")[0][1:]) < 0.7 and float(edge.split(",")[1][:-1]) < 0.7
       
        def check_quality(nonclosure, sign, sigFrac):
            return nonclosure <= 1.0 and sign >= 0.0 and sigFrac <= 1.0 and sigFrac >=0.0

        def check_overtrain(trainLoss, valLoss):
            return valLoss > 1.2 * trainLoss
 
        if input_key not in self.plot_dict.keys(): 
            self.plot_dict[input_key] = {
                "Discs": []
                }
        if input_key not in self.metadata.keys():
            self.metadata[input_key] = []

        if input_key not in self.single_trainings.keys():
            self.single_trainings[input_key] = []

        with open(inpath, "r") as f:
            raw = json.load(f)

        f.close()

        temp = {}
        temp_plot_dict = {'Discs': []}
        # Keys of dictionary are formated as "(d1, d2)"
        count = 0
        temp_sign = 0
        for key in raw.keys():
            if type(raw[key]) is not dict:
                #if check_overtrain(raw["trainLoss"], raw["valLoss"]): 
                #    print("Overtrained... \nTrain Loss: {}\tValidation Loss: {}".format(raw["trainLoss"], raw["valLoss"]))
                #    temp["plot"] = False
                #temp["lossDiff"] = raw["valLoss"] - raw["trainLoss"]
                temp[key] = raw[key]
                temp["plot"] = True
                continue
            elif not filter_data(key):
                #print("Did not pass filter")
                continue
            for data_type in raw[key].keys():
               
                if not check_quality(raw[key]["NonClosure"], raw[key]["Sign"], raw[key]["normSigFracs"]): continue
 
                if data_type not in self.plot_dict[input_key].keys():
                    self.plot_dict[input_key][data_type] = []
                if data_type not in temp_plot_dict.keys():
                    temp_plot_dict[data_type] = []

                if data_type == "SignWithNonClosure":
                    count += 1
                    temp_sign += raw[key][data_type]

                self.plot_dict[input_key][data_type].append(raw[key][data_type])
                temp_plot_dict[data_type].append(raw[key][data_type])

            self.plot_dict[input_key]["Discs"].append((key.split(",")[0][1:], key.split(",")[1][:-1]))
            temp_plot_dict["Discs"].append((key.split(",")[0][1:], key.split(",")[1][:-1]))

        signWithNonClosure = temp_sign / float(count)
        temp["AvgSignWithNonClosure"] = signWithNonClosure

        self.metadata[input_key].append(temp)

        self.single_trainings[input_key].append((temp_plot_dict, temp))

        return self.plot_dict, self.metadata

    def plot1D(self, plot_dict, var):
      
        keyColorMap = {
            "CloseAndDisCo": 'green',
            "Close": 'red',
            "DisCo": 'blue',
            "None": 'red'
        }

        keyLabelMap = {
            "CloseAndDisCo": 'Both',
            "Close": 'Closure Loss',
            "DisCo": 'DisCo Loss',
            "None": 'None'
        }

        hist_dict = {}

        fig = plt.figure()

        for input_key in plot_dict.keys():
            counts, bins = np.histogram(plot_dict[input_key][var], bins=10, range=(0.0,1.0))
        
            plt.hist(bins[:-1], bins, weights=counts, histtype="step", color=keyColorMap[input_key], label=keyLabelMap[input_key])    
            

        plt.xlabel(var)
        plt.ylabel("Unweighted Event Count")
        plt.title("{} Distribution".format(var))
        plt.ylim(max(plot_dict["CloseAndDisCo"][var])+200)
        plt.legend()
        plt.savefig("plot_{}_ClosureDisCo.pdf".format(var), format="pdf", dpi=fig.dpi)
        plt.clf()

    def scatter_hist(self, x, y, ax, ax_histx, ax_histy, c=None, label=None, s=None, cmap=None):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        ax_histx.set_ylabel('Counts')
        ax_histy.set_xlabel('Counts')

        if label == "CloseAndDisCo":
            label = "Both"

        # the scatter plot:
        ax.scatter(x, y, c=c, label=label, s=s, cmap=cmap)

        # now determine nice limits by hand:
        ax_histx.hist(x, bins=np.arange(0.0, 1.1, 0.1), color = c, histtype='step')
        ax_histy.hist(y, bins=np.arange(0.0, 1.1, 0.1), orientation='horizontal', color = c, histtype='step')

    def plot_SVJ(self, single_trainings, x_var, y_var, z_var=None, meta=None):

        fig, axs = plt.subplots(3, 1, figsize=(5, 8), tight_layout=True)
       
        single_trainings = OrderedDict({"CloseAndDisCo": single_trainings["CloseAndDisCo"], "Close": single_trainings["Close"], "DisCo": single_trainings["DisCo"]})
 
        for i,key in enumerate(single_trainings.keys()):
            
            #ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
            #ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
            #ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

            axs[i].set_yscale('log')
       
            for tup in single_trainings[key]:

                plot_dict = tup[0]
                meta = tup[1]

                var = np.var(plot_dict[y_var])
                avg = np.mean(plot_dict[y_var])

                z = avg if "Avg" in z_var else var

                vmax = 0.05 if "Var" in z_var else 0.8
 
                scat = axs[i].scatter(plot_dict[x_var], plot_dict[y_var], c=[z]*len(plot_dict[x_var]), cmap="viridis", label=key, s=5, marker=".", vmin=0.0, vmax=vmax)

            axs[i].set_xlim(0, 1.0)
            axs[i].set_ylim(0.01, 5.0)

            if key == "CloseAndDisCo":
                plt_txt = "Closure + DisCo"
            elif key == "Close":
                plt_txt = "Closure only"
            elif key == "DisCo":
                plt_txt = "DisCo only"
            axs[i].text(0.75, 0.9, plt_txt, horizontalalignment='center', transform=axs[i].transAxes)

            if "Avg" in z_var:
                fig.colorbar(scat, ax=axs[i], label="Closure Means")
            else:
                fig.colorbar(scat, ax=axs[i], label="Closure Variances")
                

        for ax in axs.flat:
            if ax == axs.flat[-1]:
                ax.set(xlabel="Significance (A.U.)", ylabel="Non-Closure (A.U.)")
            else:
                ax.set(ylabel="Non-Closure (A.U.)")

        plt.savefig("plot_{}vs{}with{}.pdf".format(x_var, y_var, z_var), format="pdf", dpi=fig.dpi)
        plt.clf()


    def plot(self, plot_dict, x_var, y_var, z_var=None, typeTrain=None, meta=None):
        

        def remove_outliers(sig_vals):

            sig_arr = np.array(sig_vals)

            std = np.std(sig_arr)
            mean = np.mean(sig_arr)

            #return (sig_arr < (mean + std)) & (sig_arr > (mean - std))
            return (sig_arr < (mean + 2*std)) & (sig_arr > (mean - 2*std))


        keyColorMap = {
            "CloseAndDisCo": '#5790fc',
            "Close": '#f89c20',
            "DisCo": '#e42536',
            "None": 'red'
        }

        keyLabelMap = {
            "CloseAndDisCo": 'Both',
            "Close": 'Closure Loss',
            "DisCo": 'DisCo Loss',
            "None": 'None'
        }

        plot_dict = OrderedDict({"DisCo": plot_dict["DisCo"], "Close": plot_dict["Close"], "CloseAndDisCo": plot_dict["CloseAndDisCo"]})

        fig = plt.figure(figsize=(12,10))

        ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
        ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
        ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

        if z_var is not None:

            scat = self.scatter_hist(plot_dict[typeTrain][x_var], plot_dict[typeTrain][y_var], ax, ax_histx, ax_histy, c=plot_dict[typeTrain]["meta"][z_var], cmap="viridis", label=typeTrain, s=1)
            #plt.scatter(plot_dict[input_key][x_var], plot_dict[input_key][y_var], c=keyColorMap[input_key], label=input_key, s=1)
            plt.colorbar(scat)

        else:
            maximum = -9999
            for input_key in plot_dict.keys():
                #mask = remove_outliers(plot_dict[input_key][x_var])
                #no_os = np.array(plot_dict[input_key][x_var])[mask]
                if input_key != "CloseAndDisCo": continue
                no_os = plot_dict[input_key][x_var]
                temp_max = max(no_os)
                if temp_max > maximum:
                    maximum = temp_max

            for input_key in plot_dict.keys():
                #mask = remove_outliers(plot_dict[input_key][x_var])

                #scat = self.scatter_hist(np.array(plot_dict[input_key][x_var])[mask] / maximum, np.array(plot_dict[input_key][y_var])[mask], ax, ax_histx, ax_histy, c=keyColorMap[input_key], label=keyLabelMap[input_key], s=1)

                mask = np.array(plot_dict[input_key][x_var]) / maximum / 0.8 <= 1.0
                print(len(np.array(plot_dict[input_key][x_var])[mask] / maximum / 0.8), len(np.array(plot_dict[input_key][y_var])[mask]))
                scat = self.scatter_hist(np.array(plot_dict[input_key][x_var])[mask] / maximum / 0.8, np.array(plot_dict[input_key][y_var])[mask], ax, ax_histx, ax_histy, c=keyColorMap[input_key], label=keyLabelMap[input_key], s=1.)

        hep.cms.label(llabel="Simulation", data=False, paper=False, year="", ax=ax_histx)

        plt.xlabel("Normalized significance (A.U.)")
        plt.ylabel("Non-closure (A.U.)")
        if z_var is not None:
            plt.savefig("plot_{}vs{}with{}for{}_ClosureDisCo.pdf".format(x_var, y_var, z_var, typeTrain), format="pdf", dpi=fig.dpi)
            plt.text(0.6, 0.9, typeTrain)
        else:
            legend = plt.legend(bbox_to_anchor=(1.3, 1.25), borderaxespad=0)
            legend.legendHandles[0]._sizes = [30]
            legend.legendHandles[1]._sizes = [30]
            legend.legendHandles[2]._sizes = [30]
            plt.gca().set_xlim([0., 1.1])

            plt.text(0.5, 0.95, r"DisCo Only: $5000 \leq \lambda_{DisCo} \leq 50000$", fontsize=18)
            plt.text(0.5, 0.875, r"Closure Only: $1 \leq \lambda_{\mathrm{Non-closure}} \leq 1000$", fontsize=18)
            plt.text(0.5, 0.8, r"Both: $1 \leq \lambda_{Disco, \mathrm{Non-closure}} \leq 100$", fontsize=18)
            plt.savefig("plot_{}vs{}_ClosureDisCo.pdf".format(x_var, y_var), format="pdf", dpi=fig.dpi)
        plt.clf()

    def plot_meta(self, metadata, x_var, y_var):
        
        keyColorMap = {
            "CloseAndDisCo": 'green',
            "Close": 'red',
            "DisCo": 'blue',
            "None": 'red'
        }
        
        keyLabelMap = {
            "CloseAndDisCo": 'Both',
            "Close": 'Closure Loss',
            "DisCo": 'DisCo Loss',
            "None": 'None'
        }

        fig = plt.figure()

        metadata = OrderedDict({"DisCo": metadata["DisCo"], "Close": metadata["Close"], "CloseAndDisCo": metadata["CloseAndDisCo"]})


        maximum = -9999
        for input_key in metadata.keys():
            for meta_dict in metadata[input_key]:
                if maximum < meta_dict[y_var]:
                    maximum = meta_dict[y_var]

        for input_key in metadata.keys():
            temp_meta_dict = {
                x_var: [],
                y_var: []
            }
            for meta_dict in metadata[input_key]:
                temp_meta_dict[x_var].append(meta_dict[x_var])
                temp_meta_dict[y_var].append(meta_dict[y_var])

            plt.scatter(np.array(temp_meta_dict[y_var])/maximum, np.array(temp_meta_dict[x_var]), c=keyColorMap[input_key], label=keyLabelMap[input_key], s=2)

        y_label = y_var
        x_label = x_var

        if x_var == "AvgNonClosure":
            x_label = "Avg. Non-Closure"
        if y_var == "AvgSignWithNonClosure":
            y_label = "Avg. Normalized Significance"
        

        ax = fig.axes[0]
        plt.xlabel(y_label)
        plt.ylabel(x_label)
        #plt.title("Training Comparison")
        hep.cms.label(llabel="Simulation", data=False, paper=False, year="Run 2", ax=ax)
        plt.legend()
        plt.savefig("plot_{}vs{}_ClosureDisCo.pdf".format(x_var, y_var), format="pdf", dpi=fig.dpi)
        plt.clf()
        

if __name__ == "__main__":
   
    parser = ArgumentParser()
    
    parser.add_argument("--basedir", type=str, help="Base directory containing validation json", nargs="+", default="./batch/BigTrainRerun_*/Output/Output/*/Run2/") 
    parser.add_argument("--filename", type=str, help="Name of validation json file", default="ValData*.json") 

    args = parser.parse_args()

    Plotter(args.basedir, args.filename)
                    
