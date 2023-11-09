
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser
from collections import OrderedDict

class Plotter:

    def __init__(self, basedir, input_name = "ValData*.json"):
        
        self.basedir = basedir
        self.input_name = input_name

        input_files = []

        for dir in self.basedir:
            input_files += glob.glob("{}/{}".format(dir, input_name))

        self.plot_dict = {}
        self.metadata = {}

        for i_train,i in enumerate(input_files):
            input_key = i.split("/")[-1].split("_")[2]
            
            self.plot_dict, self.metadata = self.parse_jsons(i, input_key)

        # Specify two metrics for each axis
        x_var = "Sign"
        y_var = "NonClosure"

        self.plot(self.plot_dict, x_var, y_var)
        self.plot(self.plot_dict, x_var, "normSigFracs")
        self.plot(self.plot_dict, y_var, "normSigFracs")

        self.plot1D(self.plot_dict, x_var)
        self.plot1D(self.plot_dict, y_var)
        self.plot1D(self.plot_dict, "normSigFracs")

        self.plot_meta(self.metadata, "AvgNonClosure", "AvgSign")
        self.plot_meta(self.metadata, "AvgNonClosure", "CountReasonable")
        self.plot_meta(self.metadata, "AvgNonClosure", "lossDiff")

    def parse_jsons(self, inpath, input_key):
       
        def filter_data(edge):
            return float(edge.split(",")[0][1:]) > 0.2 and float(edge.split(",")[1][:-1]) > 0.2 and float(edge.split(",")[0][1:]) < 0.9 and float(edge.split(",")[1][:-1]) < 0.9
       
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

        with open(inpath, "r") as f:
            raw = json.load(f)

        f.close()

        temp = {}
        # Keys of dictionary are formated as "(d1, d2)"
        for key in raw.keys():
            if type(raw[key]) is not dict:
                if check_overtrain(raw["trainLoss"], raw["valLoss"]): 
                    print("Overtrained... \nTrain Loss: {}\tValidation Loss: {}".format(raw["trainLoss"], raw["valLoss"]))
                    temp["plot"] = False
                temp["lossDiff"] = raw["valLoss"] - raw["trainLoss"]
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

                self.plot_dict[input_key][data_type].append(raw[key][data_type])


            self.plot_dict[input_key]["Discs"].append((key.split(",")[0][1:], key.split(",")[1][:-1]))

        self.metadata[input_key].append(temp)

        return self.plot_dict, self.metadata


    def plot1D(self, plot_dict, var):
      
        keyColorMap = {
            "CloseAndDisCo": 'green',
            "Close": 'red',
            "DisCo": 'blue',
            "None": 'red'
        }

        hist_dict = {}

        fig = plt.figure()

        for input_key in plot_dict.keys():
            counts, bins = np.histogram(plot_dict[input_key][var], bins=10, range=(0.0,1.0))
        
            plt.hist(bins[:-1], bins, weights=counts, histtype="step", color=keyColorMap[input_key], label=input_key)    
            

        plt.xlabel(var)
        plt.ylabel("Unweighted Event Count")
        plt.title("{} Distribution".format(var))
        plt.ylim(max(plot_dict["CloseAndDisCo"][var])+200)
        plt.legend()
        plt.savefig("plot_{}_ClosureDisCo.pdf".format(var), format="pdf", dpi=fig.dpi)
        plt.clf()

    def scatter_hist(self, x, y, ax, ax_histx, ax_histy, c=None, label=None, s=None):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        if label == "CloseAndDisCo":
            label = "Both"

        # the scatter plot:
        ax.scatter(x, y, c=c, label=label, s=s)

        # now determine nice limits by hand:
        ax_histx.hist(x, bins=np.arange(0.0, 385, 35), color = c, histtype='step')
        ax_histy.hist(y, bins=np.arange(0.0, 1.1, 0.1), orientation='horizontal', color = c, histtype='step')

    def plot(self, plot_dict, x_var, y_var):

        keyColorMap = {
            "CloseAndDisCo": 'green',
            "Close": 'red',
            "DisCo": 'blue',
            "None": 'red'
        }

        plot_dict = OrderedDict({"DisCo": plot_dict["DisCo"], "Close": plot_dict["Close"], "CloseAndDisCo": plot_dict["CloseAndDisCo"]})

        fig = plt.figure()

        ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
        ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
        ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

        for input_key in plot_dict.keys():

            self.scatter_hist(plot_dict[input_key][x_var], plot_dict[input_key][y_var], ax, ax_histx, ax_histy, c=keyColorMap[input_key], label=input_key, s=1)
            #plt.scatter(plot_dict[input_key][x_var], plot_dict[input_key][y_var], c=keyColorMap[input_key], label=input_key, s=1)

        plt.xlabel("Significance (A.U.)")
        plt.ylabel("Non-Closure (A.U.)")
        plt.legend(bbox_to_anchor=(1.3, 1.3), borderaxespad=0)
        plt.savefig("plot_{}vs{}_ClosureDisCo.pdf".format(x_var, y_var), format="pdf", dpi=fig.dpi)
        plt.clf()

    def plot_meta(self, metadata, x_var, y_var):
        
        keyColorMap = {
            "CloseAndDisCo": 'green',
            "Close": 'red',
            "DisCo": 'blue',
            "None": 'red'
        }
        
        fig = plt.figure()

        for input_key in metadata.keys():
            temp_meta_dict = {
                x_var: [],
                y_var: []
            }
            for meta_dict in metadata[input_key]:
                temp_meta_dict[x_var].append(meta_dict[x_var])
                temp_meta_dict[y_var].append(meta_dict[y_var])
            plt.scatter(temp_meta_dict[x_var], temp_meta_dict[y_var], c=keyColorMap[input_key], label=input_key, s=2)

        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.title("Training Comparison")
        plt.legend()
        plt.savefig("plot_{}vs{}_ClosureDisCo.pdf".format(x_var, y_var), format="pdf", dpi=fig.dpi)
        plt.clf()
        

if __name__ == "__main__":
   
    parser = ArgumentParser()
    
    parser.add_argument("--basedir", type=str, help="Base directory containing validation json", nargs="+", default=".") 
    parser.add_argument("--filename", type=str, help="Name of validation json file", default="ValData*.json") 

    args = parser.parse_args()

    Plotter(args.basedir, args.filename)
                    
