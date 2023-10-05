import json
import numpy as np
import glob
from argparse import ArgumentParser
import matplotlib.pyplot as plt

class Plotter:

    def __init__(self, basedir, input_name = "ValData*.json"):
        
        self.basedir = basedir
        self.input_name = input_name

        input_files = []

        for dir in self.basedir:
            input_files += glob.glob("{}/{}".format(dir, input_name))

        self.plot_dict = {}

        for i in input_files:
            input_key = i.split("/")[-1].split("_")[1]
            print(input_key)
            
            self.plot_dict = self.parse_jsons(i, input_key)

        # Specify two metrics for each axis
        x_var = "Sign"
        y_var = "NonClosure"

        self.plot(self.plot_dict, x_var, y_var)
        self.plot(self.plot_dict, x_var, "normSigFracs")
        self.plot(self.plot_dict, y_var, "normSigFracs")

        self.plot1D(self.plot_dict, x_var)
        self.plot1D(self.plot_dict, y_var)
        self.plot1D(self.plot_dict, "normSigFracs")

    def parse_jsons(self, inpath, input_key):
       
        def filter_data(edge):
            return float(edge.split(",")[0][1:]) > 0.2 and float(edge.split(",")[1][:-1]) > 0.2 and float(edge.split(",")[0][1:]) < 0.9 and float(edge.split(",")[1][:-1]) < 0.9
       
        def check_quality(nonclosure, sign, sigFrac):
            return nonclosure <= 1.0 and sign >= 0.0 and sigFrac <= 1.0 and sigFrac >=0.0
 
        if input_key not in self.plot_dict.keys(): 
            self.plot_dict[input_key] = {
                "Discs": []
                }

        with open(inpath, "r") as f:
            raw = json.load(f)

        f.close()

        # Keys of dictionary are formated as "(d1, d2)"
        for key in raw.keys():
            if not filter_data(key):
                #print("Did not pass filter")
                continue
            for data_type in raw[key].keys():
               
                if not check_quality(raw[key]["NonClosure"], raw[key]["Sign"], raw[key]["normSigFracs"]): continue
 
                if data_type not in self.plot_dict[input_key].keys():
                    self.plot_dict[input_key][data_type] = []

                self.plot_dict[input_key][data_type].append(raw[key][data_type])


            self.plot_dict[input_key]["Discs"].append((key.split(",")[0][1:], key.split(",")[1][:-1]))

        return self.plot_dict 


    def plot1D(self, plot_dict, var):
       
        keyColorMap = {
            "CloseAndDisCo": 'green',
            "Close": 'orange',
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

    def plot(self, plot_dict, x_var, y_var):

        keyColorMap = {
            "CloseAndDisCo": 'green',
            "Close": 'orange',
            "DisCo": 'blue',
            "None": 'red'
        }

        fig = plt.figure()

        for input_key in plot_dict.keys():

            plt.scatter(plot_dict[input_key][x_var], plot_dict[input_key][y_var], c=keyColorMap[input_key], label=input_key, s=1)

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
                    
