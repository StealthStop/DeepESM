import json
import glob
from argparse import ArgumentParser
import matplotlib.pyplot as plt

class Plotter:

    def __init__(self, basedir, input_name = "ValData.json"):
        
        self.basedir = basedir
        self.input_name = input_name

        input_files = glob.glob("{}/{}".format(basedir, input_name))

        self.plot_dict = {}

        for i in input_files:
            input_key = i.split("_")[1][:-5]
            print(i)
            self.plot_dict = self.parse_jsons(i, input_key)

        # Specify two metrics for each axis
        x_var = "Sign"
        y_var = "NonClosure"

        self.plot(self.plot_dict, x_var, y_var)

    def parse_jsons(self, inpath, input_key):
       
        if input_key not in self.plot_dict.keys(): 
            self.plot_dict[input_key] = {
                "Discs": []
                }

        with open(inpath, "r") as f:
            raw = json.load(f)

        f.close()

        # Keys of dictionary are formated as "(d1, d2)"
        for key in raw.keys():
            for data_type in raw[key].keys():
                
                if data_type not in self.plot_dict[input_key].keys():
                    self.plot_dict[input_key][data_type] = []

                self.plot_dict[input_key][data_type].append(raw[key][data_type])

            self.plot_dict[input_key]["Discs"].append((key.split(",")[0][1:], key.split(",")[1][:-1]))

        return self.plot_dict 

    def plot(self, plot_dict, x_var, y_var):

        keyColorMap = {
            "CloseAndDisCo": 'g',
            "Close": 'o',
            "DisCo": 'b',
            "None": 'y'
        }

        for input_key in plot_dict.keys():
            plt.scatter(plot_dict[input_key][x_var], plot_dict[input_key][y_var], c=keyColorMap[input_key], label=input_key)

        plt.legend()
        plt.savefig("plot_{}vs{}_ClosureDisCo.png".format(x_var, y_var))

if __name__ == "__main__":
   
    parser = ArgumentParser()
    
    parser.add_argument("--basedir", type=str, help="Base directory containing validation json", default = ".") 
    parser.add_argument("--filename", type=str, help="Name of validation json file", default="ValData_CloseAndDisCo.json") 

    args = parser.parse_args()

    Plotter(args.basedir, args.filename)
                    
