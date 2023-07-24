from DataLoader import DataLoader
from glob import glob
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import json
import os

parser = ArgumentParser(description='Determine importance of input variables')
parser.add_argument('--channel',    action='store', help='0l,1l,2l')
parser.add_argument('--model',      action='store', help='RPV,SYY')
parser.add_argument('--mass',       action='store', help='Stop mass in GeV')
args = parser.parse_args()

model = args.model
if args.channel == "0l":
    channel = "TTToHad"
elif args.channel == "1l":
    channel = "TTToSemi"
elif args.channel == "2l":
    channel = "TTTo2L"
else:
    channel = "TT"
mass = args.mass

plt.style.use('ggplot')

def get_config():
    cfg_file = open("config.json", "r")
    return json.loads(cfg_file.read())


def get_loader(config):
    ds = glob("NN_inputs/*201*{}*".format(model)) + glob("NN_inputs/*201*{}*".format(channel))
    print(ds)

    if mass == "0":
        sg = []
        for i_mass in range(300, 1450, 50):
            sg = []
            #sg += [x for x in ds if "Train" in x and "-{}".format(i_mass) in x]
    else:
        #sg = [x for x in ds if "Train" in x and "-{}".format(mass) in x]
        sg = []
    bg = [x for x in ds if "Train" in x and "{}".format(channel) in x]

    return DataLoader(config, sg, bg)

config = get_config()
loader = get_loader(config)

flat = loader.getFlatData()
print(flat['model'])
data = flat['inputs']
labels = flat['label']
print(len(data))

select =  SelectKBest(f_classif, k=20)
select.fit(data, labels)
top = select.get_feature_names_out(flat['vars'])

output = zip(flat['vars'], select.scores_)

sort = sorted(output, key=lambda x: x[1])

sort.reverse()

print("Writing file ./Importance/FScore_{}_{}_{}.txt...".format(args.channel, model, mass))

with open("./Importance/FScore_{}_{}_{}.txt".format(args.channel, model, mass), "w") as f:
    f.write("F Score Rankings for {} {} {}\n".format(args.channel, model, mass))
    f.write("-----------------------------\n")
    for i in sort:
        f.write("{}: {}\n".format(i[0], i[1]))

f.close()
