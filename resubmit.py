#! /bin/env/python

import os, glob, argparse, subprocess

def generate_qsub_config(taskPath, workPath, resubs, cluster, walltime, memory):

    pbsPath = "%s/job_submit.pbs"%(taskPath) 
    g = open(pbsPath, "r")

    lines = g.readlines()
    g.close()

    f = open(pbsPath.replace("submit", "resubmit"), "w")

    resubStr = ""
    for resub in resubs:
        resubStr += "%s,"%(resub)
    resubStr = resubStr[0:-1]

    for line in lines:
        if   "time" in line and walltime != "NULL":
            f.write("#SBATCH --time=%s\n"%(walltime))
        elif "gres" in line and cluster != "NULL":
            f.write("#SBATCH --gres=gpu:%s:1\n"%(cluster.partition("-")[0]))
        elif "-p " in line and cluster != "NULL":
            f.write("#SBATCH -p %s\n"%(cluster))
        elif "array" in line:
            f.write("#SBATCH --array=%s\n"%(resubStr))
        else:
            f.write(line)
    f.close()

if __name__ == '__main__':     

    parser = argparse.ArgumentParser()
    parser.add_argument("--jobDir",   dest="jobDir",   help="Directory where jobs submitted from",  default="RPV",  type=str)
    parser.add_argument("--cluster",  dest="cluster",  help="which cluster run on",                 default="NULL", type=str)
    parser.add_argument("--memory",   dest="memory",   help="how much mem to request",              default="NULL", type=str)
    parser.add_argument("--walltime", dest="walltime", help="how much time to request",             default="NULL", type=str)
    args = parser.parse_args()

    USER = os.getenv("USER")

    workDir = "/home/nstrobbe/%s/Train/DeepESM/"%(USER)
    taskDir = "./batch/%s"%(args.jobDir)

    jsons1 = glob.glob(taskDir + "/*.json")
    jsons2 = glob.glob(taskDir + "/*_metric.json")

    resubs = []
    for aJson in jsons1:

        if "_metric" in aJson:
            continue
        jobMetricName = aJson.split("/")[-1].split(".json")[0] + "_metric.json"

        foundMetric = False
        for bJson in jsons2:

            if jobMetricName in bJson:
                foundMetric = True
                break

        if not foundMetric:
            jobID = aJson.split("/")[-1].split(".json")[0].split("temp")[-1]
            resubs.append(jobID)

    if len(resubs) == 0:
        print("All jobs completed, nothing to resubmit !")
        quit()

    generate_qsub_config(taskDir, workDir, resubs, args.cluster, args.walltime, args.memory)
