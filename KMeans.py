#!/bin/env python

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from Validation import Validation
import uproot4 as uproot
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import math

class KMeans(MeanShift):

    def __init__(self, modelPath, bkgSample):
        self.bkgSample = bkgSample
        #self.model = self.load_model(modelPath)

        #self.load_data()

        if not os.path.isdir('MeanShiftResults'):
            os.makedirs('MeanShiftResults')

        self.run_demo([[0.25,0.25],[0.25,0.75],[0.75,0.25],[0.75,0.75]], 0.1, tag='4Centers')
        self.run_uneven([[0.25,0.25],[0.25,0.75],[0.75,0.25],[0.75,0.75]], 0.1, tag='uneven')
        self.run_demo([[0.25,0.25]], 0.3, tag='1Centers')
        self.run_demo([[0.25,0.25],[0.75,0.75]], 0.15, tag='exp')
        self.run_exp_demo(0.3)

    def run_demo(self, centers, std, tag=''):
        self.tag = tag
        self.data = self.generate_fake_data(centers, std)

        self.plot_results(self.data)

        self.cluster(self.data)

        self.plot_clusters(self.data)

    def run_uneven(self, centers, std, tag='uneven'):
        self.tag = tag
        self.data = self.generate_uneven_pops(centers, std)

        self.plot_results(self.data)

        self.cluster(self.data)

        self.plot_clusters(self.data)

    def run_exp_demo(self, scale, tag='exp'):
        self.tag = tag
        self.data = self.generate_exponential(scale)

        self.plot_results(self.data)

        self.cluster(self.data)

        self.plot_clusters(self.data)

    # Fake data set to play around with 
    def generate_fake_data(self, centers, std):
        data, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=std)

        return data[np.where((data[:,0] >= 0.0) & (data[:,1] >= 0.0) & (data[:,0] <= 1.0) & (data[:,1] <= 1.0))]

    def generate_uneven_pops(self, centers, std):
        for i,c in enumerate(centers):
            if i == 0:
                data, _ = make_blobs(n_samples=10000, centers = [c], cluster_std = std)
            elif i == 3:
                new_data, _ = make_blobs(n_samples=1000, centers = [c], cluster_std = std)
                data = np.concatenate((data, new_data))
            else:
                new_data, _ = make_blobs(n_samples=5000, centers = [c], cluster_std = std)
                data = np.concatenate((data, new_data))

        return data[np.where((data[:,0] >= 0.0) & (data[:,1] >= 0.0) & (data[:,0] <= 1.0) & (data[:,1] <= 1.0))]
            

    def generate_exponential(self, scale):
        data = np.random.exponential(scale=scale, size= (10000,2))

        return data[np.where((data[:,0] >= 0.0) & (data[:,1] >= 0.0) & (data[:,0] <= 1.0) & (data[:,1] <= 1.0))]

    # Load all inputs into a np array for sklearn to use
    def load_data(self):
        f = uproot.open(self.bkgSample)
        theVars = ["Jet_ptrHT_1", "Jet_ptrHT_2", "Jet_ptrHT_3", "Jet_ptrHT_4", "Jet_ptrHT_5", "Jet_ptrHT_6", "Jet_ptrHT_7", "Jet_eta_1", "Jet_eta_2", "Jet_eta_3", "Jet_eta_4", "Jet_eta_5", "Jet_eta_6", "Jet_eta_7", "Jet_phi_2", "Jet_phi_3", "Jet_phi_4", "Jet_phi_5", "Jet_phi_6", "Jet_phi_7", "Jet_CSVb_1", "Jet_CSVb_2", "Jet_CSVb_3", "Jet_CSVb_4", "Jet_CSVb_5", "Jet_CSVb_6", "Jet_CSVb_7", "HT_trigger_pt30", "fwm2_top6", "fwm3_top6", "fwm4_top6", "fwm5_top6", "jmt_ev0_top6", "jmt_ev1_top6", "jmt_ev2_top6", "Stop1_mass_cm_OldSeed", "Stop1_pt_cm_OldSeed", "Stop1_phi_cm_OldSeed", "Stop1_eta_cm_OldSeed", "Stop2_mass_cm_OldSeed", "Stop2_pt_cm_OldSeed", "Stop2_phi_cm_OldSeed", "Stop2_eta_cm_OldSeed" ]
        self.columnHeaders = f["myMiniTree_1l"].arrays(expressions=theVars, library="np")

        print(self.columnHeaders)

        f.close()

    # Load pbs file for inference
    def load_model(self, modelPath):
        return tf.saved_model.load(modelPath)

    # Pull in model we want to use and validation data, inference on validation data with model and produce output
    def get_disc_results():
        return

    # Take in (disc 1, disc 2) information from loader, run MeanShift and determine cluster labels
    @tf.function
    def cluster(self, data):
        bw = estimate_bandwidth(data, quantile=0.2, n_samples=500)

        self.ms = MeanShift(bandwidth=bw, bin_seeding=True)
        self.ms.fit(data)
        self.labels = self.ms.labels_
        self.cluster_centers = self.ms.cluster_centers_
        
        labels_unique = np.unique(self.labels)
        self.n_clusters_ = len(labels_unique)

        self.cluster_percent = [len(data[self.labels == i, 0])/len(data) for i in range(self.n_clusters_)]
        print("Cluster Distribution: ", self.cluster_percent)

        print("number of estimated clusters : %d" % self.n_clusters_)
        print("Centers: ", self.cluster_centers)

    def plot_clusters(self, data):
        plt.clf()
        for i in range(self.n_clusters_):
            plt.scatter(data[self.labels == i, 0], data[self.labels == i, 1], label = i)
        plt.legend()
        loss = self.cluster_loss(self.data)
        plt.text(1.0,1.1,"Loss: {}".format(round(loss,4)), horizontalalignment='right')
        plt.savefig('MeanShiftResults/Clusters_{}.png'.format(self.tag))

    def cluster_loss(self, data):
        idx_min = self.find_min_center()
        min = self.cluster_centers[idx_min]

        print("Minimum center = ", self.cluster_centers[idx_min])

        loss = 0.0000
        for i,point in enumerate(self.cluster_centers):
            loss += (self.n_clusters_ - 1) * self.distance(point, min) * self.cluster_percent[i]

        print("Loss: ", loss)
        return loss
 
    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def find_min_center(self):
        min = 999.0
        for c in self.cluster_centers:
            dist = c[0] ** 2 + c[1] ** 2
            if dist < min:
                min = dist
                idx_min = np.where(self.cluster_centers == c)

        return idx_min

    # Plot showing the clusters decided on by algorithm
    def plot_results(self, data):
        plt.clf()
        x = data.transpose()[0,:]
        y = data.transpose()[1,:]
        plt.hist2d(x, y, bins=(50,50), range=[[0,1],[0,1]])
        plt.savefig('MeanShiftResults/FakeDataHist_{}.png'.format(self.tag))

KMeans("/home/nstrobbe/cros0400/Train/DeepESM/MeanShiftTraining","/home/nstrobbe/shared/hadd_NNInputs_6_24_22/MyAnalysis_2016preVFP_TTToSemiLeptonic_Val.root")

