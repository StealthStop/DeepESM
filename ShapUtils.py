#SHAP tools file

import numpy as np
import shap
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
from shap.plots import waterfall
from shap import kmeans
import pandas as pd


def predict(data, model):
  return model.predict([data[:,i] for i in range(data.shape[1])]).flatten()


def make_shap_plots(model, data, outpath):
    """
    Creates shap summary, bar, heatmap, violin, and layer violin plots.
    """
    def predict_disc1(data):
        return model.predict(data)[0][:,0]

    def predict_disc2(data):
        return model.predict(data)[0][:,1]

    # Need to be conservative about the number of events to make plots
    # For each event, Shap will remove one variable at a time and rerun inferencing
    # Modify numEvents below to change the number of points in each plot
    numEvents = 100

    inputs = data["inputs"]
    inputs = inputs[:numEvents,:]
    names = data["vars"]

    #Making shap values for disc1
    explainer = shap.KernelExplainer(predict_disc1, inputs, feature_names=names)
    shap_values = explainer.shap_values(inputs[:numEvents,:], nsamples=500)
    explanation = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=inputs, feature_names=names)

    #making the plots for disc1
    shap.summary_plot(shap_values, features=inputs[:numEvents,:], feature_names=names, max_display=50)
    save_plot("{}/summary_plot_disc1_plot.png".format(outpath))
  
    shap.summary_plot(shap_values, features=inputs[:numEvents,:], feature_names=names, plot_type="bar", max_display=50)
    save_plot("{}/bar_plot_disc1_plot.png".format(outpath))
  
    shap.plots.heatmap(explanation, max_display=15)
    save_plot("{}/heatmap_plot_disc1_plot.png".format(outpath))
  
    shap.summary_plot(shap_values, features=inputs[:numEvents,:], feature_names=names, plot_type="violin", max_display=50)
    save_plot("{}/violin_plot_disc1_plot.png".format(outpath))
  
    shap.plots.violin(shap_values, features=inputs[:numEvents,:], feature_names=names, plot_type="layered_violin", max_display=50)
    save_plot("{}/layered_violin_plot_disc1_plot.png".format(outpath))


    #Making shap values for disc2
    explainer = shap.KernelExplainer(predict_disc2, inputs, feature_names=names)
    shap_values = explainer.shap_values(inputs[:numEvents,:], nsamples=500)
    explanation = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=inputs, feature_names=names)

    #making the plots for disc2
    shap.summary_plot(shap_values, features=inputs[:numEvents,:], feature_names=names, max_display=50)
    save_plot("{}/summary_plot_disc2_plot.png".format(outpath))
  
    shap.summary_plot(shap_values, features=inputs[:numEvents,:], feature_names=names, plot_type="bar", max_display=50)
    save_plot("{}/bar_plot_disc2_plot.png".format(outpath))
  
    shap.plots.heatmap(explanation, max_display=15)
    save_plot("{}/heatmap_plot_disc2_plot.png".format(outpath))
  
    shap.summary_plot(shap_values, features=inputs[:numEvents,:], feature_names=names, plot_type="violin", max_display=50)
    save_plot("{}/violin_plot_disc2_plot.png".format(outpath))
  
    shap.plots.violin(shap_values, features=inputs[:numEvents,:], feature_names=names, plot_type="layered_violin", max_display=50)
    save_plot("{}/layered_violin_plot_disc2_plot.png".format(outpath))


    #heatmap plots with 20 predictions
    numEvents = 20
    inputs = data["inputs"]
    inputs = inputs[:numEvents,:]
    names = data["vars"]
    explainer = shap.KernelExplainer(predict_disc1, inputs, feature_names=names)
    shap_values = explainer.shap_values(inputs[:numEvents,:], nsamples=500)
    explanation = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=inputs, feature_names=names)

    shap.plots.heatmap(explanation, max_display=15)
    save_plot("{}/heatmap20_plot_disc1_plot.png".format(outpath))

    explainer = shap.KernelExplainer(predict_disc2, inputs, feature_names=names)
    shap_values = explainer.shap_values(inputs[:numEvents,:], nsamples=500)
    explanation = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=inputs, feature_names=names)

    shap.plots.heatmap(explanation, max_display=15)
    save_plot("{}/heatmap20_plot_disc2_plot.png".format(outpath))


def save_plot(name):
  """
  saves plot as 'name'.png
  """
  plt.savefig(name, bbox_inches='tight', format='png')
  plt.close()
