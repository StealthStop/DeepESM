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

  
def waterfall2(model, data, instance_index):
  """
  Creates a SHAP waterfall plot for a given prediction.
  """
  inputs = np.array(data["inputs"])
  names = data["vars"]
  explainer = shap.KernelExplainer(model.predict, inputs, feature_names=names)
  shap_values = explainer.shap_values(inputs)
  explanation = shap.Explanation(values=shap_values[0][instance_index], base_values=explainer.expected_value[0], data=data.iloc[instance_index], feature_names=data.columns.tolist())
  shap.plots.waterfall(explanation)

  save_plot("waterfall_plot.png")

# Wrapper function around Keras predict
# This function needs to be passed into the Kernel Explainer instead of the actual prediction
# See https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/kernel_explainer/Census%20income%20classification%20with%20Keras.html for detailed example
def summary_plt(model, data, numEvents):
    """
    Creates a SHAP summary plot for a given prediction.
    """

    def predict_disc1(data):
        return model.predict(data)[0][:,0]

    def predict_disc2(data):
        return model.predict(data)[0][:,1]

    # Need to be conservative about the number of events to make plots
    # For each event, Shap will remove one variable at a time and rerun inferencing
    # Modify numEvents below to change the number of points in each plot
    numEvents = 50
    inputs = data["inputs"]
    inputs = inputs[:50,:]
    print(inputs)
    names = data["vars"]

    # Use the modified predict_with_model function
    explainer = shap.KernelExplainer(predict_disc1, inputs, feature_names=names)

    # Selecting 50 events to make the waterfall plot with
    # Note that we are using 500 perterbations of each event to estimate the average shapely values for that event
    # Be careful with scaling
    shap_values = explainer.shap_values(inputs[:50,:], nsamples=500)
    #explanation = shap.Explanation(
    #    values=shap_values,
    #    base_values=explainer.expected_value,
    #    data=data[instance_index,:],
    #    feature_names=data["vars"]
    #)
    
    # Changing this to summary plot for now because that seems like the most interesting to me (Bryan)
    # This should be changed back to waterfall if we want to look at individual events
    shap.summary_plot(shap_values, features=inputs[:50,:], feature_names=names)

    save_plot("waterfall_plot.png")
  
def makeSomePlots(model, data):
    """
    Creates a SHAP summary plot and a beeswarm plot plot for the first prediction.
    """

    def predict_disc1(data):
        return model.predict(data)[0][:,0]

    def predict_disc2(data):
        return model.predict(data)[0][:,1]

    # Need to be conservative about the number of events to make plots
    # For each event, Shap will remove one variable at a time and rerun inferencing
    # Modify numEvents below to change the number of points in each plot
    numEvents = 50
    inputs = data["inputs"]
    inputs = inputs[:50,:]
    print(inputs)
    names = data["vars"]

    # Use the modified predict_with_model function
    explainer = shap.KernelExplainer(predict_disc1, inputs, feature_names=names)

    # Selecting 50 events to make the waterfall plot with
    # Note that we are using 500 perterbations of each event to estimate the average shapely values for that event
    # Be careful with scaling
    shap_values = explainer.shap_values(inputs[:50,:], nsamples=500)
    #explanation = shap.Explanation(
    #    values=shap_values,
    #    base_values=explainer.expected_value,
    #    data=data[instance_index,:],
    #    feature_names=data["vars"]
    #)
    
    # Changing this to summary plot for now because that seems like the most interesting to me (Bryan)
    # This should be changed back to waterfall if we want to look at individual events
    # Generate a waterfall plot for the first prediction
  
    shap.summary_plot(shap_values, features=inputs[:50,:], feature_names=names)
    save_plot("summary_plot.png")
  
    shap.plots.beeswarm(shap_values, features=inputs[:50,:], feature_names=names)
    save_plot("beeswarm_plot.png")
  
  
def waterfall4(explanation):
  shap.plots.waterfall(explanation)
  save_plot("waterfall_plot.png")

  
def summary_plot1(model, data):
  """
  Creates a SHAP summary plot.
  """
  inputs = data["inputs"]
  names = data["vars"]
  explainer = shap.KernelExplainer(model.predict, inputs, feature_names=names)
  shap_values = explainer.shap_values(inputs)
  shap.summary_plot(shap_values, inputs, feature_names=names)

def beeswarm_plot(explanation):
  shap.plots.beeswarm(explanation)
  save_plot("beeswarm_plot.png")
  

def save_plot(name):
  """
  saves plot as 'name'.png
  """
  plt.savefig(name, bbox_inches='tight', format='png')
  plt.close()


