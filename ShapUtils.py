#SHAP tools file

import numpy as np
import shap
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
from shap.plots import waterfall
import pandas as pd

  
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

def predict_with_model(model, inputs):
    # Ensure that 'inputs' is in the correct format and shape
    predictions = model.predict(inputs)
    predictions = np.array(predictions)

    return predictions

def waterfall3(model, data, instance_index):
    """
    Creates a SHAP waterfall plot for a given prediction.
    """
    inputs = np.array(data["inputs"])
    names = data["vars"]

    # Use the modified predict_with_model function
    explainer = shap.KernelExplainer(predict_with_model(model, x), inputs, feature_names=names)
    shap_values = explainer.shap_values(inputs)
    explanation = shap.Explanation(
        values=shap_values[0][instance_index],
        base_values=explainer.expected_value[0],
        data=data.iloc[instance_index],
        feature_names=data.columns.tolist()
    )
    
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

def save_plot(name):
  """
  saves plot as 'name'.png
  """
  plt.savefig(name, bbox_inches='tight', format='png')
  plt.close()


