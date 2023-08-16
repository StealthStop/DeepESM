#SHAP tools file

import numpy as np
import shap
import sklearn
import tensorflow as tf
from shap.plots import waterfall
  
def waterfall2(model, data, instance_index):
  """
  Creates a SHAP waterfall plot for a given prediction.
  """
  explainer = shap.KernelExplainer(model.predict, data)
  shap_values = explainer.shap_values(data)
  explanation = shap.Explanation(values=shap_values[0][instance_index], base_values=explainer.expected_value[0], data=data.iloc[instance_index], feature_names=data.columns.tolist())
  shap.plots.waterfall(explanation)
  
def summary_plot1(model, data):
  """
  Creates a SHAP summary plot.
  """
  explainer = shap.KernelExplainer(model.predict, data)
  shap_values = explainer.shap_values(data)
  shap.summary_plot(shap_values, data)
