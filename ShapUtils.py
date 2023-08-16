#SHAP tools file

import numpy as np
import pandas as pd
import shap
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
shap.initjs()
from shap.plots import waterfall

def waterfall1(model, data_scaled, data_reg, instance_index):
  """
  Creates a SHAP waterfall plot for a given prediction.
  """
  explainer = shap.KernelExplainer(model.predict, data_scaled)
  shap_values = explainer.shap_values(data_scaled)
  explanation = shap.Explanation(values=shap_values[0][instance_index], base_values=explainer.expected_value[0], data=data_reg.iloc[instance_index], feature_names=data_reg.columns.tolist())
  shap.plots.waterfall(explanation)
  
def waterfall2(model, data, instance_index):
  """
  Creates a SHAP waterfall plot for a given prediction.
  """
  explainer = shap.KernelExplainer(model.predict, data)
  shap_values = explainer.shap_values(data)
  explanation = shap.Explanation(values=shap_values[0][instance_index], base_values=explainer.expected_value[0], data=data.iloc[instance_index], feature_names=data.columns.tolist())
  shap.plots.waterfall(explanation)
