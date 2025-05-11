# %% [markdown] 
# # Woehler Analysis Validation

# %% Essential imports
import os
import traceback
import math
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import norm, linregress
import pylife.materialdata.woehler as woehler
from pylife.materiallaws import WoehlerCurve
from scipy import optimize
from scipy import stats
from datetime import datetime


# %% Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# %%
# Constants
N_LCF = 10000  # Pivot point in LCF
NG = 5000000   # Maximum number of cycles


# %% function to load and inspect fatigue test data
def analyze_fatigue_file(file_path):
    """Basic function to load and inspect fatigue test data"""
    try:
        print(f"\nLoading file: {file_path}")
        
        # Read test data
        df_test = pd.read_excel(file_path, sheet_name='Data')
        
        # Rename to 'load' (if column name is 'loads')
        if 'loads' in df_test.columns:
            df_test = df_test.rename(columns={'loads': 'load'})
            
        # Read Jurojin reference values if they exist
        try:
            df_ref = pd.read_excel(file_path, sheet_name='Jurojin_results', header=None)
            
            # Extract reference values
            ref_values = {}
            for idx, row in df_ref.iterrows():
                param_name = row[0]
                param_value = row[1]
                if isinstance(param_value, str) and ',' in param_value:
                    param_value = float(param_value.replace(',', '.'))
                ref_values[param_name] = param_value
        except:
            ref_values = None
            print("No reference values found")
        
        return df_test, ref_values
        
    except Exception as e:
        print(f"Error processing {file_path}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        return None, None