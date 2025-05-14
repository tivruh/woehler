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
    
    
# %% Load a test file

file_path = "All Data/4PB_7.xlsx"
df_test, ref_values = analyze_fatigue_file(file_path)


# %% Prepare data for analysis
df_prepared = df_test[['load', 'cycles', 'censor']]
df_prepared = woehler.determine_fractures(df_prepared, NG)
fatigue_data = df_prepared.fatigue_data


# %% Run normal MaxLikeInf analysis and see the results
analyzer = woehler.MaxLikeInf(fatigue_data)
result = analyzer.analyze()
print(f"Standard analysis results:")
print(f"SD: {result.SD}")
print(f"TS: {result.TS}")
print(f"ND: {result.ND}")
print(f"k_1: {result.k_1}")


# %% Access the optimizer result
# We need to look at private method
# First, get the initial values

SD_start = fatigue_data.fatigue_limit
TS_start = 1.2
print(f"Initial values - SD: {SD_start}, TS: {TS_start}")

# Run the optimizer directly and get full output
from pylife.materialdata.woehler.likelihood import Likelihood
lh = Likelihood(fatigue_data)

# Run the optimization with full output
var_opt = optimize.fmin(
    lambda p: -lh.likelihood_infinite(p[0], p[1]),
    [SD_start, TS_start], 
    disp=True,  # Show convergence messages
    full_output=True
)

# Check the structure of the result
print("\nOptimization result components:")
for i, component in enumerate(var_opt):
    print(f"Component {i}: {type(component)}")


# %% Extract the success indicators

success = (var_opt[4] == 0)  # warnflag - 0 means success
message = var_opt[5] if len(var_opt) > 5 else "No message"
iterations = var_opt[3]
function_calls = var_opt[2]

# Print the optimizer status
print(f"Optimization {'succeeded' if success else 'FAILED'}")
print(f"Message: {message}")
print(f"Iterations: {iterations}")
print(f"Function calls: {function_calls}")

# Check if values are reasonable
min_load = fatigue_data.load.min()
max_load = fatigue_data.load.max()
SD_50 = var_opt[0][0]
TS = var_opt[0][1]

reasonable_values = True
if SD_50 < min_load * 0.5 or SD_50 > max_load * 2.0:
    print(f"WARNING: SD value {SD_50:.2f} outside reasonable range [{min_load*0.5:.2f}, {max_load*2.0:.2f}]")
    reasonable_values = False

if TS < 1.0 or TS > 2.0:
    print(f"WARNING: TS value {TS:.2f} outside typical range [1.0, 2.0]")
    reasonable_values = False

print(f"Values reasonable: {reasonable_values}")

# %%
