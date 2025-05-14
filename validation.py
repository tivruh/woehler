# %% [markdown] 
# # Woehler Analysis Validation

# %% imports

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
from pylife.materialdata.woehler.likelihood import Likelihood
from woehler_utils import *

from scipy import optimize
from scipy import stats
from datetime import datetime


# %% Constants, display options
# Sheet containing data must be named 'Data'

file_path = "All Data/4PB_7.xlsx"

N_LCF = 10000  # Pivot point in LCF
NG = 5000000   # Maximum number of cycles


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


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


# %% Prepare data for analysis

df_test, ref_values = analyze_fatigue_file(file_path)
df_prepared = df_test[['load', 'cycles', 'censor']]
df_prepared = woehler.determine_fractures(df_prepared, NG)
fatigue_data = df_prepared.fatigue_data


# %% CONTROL: Run normal MaxLikeInf analysis BEFORE running further blocks
analyzer = woehler.MaxLikeInf(fatigue_data)
result = analyzer.analyze()
print(f"Pylife results out of the box:")
print(f"SD: {result.SD}")
print(f"TS: {result.TS}")
print(f"ND: {result.ND}")
print(f"k_1: {result.k_1}")


# %% Track optimization progress with MaxLikeInf

# Create list to store optimization steps
optimization_steps = []

# Create likelihood object
lh = Likelihood(fatigue_data)

# Run Nelder-Mead first
nm_results = run_optimization_with_tracking(lh, [fatigue_data.fatigue_limit, 1.2], method='nelder-mead')

# Try L-BFGS-B if needed
if not nm_results['success'] or not nm_results['reasonable_values']:
    print("\nNelder-Mead failed or produced unreasonable values. Trying L-BFGS-B...")
    bounds = [(fatigue_data.load.min() * 0.5, fatigue_data.load.max() * 2.0), (1.0, 10.0)]
    
    lbfgs_results = run_optimization_with_tracking(
        lh, 
        [fatigue_data.fatigue_limit, 1.2], 
        method='l-bfgs-b',
        bounds=bounds
    )
    
    # Compare results
    print("\nComparison of methods:")
    print(f"Nelder-Mead: SD={nm_results['SD']:.2f}, TS={nm_results['TS']:.2f}, Success={nm_results['success']}")
    print(f"L-BFGS-B: SD={lbfgs_results['SD']:.2f}, TS={lbfgs_results['TS']:.2f}, Success={lbfgs_results['success']}")


# %% [markdown] about fmin() output 
# ## SciPy fmin optimization output
# 
# PyLife's Woehler module uses SciPy's `optimize.fmin()` function (Nelder-Mead algorithm) in MaxLikeInf to determine SD (endurance limit) and TS (scatter).
# 
# [Scipy fmin output](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html)
#
# ### PyLife's original MaxLikeInf implementation (__max_likelihood_inf_limit method)
# ```python
# SD_start = self._fd.fatigue_limit
# TS_start = 1.2

# var_opt = optimize.fmin(
#     lambda p: -self._lh.likelihood_infinite(p[0], p[1]),
#     [SD_start, TS_start], 
#     disp=False,  # Note: PyLife suppresses output
#     full_output=True
# )
# extracts values withoutchecking warnflags
# SD_50 = var_opt[0][0]
# TS = var_opt[0][1]
#
# return SD_50, TS
# 
# # likelihood_infinite(self, SD, TS): 
# Calculates likelihood for points in the infinite zone (horizontal part of curve)
# 
# infinite_zone = self._fd.infinite_zone
# std_log = scattering_range_to_std(TS)
# t = np.logical_not(self._fd.infinite_zone.fracture).astype(np.float64)
# likelihood = stats.norm.cdf(np.log10(infinite_zone.load/SD), scale=abs(std_log))
# non_log_likelihood = t+(1.-2.*t)*likelihood
# if non_log_likelihood.eq(0.0).any():
#     return -np.inf
# return np.log(non_log_likelihood).sum()
# ```
#
# When using `full_output=True`, fmin returns a tuple where:
# - `var_opt[0]` = optimized parameters (SD and TS)
# - `var_opt[1]` = final optimized function value
# - `var_opt[2]` = number of function evaluations
# - `var_opt[3]` = number of iterations
# - `var_opt[4]` = warnflag (0=success, 1=max iterations, 2=function not improving)
# - `var_opt[5]` = termination message
# 
# The warnflag at index 4 is the critical value for determining success:
# - warnflag = 0: Optimization succeeded (converged properly)
# - warnflag = 1: Maximum number of iterations reached without convergence
# - warnflag = 2: Function evaluations not changing (possible precision loss)