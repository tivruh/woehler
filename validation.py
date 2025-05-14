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
from scipy import optimize
from scipy import stats
from datetime import datetime


# %% Constants, display options
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
    

# %% [markdown] about fmin() output 
# ## SciPy fmin optimization output
# 
# PyLife's Woehler module uses SciPy's `optimize.fmin()` function (Nelder-Mead algorithm) in MaxLikeInf to determine SD (endurance limit) and TS (scatter).
# 
# ![Scipy fmin output](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html)
#
# !TODO: add codeblock for fmin usage in MaxLikeInf
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

    
    
# %% Load a test file
# Sheet containing data must be named 'Data'

file_path = "All Data/250514_Evaluation.xlsx"
df_test, ref_values = analyze_fatigue_file(file_path)


# %% Prepare data for analysis
df_prepared = df_test[['load', 'cycles', 'censor']]
df_prepared = woehler.determine_fractures(df_prepared, NG)
fatigue_data = df_prepared.fatigue_data


# %% CONTROL: Run normal MaxLikeInf analysis BEFORE running further blocks
analyzer = woehler.MaxLikeInf(fatigue_data)
result = analyzer.analyze()
print(f"Standard analysis results:")
print(f"SD: {result.SD}")
print(f"TS: {result.TS}")
print(f"ND: {result.ND}")
print(f"k_1: {result.k_1}")


# %% Track optimization progress with MaxLikeInf

# Create list to store optimization steps
optimization_steps = []

# Create likelihood object for the fatigue data
lh = Likelihood(fatigue_data)

# Define objective function that records steps
def objective_function(p):
    # Calculate likelihood
    likelihood = lh.likelihood_infinite(p[0], p[1])
    
    # Store current step
    optimization_steps.append({
        'Step': len(optimization_steps) + 1,
        'SD': p[0],
        'TS': p[1],
        'Likelihood': likelihood
    })
    
    # For minimization, return negative likelihood
    return -likelihood


# %% Run optimization with tracking

# Get initial values from fatigue_data
SD_start = fatigue_data.fatigue_limit
TS_start = 1.2

print(f"Initial values - SD: {SD_start:.2f}, TS: {TS_start:.2f}")

# Run optimization with our tracking function
var_opt = optimize.fmin(
    objective_function,
    [SD_start, TS_start],
    disp=True,
    full_output=True
)

# Extract results
SD = var_opt[0][0]
TS = var_opt[0][1]
warnflag = var_opt[4]  # Get the warnflag directly 
message = var_opt[5] if len(var_opt) > 5 else "No message"

# Print the optimizer status with more detail
warnflag_meanings = {
    0: "Success - optimization converged",
    1: "Maximum number of iterations/evaluations reached",
    2: "Function values not changing (precision loss)",
    3: "NaN result encountered"
}

print(f"\nOptimization status: {warnflag_meanings.get(warnflag, 'Unknown')}")
print(f"Raw warnflag value: {warnflag}")
print(f"Message: {message}")
print(f"Final values - SD: {SD:.2f}, TS: {TS:.2f}")

# Calculate slog from TS
slog = np.log10(TS)/2.5361
print(f"Calculated slog: {slog:.4f}")

# Check if values are reasonable
min_load = fatigue_data.load.min()
max_load = fatigue_data.load.max()

reasonable_values = True
if SD < min_load * 0.5 or SD > max_load * 2.0:
    print(f"WARNING: SD value {SD:.2f} outside reasonable range [{min_load*0.5:.2f}, {max_load*2.0:.2f}]")
    reasonable_values = False

if TS < 1.0 or TS > 10.0:
    print(f"WARNING: TS value {TS:.2f} outside typical range [1.0, 10.0]")
    reasonable_values = False

print(f"Values reasonable: {reasonable_values}")


# %% Plot convergence

#!TODO: move this to a different file

# Convert to DataFrame for easier plotting
df_steps = pd.DataFrame(optimization_steps)

# Create convergence plot
fig = go.Figure()

# Add likelihood curve
fig.add_trace(go.Scatter(
    x=df_steps['Step'], 
    y=df_steps['Likelihood'],
    mode='lines+markers', 
    name='Likelihood'
))

# Add SD parameter convergence
fig.add_trace(go.Scatter(
    x=df_steps['Step'], 
    y=df_steps['SD'],
    mode='lines+markers', 
    name='SD (Endurance Limit)',
    yaxis='y2'  # Use secondary axis
))

# Add TS parameter convergence
fig.add_trace(go.Scatter(
    x=df_steps['Step'], 
    y=df_steps['TS'],
    mode='lines+markers', 
    name='TS (Scatter)',
    yaxis='y3'  # Use tertiary axis
))

# Update layout with multiple y-axes
fig.update_layout(
    title='Optimization Convergence',
    xaxis=dict(title='Iteration'),
    yaxis=dict(title='Likelihood'),
    yaxis2=dict(
        title='SD Value',
        overlaying='y',
        side='right'
    ),
    yaxis3=dict(
        title='TS Value',
        overlaying='y',
        anchor='free',
        side='right',
        position=0.85
    ),
    legend=dict(x=0.01, y=0.99),
    width=900,
    height=600
)

fig.show()
# %%
