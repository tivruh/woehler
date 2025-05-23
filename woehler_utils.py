# woehler_utils.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import optimize

def run_optimization_with_tracking(likelihood_obj, initial_values, method='nelder-mead', bounds=None):
    """Run optimization with tracking for either method and plot results
    
    Parameters:
    -----------
    likelihood_obj : Likelihood object
        The likelihood object from PyLife
    initial_values : list
        Initial parameter values [SD_start, TS_start]
    method : str
        'nelder-mead' or 'l-bfgs-b'
    bounds : list of tuples, optional
        Required for L-BFGS-B: [(SD_min, SD_max), (TS_min, TS_max)]
    """
    # Create list to store optimization steps
    optimization_steps = []
    
    # Define objective function with tracking
    def tracked_objective(p):
        # Calculate likelihood
        likelihood = likelihood_obj.likelihood_infinite(p[0], p[1])
        
        # Store current step
        optimization_steps.append({
            'Step': len(optimization_steps) + 1,
            'SD': p[0],
            'TS': p[1],
            'Likelihood': likelihood
        })
        
        # Return negative for minimization
        return -likelihood
    
    # Initial values
    SD_start, TS_start = initial_values
    print(f"Initial values - SD: {SD_start:.2f}, TS: {TS_start:.2f}")
    
    # Run appropriate optimizer
    if method.lower() == 'nelder-mead':
        result = optimize.fmin(
            tracked_objective,
            initial_values,
            disp=True,
            full_output=True
        )
        
        # Extract results
        SD, TS = result[0]
        warnflag = result[4]
        message = result[5] if len(result) > 5 else "No message"
        success = (warnflag == 0)
        
        # Map warnflag to meaning
        warnflag_meanings = {
            0: "Success - optimization converged",
            1: "Maximum number of iterations/evaluations reached",
            2: "Function values not changing (precision loss)",
            3: "NaN result encountered"
        }
        status_text = warnflag_meanings.get(warnflag, "Unknown")
        
    elif method.lower() == 'l-bfgs-b':
        if bounds is None:
            raise ValueError("Bounds required for L-BFGS-B method")
            
        result = optimize.minimize(
            tracked_objective,
            initial_values,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Extract results
        SD, TS = result.x
        success = result.success
        message = result.message
        status_text = "Success - optimization converged" if success else "Failed to converge"
    
    # Print results
    print(f"\n{method.upper()} optimization status: {status_text}")
    print(f"Message: {message}")
    print(f"Final values - SD: {SD:.2f}, TS: {TS:.2f}")
    
    # Calculate slog
    slog = np.log10(TS)/2.5361
    print(f"Calculated slog: {slog:.4f}")
    
    # Check if values are reasonable
    min_load = likelihood_obj._fd.load.min()
    max_load = likelihood_obj._fd.load.max()
    
    reasonable_values = True
    if SD < min_load * 0.5 or SD > max_load * 2.0:
        print(f"WARNING: SD value {SD:.2f} outside reasonable range [{min_load*0.5:.2f}, {max_load*2.0:.2f}]")
        reasonable_values = False
    
    if TS < 1.0 or TS > 10.0:
        print(f"WARNING: TS value {TS:.2f} outside typical range [1.0, 10.0]")
        reasonable_values = False
    
    print(f"Values reasonable: {reasonable_values}")
    
    # Plot convergence
    plot_optimization_convergence(optimization_steps, method)
    
    # Return results
    return {
        'method': method,
        'SD': SD, 
        'TS': TS,
        'success': success,
        'message': message,
        'reasonable_values': reasonable_values,
        'optimization_steps': optimization_steps
    }
    

def plot_optimization_convergence(steps, method="optimization"):
    """Plot optimization convergence using original validation style"""
    # Convert to DataFrame
    df_steps = pd.DataFrame(steps)
    
    # Set proper method name for display
    if method.lower() == 'nelder-mead':
        display_method = 'Nelder-Mead'
    elif method.lower() == 'l-bfgs-b':
        display_method = 'L-BFGS-B'
    else:
        display_method = method.capitalize()
    
    # Create figure
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
        title=f'{display_method} Convergence',
        xaxis=dict(title='Iteration'),
        yaxis=dict(
            title='Likelihood',
            domain=[0, 0.9],
            gridcolor='lightgray'
        ),
        yaxis2=dict(
            title='SD Value',
            side='right',
            overlaying='y',
            anchor='x',
            autorange=True,
            gridcolor='lightgray'
        ),
        yaxis3=dict(
            title='TS Value',
            side='right',
            overlaying='y',
            position=0.9,
            anchor='free',
            autorange=True,
            gridcolor='lightgray'
        ),
        legend=dict(x=0.01, y=0.99),
        width=900,
        height=600,
        plot_bgcolor='rgba(240, 240, 250, 0.8)'
    )
    
    return fig