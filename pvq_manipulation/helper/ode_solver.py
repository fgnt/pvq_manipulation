import numpy as np

def ode_solver(f, initial_value, start, stop, steps):
    """
    Simple Euler ODE solver for sampling. 
    Args:
        f: function that takes (y, t) and returns Δy
        initial_value: initial value for y at time start
        start: start time
        stop: stop time
        steps: number of steps to take between start and stop"""
    t = np.linspace(start, stop, steps+1).tolist() 
      
    y = [None] * len(t)
    y[0] = initial_value

    for k in range(steps):
        delta_t = t[k+1] - t[k]  
        y[k+1] = y[k] + delta_t * f(y[k], t[k])  
    return t, y
