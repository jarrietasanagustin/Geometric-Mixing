"""
This is a RK4 solver for the geometric phase of  the sheproid. 
The aim is to speed up the computation of the trajectory to train the system 
Jorge Arrieta, May 2026
University of the Balearic Islands
"""


import numpy as np
from jbf import flow
def rk4_step(x,  dt,t ,omega_1,omega_2,R_1,R_2,epsilon_1):
    """Perform a single RK4 step."""
    k1 = np.array(flow(t,x,omega_1,omega_2,R_1,R_2,epsilon_1))
    
    x_aprox_1=x+0.5*dt*k1 
    k2 = np.array(flow(t,x_aprox_1,omega_1,omega_2,R_1,R_2,epsilon_1))
    k3 =np.array(flow(t,x + 0.5 * dt * k2,omega_1,omega_2,R_1,R_2,epsilon_1))
    k4 = np.array(flow(t,x + dt * k3,omega_1,omega_2,R_1,R_2,epsilon_1))
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)