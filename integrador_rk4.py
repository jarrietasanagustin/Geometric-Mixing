#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:58:25 2019
This program is the temporal integrator of the flow field in bipolar coordinates
The flow field is given by the function flow in jbf.py
Originally the integration was done by means of the Runge-Kutta method
In the new version the integration is done by means of the Scipy odeint function
we switch to a RK4 to speed up the integration and to have more control over the error of the integration
This will be needed for the training of the particle.
@author: jorge arrieta April 2024
"""
import numpy as np
#import scipy.io
from jbf import flow
from rk_4 import rk4_step
import sys


AR_1=2.35#aspect ratio of the ellipsoid
R_1=1.0
R_2=0.3
epsilon_1=0.4#eccentricity of the bipolar coordinates
#epsilon=epsilon_1#*(R_1-R_2)#value of the displacement of the center
b=1/(2*epsilon_1)*np.sqrt((R_1**2+R_2**2-epsilon_1**2)**2-4*R_1**2*R_2**2)
xi_1=np.arcsinh(-b/R_1)
l=-b/np.tanh(xi_1)
#_______________________________________________________________________________
#initial conditions of the flow field
#-----------------------------------------------------------------------------------
x_0=l
y_0=-0.8

#conversion to bipolar coordinates to evaluate the flow field
xi_0=-np.arctanh(2*x_0*b/(x_0**2+y_0**2+b**2))
eta_0=np.mod(np.arctan(2*y_0*b/(x_0**2+y_0**2-b**2)),2*np.pi)




"""
Conversion to bipolar coordinates to evaluate the flow field

"""

#eta_0=5.569166699515978#5.5760;
#xi_0=-0.946892322110823;
theta_0=0.0;#initial angle of the ellipsoid
y0=[xi_0,eta_0,theta_0];
#    if (t_cycle>=0.0 and t_cycle<=(2.0*np.pi)):
omega_1=1.0;
omega_2=0.0;
angle_rot=2.0*np.pi
nt=1000#number of time steps in the first cycle
dt=angle_rot/nt#time step
y=np.zeros((nt+1,3))#array to store the solution of the first cycle
y[0,:]=y0 
#_______________________________________________________________________________
#First cycle. In this cycle the outer cylinder is rotated 2*pi
#_______________________________________________________________________________
time=np.zeros(nt+1,)
kk=0
while time[kk]<angle_rot:
    y[kk+1,:]=np.array(rk4_step(y[kk,:],  dt,time[kk],omega_1,omega_2,R_1,R_2,epsilon_1))
    time[kk+1]=time[kk]+dt
    kk=kk+1 
#_______________________________________________________________________________
#Second cycle. In this cycle the inner cylinder is rotated 2*pi
#_______________________________________________________________________________


#_______________________________________________________________________________
#Third cycle. In this cycle the inner cylinder is rotated back 2*pi
#_______________________________________________________________________________


#_______________________________________________________________________________
#Fourth cycle. In this cycle the outer cylinder is rotated back 2*pi
#_______________________________________________________________________________

#position=np.concatenate((solution.y,solution_1.y,solution_2.y,solution_3.y),axis=1)
#time=np.concatenate((solution.t,solution_1.t,solution_2.t,solution_3.t),axis=0)

#x=-b*np.sinh(position[0,:])/(np.cosh(position[0,:])-np.cos(position[1,:]));
#y=b*np.sin(position[1,:])/(np.cosh(position[0,:])-np.cos(position[1,:]));
# Save solution.y and solution.t as MATLAB binary files
#scipy.io.savemat('/Users/jorge/Documents/Research/Geometric_phase_ellipsoid/programas/Python/program/solution.mat', {'y1': position,'x':x,'y':y,'t':time})