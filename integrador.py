#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:58:25 2019
This program is the temporal integrator of the flow field in bipolar coordinates
The flow field is given by the function flow in jbf.py
Originally the integration was done by means of the Runge-Kutta method
In the new version the integration is done by means of the Scipy odeint function
@author: jorge arrieta April 2024
"""
import numpy as np
import scipy.io
from jbf import flow
import sys
import scipy as scp

AR_1=2.35#aspect ratio of the ellipsoid
R_1=1.0
R_2=0.3;
epsilon_1=0.4
#epsilon=epsilon_1#*(R_1-R_2)#value of the displacement of the center
b=1/(2*epsilon_1)*np.sqrt((R_1**2+R_2**2-epsilon_1**2)**2-4*R_1**2*R_2**2);
xi_1=np.arcsinh(-b/R_1);
l=-b/np.tanh(xi_1)
#_______________________________________________________________________________
#initial conditions of the flow field
#-----------------------------------------------------------------------------------
x_0=l;
y_0=-0.8;

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
#_______________________________________________________________________________
#First cycle. In this cycle the outer cylinder is rotated 2*pi
#_______________________________________________________________________________
omega_1=1.0;
omega_2=0.0;
t_span=(0,2.0*np.pi)
solution = (scp.integrate.solve_ivp(flow,t_span,y0,method='RK45',dense_output=True,
                                   args=(omega_1,omega_2,R_1,R_2,epsilon_1),rtol=1e-10,atol=1e-15))
#_______________________________________________________________________________
#Second cycle. In this cycle the inner cylinder is rotated 2*pi
#_______________________________________________________________________________

omega_1=0.0;
omega_2=1.0;
y0=np.squeeze(solution.y[:,-1]);
t_span=(2*np.pi,4.0*np.pi)
solution_1 = (scp.integrate.solve_ivp(flow,t_span,y0,method='RK45',dense_output=True,
                                      args=(omega_1,omega_2,R_1,R_2,epsilon_1),rtol=1e-10,atol=1e-15))

#_______________________________________________________________________________
#Third cycle. In this cycle the inner cylinder is rotated back 2*pi
#_______________________________________________________________________________

omega_1=-1.0;
omega_2=0.0;
y0=np.squeeze(solution_1.y[:,-1]);
t_span=(4*np.pi,6.0*np.pi)
solution_2 = (scp.integrate.solve_ivp(flow,t_span,y0,method='RK45',dense_output=True,
                                      args=(omega_1,omega_2,R_1,R_2,epsilon_1),rtol=1e-10,atol=1e-15))

#_______________________________________________________________________________
#Fourth cycle. In this cycle the outer cylinder is rotated back 2*pi
#_______________________________________________________________________________

omega_1=0.0;
omega_2=-1.0;
y0=np.squeeze(solution_2.y[:,-1]);
t_span=(6*np.pi,8.0*np.pi)
solution_3 = (scp.integrate.solve_ivp(flow,t_span,y0,method='RK45',dense_output=True,
                                      args=(omega_1,omega_2,R_1,R_2,epsilon_1),rtol=1e-10,atol=1e-15))
position=np.concatenate((solution.y,solution_1.y,solution_2.y,solution_3.y),axis=1)
time=np.concatenate((solution.t,solution_1.t,solution_2.t,solution_3.t),axis=0)

x=-b*np.sinh(position[0,:])/(np.cosh(position[0,:])-np.cos(position[1,:]));
y=b*np.sin(position[1,:])/(np.cosh(position[0,:])-np.cos(position[1,:]));
# Save solution.y and solution.t as MATLAB binary files
scipy.io.savemat('/Users/jorge/Documents/Research/Geometric_phase_ellipsoid/programas/Python/program/solution.mat', {'y1': position,'x':x,'y':y,'t':time})