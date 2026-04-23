#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:40:22 2019
This program has been updated in 2024. All the variables were checked and 
evaluated from the analytical solution of Ballal and Rivlin making use of 
Mathematica. 
Also the vorticity and the strain tensor have verified with the analytical
solution of the journal bearing flow.
Finally it seems that there was a mistake in the calculation of the strain tensor 
in the rotatio which needs to be further checked.
@author: jorge arrieta April 2024
"""
import numpy as np
import scipy as scp
import sys
def flow(t,state,omega_1,omega_2,R_1,R_2,epsilon_1):
    xi,eta,theta=state
    epsilon=epsilon_1#*(R_1-R_2)#value of the displacement of the center
    AR=10
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Definition of the flow in bipolar coordinates
#thes constants were checked in April 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    strain_rate=np.zeros((2,2),dtype=np.double)
    coordinate_matrix=np.zeros((2,2),dtype=np.double)
    b=1/(2*epsilon)*np.sqrt((R_1**2+R_2**2-epsilon**2)**2-4*R_1**2*R_2**2);
    xi_1=np.arcsinh(-b/R_1);
    xi_2=np.arcsinh(-b/R_2);
    l=-b/np.tanh(xi_1)
    

    H=b/(np.cosh(xi)-np.cos(eta));
#############################################################################
#Constants of the stream function of the journal bearing flow
#constants were checked in April 2024
#############################################################################


    Delta=(xi_1-xi_2)**2-(np.sinh(xi_1-xi_2))**2
    Delta_bar=(xi_1-xi_2)*np.cosh(xi_1-xi_2)-np.sinh(xi_1-xi_2)
    Delta_star=np.sinh(xi_1-xi_2)*(2*np.sinh(xi_1)*np.sinh(xi_2)\
    *np.sinh(xi_1-xi_2)-(xi_1-xi_2)*((np.sinh(xi_1))**2+(np.sinh(xi_2))**2))
    h_1=(xi_1-xi_2)*np.sinh(xi_1)-np.sinh(xi_2)*np.sinh(xi_1-xi_2);
    h_2=-(xi_1-xi_2)*np.sinh(xi_2)+np.sinh(xi_1)*np.sinh(xi_1-xi_2);
    h_3=xi_1*np.sinh(xi_2)*np.sinh(xi_1-xi_2)-xi_2*(xi_1-xi_2)*np.sinh(xi_1);
    h_4=-xi_2*np.sinh(xi_1)*np.sinh(xi_1-xi_2)+xi_1*(xi_1-xi_2)*np.sinh(xi_2);
    h_5=-xi_1*np.cosh(xi_2)*np.sinh(xi_1-xi_2)+xi_2*(xi_1-xi_2)*np.cosh(xi_1);
    h_6=xi_2*np.cosh(xi_1)*np.sinh(xi_1-xi_2)-xi_1*(xi_1-xi_2)*np.cosh(xi_2);
    h_7=np.sinh(xi_2)*np.cosh(xi_1)*np.sinh(xi_1-xi_2)\
    +1/2*xi_1*np.sinh(2*xi_2)-1/2*xi_2*np.sinh(2*xi_1)-xi_2*(xi_1-xi_2)
    h_8=-np.cosh(xi_1)*np.cosh(xi_2)*np.sinh(xi_1-xi_2)\
        +xi_2*(np.cosh(xi_1))**2-xi_1*(np.cosh(xi_2))**2

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    f_1=1/Delta*(Delta_bar/Delta_star*h_1*h_7+h_3);
    f_2=1/Delta*(Delta_bar/Delta_star*h_2*h_7+h_4);
    f_3=1/Delta*(Delta_bar/Delta_star*h_1*h_8+h_5);
    f_4=1/Delta*(Delta_bar/Delta_star*h_2*h_8+h_6);
    f_5=h_1/Delta_star*np.cosh(xi_1-xi_2);
    f_6=h_2/Delta_star*np.cosh(xi_1-xi_2);
    f_7=-np.sinh(xi_2)/Delta_star*(np.sinh(xi_1-xi_2))**2
    f_8=-np.sinh(xi_1)/Delta_star*(np.sinh(xi_1-xi_2))**2
    f_9=-1/2*h_1/Delta_star*np.sinh(xi_1+xi_2);
    f_10=-1/2*h_2/Delta_star*np.sinh(xi_1+xi_2);
    f_11=1/2*h_1/Delta_star*np.cosh(xi_1+xi_2);
    f_12=1/2*h_2/Delta_star*np.cosh(xi_1+xi_2);
    f_13=h_1/(2*Delta_star)*(np.sinh(xi_1-xi_2)+2*xi_2*np.cosh(xi_1-xi_2))
    f_14=h_2/(2*Delta_star)*(np.sinh(xi_1-xi_2)+2*xi_2*np.cosh(xi_1-xi_2))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    A_0=f_1*omega_1*R_1+f_2*omega_2*R_2;
    B_0=f_3*omega_1*R_1+f_4*omega_2*R_2;
    C_0=f_5*omega_1*R_1+f_6*omega_2*R_2;
    D_0=f_7*omega_1*R_1+f_8*omega_2*R_2;

    A_1=f_9*omega_1*R_1+f_10*omega_2*R_2;
    B_1=f_11*omega_1*R_1+f_12*omega_2*R_2;
    C_1=-f_5*omega_1*R_1-f_6*omega_2*R_2;
    D_1=f_13*omega_1*R_1+f_14*omega_2*R_2;

    F_0=(A_0+C_0*xi)*np.cosh(xi)+(B_0+D_0*xi)*np.sinh(xi)
    F_1=A_1*np.cosh(2*xi)+B_1*np.sinh(2*xi)+C_1*xi+D_1
    F_1_f=F_1
    phi_0=F_0+F_1*np.cos(eta)

    
    
    s=np.sinh(xi)*np.sin(eta);
    c=np.cosh(xi)*np.cos(eta)-1;
    c_xi=np.sinh(xi)*np.cos(eta);
    s_xi=np.cosh(xi)*np.sin(eta);
    c_eta=-np.cosh(xi)*np.sin(eta);
    s_eta=np.sinh(xi)*np.cos(eta);
    #__________________________________________________________________________
    #first order derivatives of the scale factor H respect to xi and eta
    #__________________________________________________________________________
    h_eta=-b*np.sin(eta)/(np.cosh(xi) - np.cos(eta))**2
    h_xi=-b*np.sinh(xi)/(np.cosh(xi)-np.cos(eta))**2
    #__________________________________________________________________________
    #derivatives of the functions that define the stream function
    #__________________________________________________________________________
    df0_dxi=np.sinh(xi)*(A_0+xi*C_0+D_0) +np.cosh(xi)*(B_0+C_0+xi*D_0);
    df_1_dxi=2*np.sinh(2*xi)*A_1+2*np.cosh(2*xi)*B_1+C_1
    phi_0_xi=df0_dxi+df_1_dxi*np.cos(eta);
    phi_0_eta=-F_1*np.sin(eta);
    #__________________________________________________________________________
    #derivatives of the stream function
    #__________________________________________________________________________
    #diff_psi_eta=phi_0*h_eta+H*phi_0_eta;
    #diff_psi_xi=h_xi*phi_0+H*phi_0_xi;
    diff_psi_eta=-b*phi_0*np.sin(eta)/(np.cosh(xi)-np.cos(eta))**2-H*F_1_f*np.sin(eta)
    diff_psi_xi=-b*phi_0*np.sinh(xi)/(np.cosh(xi)-np.cos(eta))**2+\
	H*(C_0*np.cosh(xi)+(A_0+C_0*xi)*np.sinh(xi)+D_0*np.sinh(xi)+\
	(B_0+D_0*xi)*np.cosh(xi)+(2*A_1*np.sinh(2*xi)+2*B_1*np.cosh(2*xi)+C_1)*np.cos(eta))
    #__________________________________________________________________________
    #now we use the derivatives of the stream function to calculate the velocity
    #field both in bipolar coordinates and in cartesian coordinates
    #__________________________________________________________________________
    #Bipolar coordinates
    u_xi=1/H*diff_psi_eta;
    u_eta=-1/H*diff_psi_xi;
    
 ################################################################################
#Derivatives of the scale factor of the coordinated system needed to 
#calculate the strain rate tensor and the vorticity
#################################################################################
    h_eta_2=-b*(np.cos(eta)/(np.cosh(xi)-np.cos(eta))**2\
                -2*(np.sin(eta))**2/(np.cosh(xi)-np.cos(eta))**3)
    h_eta_xi=2*b*np.sin(eta)*np.sinh(xi)/(np.cosh(xi)-np.cos(eta))**3
    h_xi_2=-b*(np.cosh(xi)/(np.cosh(xi)-np.cos(eta))**2\
               -2*(np.sinh(xi))**2/(np.cosh(xi)-np.cos(eta))**3)
    df0_dxi_xi=(np.cosh(xi)*(A_0+xi*C_0+ 2*D_0)+np.sinh(xi)*(B_0 + 2*C_0 + xi*D_0));
    df1_dxi_xi=4*np.cosh(2*xi)*A_1 + 4*np.sinh(2*xi)*B_1;
##__________________________________________________________________________
##second order derivatives of the stream function respect to xi
##__________________________________________________________________________
    phi_0_xi_xi=df0_dxi_xi+df1_dxi_xi*np.cos(eta);
    psi_xi_2=h_xi_2*phi_0+2*h_xi*phi_0_xi+H*phi_0_xi_xi;
##__________________________________________________________________________
##second order derivatives of the stream function respect to eta
##__________________________________________________________________________
    phi_0_eta_2=-F_1*np.cos(eta);   
    phi_0_eta_xi=-df_1_dxi*np.sin(eta);    
    psi_eta_2=h_eta_2*phi_0+2*h_eta*phi_0_eta+H*phi_0_eta_2;
    diff_psi_dxi_deta=(h_eta_xi*phi_0+h_eta*phi_0_xi+h_xi*phi_0_eta\
                       +H*phi_0_eta_xi);
##__________________________________________________________________________
##Vorticity
##__________________________________________________________________________
    #u_eta_xi=-(psi_xi_2*H-h_xi*diff_psi_xi)/H**2
    #u_xi_eta=(psi_eta_2*H-h_eta*diff_psi_eta)/H**2
    #curl_u=(H*(u_eta_xi-u_xi_eta)+h_xi*u_eta-h_eta*u_xi)/H**2 
    curl_u=-(psi_xi_2+psi_eta_2)/H**2;
    vort=1/2*curl_u;
    
 ##################################################################################
#Definition of the components of the strain rate tensor in cartesian coordinates
###################################################################################   
    epsilon_xx=1/b**2*(diff_psi_xi*(-c*s_xi-s*s_eta)\
    -c*s*psi_xi_2+diff_psi_dxi_deta*(c**2-s**2)+diff_psi_eta*(c*c_xi+s*c_eta)+s*c*psi_eta_2);  
    #---------------------------------------------------------------
    epsilon_yy=-1/b**2*(diff_psi_xi*(-s*c_xi+c*c_eta)-psi_xi_2*s*c+\
                        diff_psi_dxi_deta*(c**2-s**2)+diff_psi_eta*(-s*s_xi+c*s_eta)+s*c*psi_eta_2);
    #---------------------------------------------------------------
    dux_dy=1/b**2*(diff_psi_xi*(s*s_xi-c*s_eta)+s**2*psi_xi_2\
    -2*c*s*diff_psi_dxi_deta+(c*c_eta-s*c_xi)*diff_psi_eta+c**2*psi_eta_2); 
    #---------------------------------------------------------------
    duy_dx=-1/b**2*(diff_psi_xi*(c*c_xi+s*c_eta)\
    +c**2*psi_xi_2+2*c*s*diff_psi_dxi_deta+\
    (c*s_xi+s*s_eta)*diff_psi_eta+s**2*psi_eta_2);
    epsilon_xy=1/2*(dux_dy+duy_dx);
    #---------------------------------------------------------------
    #strain_rate=np.array([[epsilon_xx, epsilon_xy],[epsilon_xy, epsilon_yy]]);
    strain_rate[0,0]=epsilon_xx;
    strain_rate[0,1]=epsilon_xy;
    strain_rate[1,0]=epsilon_xy;
    strain_rate[1,1]=epsilon_yy;
    coordinate_matrix[0,0]=np.cos(theta);
    coordinate_matrix[0,1]=-np.sin(theta);
    coordinate_matrix[1,0]=np.sin(theta);
    coordinate_matrix[1,1]=np.cos(theta);
#    coordinate_matrix=(np.array([[np.cos(theta), np.sin(theta)]
#                                 ,[-np.sin(theta), np.cos(theta)]]));
    alpha=np.matmul(strain_rate,np.transpose(coordinate_matrix))
    #print(strain_rate)
    strain_1=coordinate_matrix@(strain_rate@np.transpose(coordinate_matrix))#np.matmul(coordinate_matrix,alpha)
    epsilon_xy_prima=strain_1[0,1];
    omega_spheroid=(vort+(AR**2-1)/(AR**2+1)*epsilon_xy_prima);
    dydt = [1/H*u_xi, 1/H*u_eta, omega_spheroid]
    return dydt