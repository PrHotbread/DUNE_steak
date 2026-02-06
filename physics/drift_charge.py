#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:38:58 2023

@author: pinchault

This module simulates the drift of ionization electrons in liquid argon.
It includes:
    - Drift under an electric field
    - Longitudinal and transverse diffusion
    - Temperature dependence of transport coefficients

The transport parameters are derived from fits presented in:
    "Measurement of longitudinal electron diffusion in liquid argon"
    Yichen Li et al.

NB:
- The diffusion model is valid for electric fields between 0.5 and 1.5 kV/cm.
- Compared to Shibamura (1979), transverse diffusion is slightly overestimated
  at high electric field.
"""

import numba as nb
import numpy as np
from numba import njit

from .interpolation import linear_interp_field
from .physics_parameters import (
    longitudinal_diffusion,
    transverse_diffusion,
    coeffs_Astra,
    coeffs_diff,
    coeffs_translate,
    mobility
)


#@nb.jit(nopython = True, parallel = True)
def elec_gun(init_position, ne, n_step, e_drift, vol, dm: float, te: float, x_s, x_v0, x_v1, x_v2, step_x, step_yz, TAr):
    """
    Simulate the drift of multiple ionization electrons in liquid argon.

    Parameters
    ----------
    init_position : array (ne, 3)
        Initial positions of electrons [mm]
    ne : int
        Number of electrons
    n_step : int
        Maximum number of drift steps
    e_drift : array
        Electric field map and boundary conditions
    vol : 3D array
        Volume containing the electric field grid
    dm : float
        Dielectric map
    te : float
        Time step [µs]
    x_s, x_v0, x_v1, x_v2 : float
        X positions of shielding and collection planes
    step_x, step_yz : float
        Spatial discretization [mm]
    TAr : float
        Liquid argon temperature [K]
    """

    # Storage arrays for all electrons
    trajs = np.zeros((ne,3, n_step), dtype = np.float32)
    velocities = np.zeros((ne,3, n_step), dtype = np.float32)
    fields = np.zeros((ne, n_step), dtype = np.float32)
    transverse_energies = np.zeros((ne, n_step), dtype = np.float32)
    mobilities = np.zeros((ne, n_step), dtype = np.float32)
    transverse_coefs = np.zeros((ne, n_step), dtype = np.float32)
    longitudinal_coefs = np.zeros((ne, n_step), dtype = np.float32)
    collects = np.zeros((ne, 5), dtype = np.int32)

    # Parallel loop over electrons
    for e in nb.prange(ne):
        print(e)
        traj, velocity, field, energy, mobility, transverse_coef, longitudinal_coef, collect = drift(
            init_position[e], vol, e_drift, dm, 
            step_x, step_yz, te, n_step, 
            x_s, x_v0, x_v1, x_v2, TAr)
        

        """ Fill arrays with simulation parameters """
        
        trajs[e, 0] = traj[0]
        trajs[e, 1] = traj[1]
        trajs[e, 2] = traj[2]
        fields[e] = field
        transverse_energies[e] = energy
        mobilities[e] = mobility
        transverse_coefs[e] = transverse_coef
        longitudinal_coefs[e] = longitudinal_coef
        velocities[e, 0] = velocity[0]
        velocities[e, 1] = velocity[1]
        velocities[e, 2] = velocity[2]
        collects[e, 0] = collect[0]
        collects[e, 1] = collect[1]
        collects[e, 2] = collect[2]
        collects[e, 3] = collect[3]
        collects[e, 4] = collect[4]
        """
        trajs[e] = traj
        velocities[e] = velocity
        fields[e] = field
        transverse_energies[e] = energy
        mobilities[e] = mobility
        transverse_coefs[e] = transverse_coef
        longitudinal_coefs[e] = longitudinal_coef
        collects[e] = collect
        """
    return trajs, velocities, fields, mobilities, transverse_energies, transverse_coefs, longitudinal_coefs, collects





#@njit
def drift(position, volume, field, dielec, step_x, step_yz, Te, n_step, x_s, x_v0, x_v1, x_v2, TAr):#, step_field, n_step):

    """
    Drift and diffusion of a single electron in liquid argon.

    The electron motion is governed by:
        - Local electric field
        - Field-dependent mobility
        - Longitudinal and transverse diffusion
        - Temperature-dependent transport parameters
    """

    """ Initial position"""

    x, y, z = position
    Ex, Ey, Ez, V = field
    
    
    # Counters for charge collection
    shield = np.int32(0)
    view0 = np.int32(0)
    view1 = np.int32(0)
    view2 = np.int32(0)
    fr4_capture = np.int32(0)
    
    # Trajectory storage
    traj = np.zeros((3, n_step) ,dtype = np.float32)
    XS = np.zeros(n_step, dtype = np.float32)
    YS = np.zeros(n_step, dtype = np.float32)
    ZS = np.zeros(n_step, dtype = np.float32)
                  
    # Step length
    S = np.zeros(n_step, dtype = np.float32)

    # Transverse diffusion displacement
    transv_position = np.zeros(3, dtype = np.float32)

    t = np.zeros(n_step, dtype = np.float32) #time
    e = np.zeros(3, dtype = np.float32)

    # Electron velocity components
    vx = np.zeros(n_step, dtype = np.float32)
    vy = np.zeros(n_step, dtype = np.float32)
    vz = np.zeros(n_step, dtype = np.float32)
    
    test = np.zeros(n_step, dtype = np.float32)

    # Transport parameters
    Dl = np.zeros(n_step, dtype = np.float32) #Longitudinal diffusion coefficient
    Dt = np.zeros(n_step, dtype = np.float32) #Transverse diffusion coefficient
    sigmaL = np.zeros(n_step, dtype = np.float32) #Average longitudinal length
    sigmaT = np.zeros(n_step, dtype = np.float32) #Average transverse length
    mobilities = np.zeros(n_step, dtype = np.float32) #Mobility
    fields = np.zeros(n_step, dtype = np.float32) #Drift field
    energies = np.zeros(n_step, dtype = np.float32) #Effective electron energy
    V = np.zeros(n_step, dtype = np.float32) #Electron velocity

    for ds in range(n_step):

        # Convert position to grid indices
        i = np.int32(np.round(x / step_x))
        j = np.int32(np.round(y / step_yz))
        k = np.int32(np.round(z / step_yz)) 
   
        """ Field interpolation composents """
        
        #Ex0, Ey0, Ez0 = linear_interp_field(x, y, z, volume, i, j, k, step_x, step_yz, Ex, Ey, Ez)
        #Ex0, Ey0, Ez0 = Ex0 * 1e-2, Ey0 * 1e-2, Ez0 * 1e-2 #1e-2 facteur de convertion V/mm -> kv/cm
        Ex0, Ey0, Ez0 = linear_interp_field(x, y, z, volume, i, j, k, step_x, step_yz, Ex, Ey, Ez)

        # Convert field from V/mm to kV/cm
        e[0] = Ex0 * 1e-2 
        e[1] = Ey0 * 1e-2
        e[2] = Ez0 * 1e-2

        E = np.linalg.norm(e)
        #mu = mobility(E, TAr)
        """ Velocity composents for the induced signal calculation """
        vx[ds] = np.float32(mobility(E, TAr) * e[0] * 1e-2) #mm/micros
        vy[ds] = np.float32(mobility(E, TAr) * e[1] * 1e-2) #mm/micros
        vz[ds] = np.float32(mobility(E, TAr) * e[2] * 1e-2) #mm/micros©
        fields[ds] = E
        mobilities[ds] = mobility(E, TAr)
        """ Drift velocity in Liquid Argon """
        V[ds] = mobilities[ds] * fields[ds] * 1e-2 #mm/micros
        #print(V[ds]) check

        """ Long and Transverse Diffusion """
        S[ds] = (V[ds]) * Te # mm
        #Dt[ds], Dl[ds] = 12, 7.2 #cm^2/s
        Dt[ds], Dl[ds] = coeffs_diff(E, TAr)
        #Dt[ds], Dl[ds] = coeffs_Astra(E, TAr)
        #Dt[ds], Dl[ds] = coeffs_translate(fields[ds] * 1e3)

        S[ds], Dl[ds], energies[ds] = longitudinal_diffusion(E, TAr, Te * 1e-6, S[ds], Dl[ds]) # Te micro s --> s

        transv_position, Dt[ds] = transverse_diffusion(Dt[ds], E, e, TAr, Te * 1e-6, transv_position) # Te micro s --> s
        
        tx, ty, tz = transv_position
            
        XS[ds] = x + e[0] / E * S[ds] + tx
        YS[ds] = y + e[1] / E * S[ds] + ty
        ZS[ds] = z + e[2] / E * S[ds] + tz

        x, y, z = XS[ds], YS[ds], ZS[ds]
        """ Looking for the end point """
        if dielec[i,j,k] > 4. and ((XS[ds] > (x_v2 - step_x)) and (XS[ds] < (x_v2 + step_x))):# and XS[ds] < x_v2 + step_x): # charge collected by view 2
            view2 += 1
            break
        elif dielec[i,j,k] > 4. and ((XS[ds] > (x_v1 - step_x)) and (XS[ds] < (x_v1 + step_x))): # charge collected by view 1
            view1 += 1
            break
        elif dielec[i,j,k] > 4. and ((XS[ds] > (x_v0 - step_x)) and (XS[ds] < (x_v0 + step_x))): # charge collected by view 0
            view0 += 1
            break
        elif dielec[i,j,k] > 4. and ((XS[ds] > (x_s - step_x)) and (XS[ds] < (x_s + step_x))): # charge collected by shielding
            shield += 1
            break
        elif dielec[i,j,k] == 4. and (((XS[ds] > ((x_s + x_v0)/2 ) - step_x)) and (XS[ds] < (x_v0 - step_x))):#or (XS[ds] > x_v1 and XS[ds] < (x_v1 + x_v2)/2)) :
            fr4_capture += 1
            break
        elif dielec[i,j,k] == -9999: # The end point for the electron drifting at large scale
            break
    """ Velocity and position storage"""
    
    traj = [XS, YS, ZS]
    velo_field = [vx, vy, vz]
    collect = [view0, view1, view2, shield, fr4_capture]
    return  traj, velo_field, fields, energies, mobilities, Dt, Dl, collect
