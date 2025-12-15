#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb 16 09:33:56 2023

This script performs a 3D Drift Field calculation using the Finite Difference Method (FDM).
The computation is based on the anode technology using a stack of perforated 
Printed Circuit Boards (PCBs) for the Single Phase Vertical Drift from the DUNE project.

@author: pinchault
"""

import numpy as np
import argparse
import time as time
from calc_field import finite_difference as fdm
from calc_field import elecfield as field
from settings import parameters
import geometry as geo
from plotting import field as plot
from physics import physics_parameters
from settings.gen_file import save_field


# -------------------------------
# Argument parsing
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    '-conv', 
    help="Stopping convergence criterion for the Finite Difference Method",
    default=1e-1, 
    required=True
)
parser.add_argument(
    '-namefile', 
    help="Name of the output binary file", 
    required=True
)
parser.add_argument(
    '-s', 
    dest='setup', 
    help='Optional setup file', 
    default=''
)

args = parser.parse_args()
setup = args.setup

# -------------------------------
# Geometry and parameters
# -------------------------------
params = {}
parameters.set_geometry(params, setup)


# -------------------------------
# Helper functions
# -------------------------------
def set_V_electrode(field, idx, pattern, cond):
    """Apply electrode potential conditions to the field matrix."""
    field[idx] = field[idx] * pattern + cond
    return field
    

def set_V_boundary(Field, Vbot, Vtop):
    """Set boundary potentials for the top and bottom planes."""
    Field[0] = Vbot
    Field[-1] = Vtop
    return Field


def init_cond():
    """
    Initialize the electric potential and boundary matrices.
    
    Returns
    -------
    field : numpy.ndarray
        3D matrix containing the initial potential distribution.
    boundary : numpy.ndarray
        3D matrix encoding boundary conditions.
    diel : numpy.ndarray
        Dielectric constant distribution.
    """
    
    pattern, pattern_shifted = geo.set_pattern(params)
    
    # Define electrode potential masks
    mask_shield = physics_parameters.electrode(params['V_shield'], pattern)
    mask_induc1 = physics_parameters.electrode(params['V_ind1'], pattern_shifted)
    mask_induc2 = physics_parameters.electrode(params['V_ind2'], pattern)
    mask_coll = physics_parameters.electrode(params['V_coll'], pattern)
    
    # Build linear potential along drift direction
    V1 = np.linspace(params['V_drift'], params['V_shield'], parameters.index(params['L_drift'], params['step']*params['delta_x']))
    V2 = np.linspace(params['V_shield'], params['V_ind1'], parameters.index(params['L_pcb1'], params['step']*params['delta_x']))
    V3 = np.linspace(params['V_ind1'], params['V_ind2'], parameters.index(params['L_gap'], params['step']*params['delta_x']))
    V4 = np.linspace(params['V_ind2'], params['V_coll'], parameters.index(params['L_pcb2'], params['step']*params['delta_x']))
    V5 = np.linspace(params['V_coll'], params['V_ground'], parameters.index(params['L_ground'], params['step']*params['delta_x']))
    V = np.array([])
    V = np.append(V, np.append(np.append(np.append(V1, V2), np.append(V3, V4)), V5))
    
    # Initialize 3D field
    field = np.tile(V, (params['nz'] + 1, params['ny'] + 1, 1)).T
    field = set_V_boundary(field, params['V_drift'], params['V_ground'])
    
    # Apply electrode potentials
    field = set_V_electrode(field, params['idx_shield'], pattern, mask_shield)
    field = set_V_electrode(field, params['idx_ind1'], pattern_shifted, mask_induc1)
    field = set_V_electrode(field, params['idx_ind2'], pattern, mask_induc2)
    field = set_V_electrode(field, params['idx_coll'], pattern, mask_coll)
    
    # Compute dielectric map
    diel = physics_parameters.dielectric_matrix(pattern, pattern_shifted, params)
    
    # Define boundary condition mask
    boundary = np.ones(field.shape, dtype=np.float32)
    boundary[:,:,0] = 2
    boundary[:,:,-1] = 3
    boundary[:,0,:] = 4
    boundary[:,-1,:] = 5
    boundary[:,0,0] = 6
    boundary[:,0,-1] = 7
    boundary[:,-1,0] = 8
    boundary[:,-1,-1] = 9
    
    # Apply boundary mask to electrodes
    boundary[params['idx_shield']] *= pattern
    boundary[params['idx_ind1']] *= pattern_shifted
    boundary[params['idx_ind2']] *= pattern
    boundary[params['idx_coll']] *= pattern
    
    return field, boundary, diel
    
    
def FDMiteration(field: float, B: float, D: float, stopping_criterion=np.float32(args.conv), *kwargs):
    """
    Perform Finite Difference Method (FDM) iterations until convergence.

    Parameters
    ----------
    field : numpy.ndarray
        Initial potential distribution (0th iteration).
    B : numpy.ndarray
        Boundary condition mask.
    D : numpy.ndarray
        Dielectric constant matrix.
    stopping_criterion : float, optional
        Convergence threshold (default from args.conv).

    Returns
    -------
    field : numpy.ndarray
        Potential distribution after convergence.
    """
    
    n_iter = np.int32(0)
    delta = np.float32(500.0)
    
    while delta > stopping_criterion:
        print("Iteration No =", n_iter, "| delta =", delta)
        new_field = fdm.fdm_drift_same_scale_dielec(field, B, D)
        delta = np.max(np.abs(field - new_field))
        field, new_field = new_field, field
        n_iter += 1
    
    return field


# -------------------------------
# Main execution
# -------------------------------
print("Welcome to the drift field calculation\n")
print("#" * 40)
print("\nSettings used:")
print("Voltage bias setup: shield = %.2f V ; view0 = %.2f V ; view1 = %.2f V ; view2 = %.2f V"
      % (params['V_shield'], params['V_ind1'], params['V_ind2'], params['V_coll']))
print("Step: %.2f mm" % params['step'])
print("\n" + "#" * 40 + "\n")

print("Ly:", params['Ly'])
print("Lz:", params['Lz'])
print("Holes shifted setup:", params['hole_shifted'])
print("Convergence criterion:", args.conv)

# Initialize potential
drift_potential, boundary, diel = init_cond()

# -------------------------------
# Field calculation
# -------------------------------
tstart = time.time()
drift_potential = FDMiteration(drift_potential, boundary, diel)
tstop = time.time()
print("Field calculation took %.2f s for convergence criterion: %s V" % ((tstop - tstart), args.conv))

print("Field shape:", drift_potential.shape)

# Reshape field to match geometry
drift_potential = geo.shape(drift_potential, params['n_strip'], axis="1")
drift_potential = geo.shape(drift_potential, params['n_strip'], axis="2")

print("Final field shape:", drift_potential.shape)

# Compute electric field components
E = field.field_component(drift_potential, "drift", params['hx'], params['hyz'])

# Save results
save_field(drift_potential, E[0], E[1], E[2], params['drift_path'], "drift", args.namefile)
