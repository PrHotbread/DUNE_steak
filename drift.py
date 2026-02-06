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
import time as time

import build_field

from field_calc import finite_difference as fdm
from field_calc import elecfield as field
from field_calc import setup_fdm

from settings import parameters
from settings import geometry as geo
from settings import args
from settings import filesIO

from physics import materials

# -------------------------------
# Geometry and parameters
# -------------------------------
args = args.drift() # Argument parsing
params = parameters.get_parameters(args.setup)
g = params['geometry']
p = params['physics']


def main():
    pcb = geo.PCB(g['ny'] + 1, g['nz'] + 1, g['step'], g['r_hole'])

    plane = pcb.hex_pattern(Ly = g['Ly'], Lz = g['Lz'], shift =  0.0)
    plane_shifted = pcb.hex_pattern(Ly = g['Ly'], Lz = g['Lz'], shift = g['shift'])

    pcb.plot(plane, "PCB – Base Pattern")

    #builder = build_field.PotentialField(plane, plane_shifted, g['nx'], g['ny'] + 1, g['nz'] + 1, 
                                #g['idx_shield'], g['idx_ind1'], g['idx_ind2'], g['idx_coll'], g['idx_drift'], g['idx_ground'],
                                #p['e_Ar'], p['e_fr4'], p['e_Cu'])
                                
    builder = build_field.PotentialField(plane, plane_shifted, g['nx'], g['ny'] + 1, g['nz'] + 1, 
                            g['idx_shield'], g['idx_ind1'], g['idx_ind2'], g['idx_coll'], g['idx_drift'], g['idx_ground'],
                            materials.LiquidArgon.dielectric_permitivity, materials.FR4.dielectric_permitivity, materials.Copper.dielectric_permitivity)
    
    drift_potential, diel, boundary = builder.set_drift(p['V_drift'], p['V_shield'], p['V_ind1'], p['V_ind2'], p['V_coll'], p['V_ground'])
    #drift_potential, diel, boundary = builder.set_drift(0, 0, 0, 0, 0, 0)
    #shaper = geo.FieldShaper(g['n_strip'])
    #diel= shaper.extend_volume(diel, axis = 1)
    #pcb.plot(diel[g['idx_ind1']], "PCB – Base Pattern")
    # -------------------------------
    # Field calculation
    # -------------------------------

    cfg_fdm = setup_fdm.FDMConfig(
        fdm_func=fdm.fdm_drift_dielec,
        stopping_criterion=np.float32(args.conv)
    )
    drift_potential = setup_fdm.solve_fdm(drift_potential, boundary, diel, cfg_fdm)

    # Reshape field to match geometry
    shaper = geo.FieldShaper(g['n_strip'])
    drift_potential = shaper.extend_volume(drift_potential, axis = 1)
    drift_potential = shaper.extend_volume(drift_potential, axis = 2)

    # Compute electric field components
    setup_electric_field = build_field.ElectricField(g['hx'], g['hyz'])
    E = setup_electric_field.compute_field(drift_potential, "weighting")

    # Save results
    f_io = filesIO.FieldIO(params['paths']['base'])
    f_io.save_field(drift_potential, E[0], E[1], E[2], namefile = args.namefile, type_field="drift")
    return None

# -------------------------------
# Main execution
# -------------------------------

print("Welcome to the drift field calculation\n")
print("#" * 40)
print("\nSettings used:")
print("Voltage bias setup: shield = %.2f V ; view0 = %.2f V ; view1 = %.2f V ; view2 = %.2f V"
      % (p['V_shield'], p['V_ind1'], p['V_ind2'], p['V_coll']))
print("Step: %.2f mm" % g['step'])
print("\n" + "#" * 40 + "\n")

print("Ly:", g['Ly'])
print("Lz:", g['Lz'])
print("Holes shifted setup:", g['shift'])
print("Convergence criterion:", args.conv)

tstart = time.time()
main()
tstop = time.time()
print("Field calculation took %.2f s for convergence criterion: %s V" % ((tstop - tstart), args.conv)) 