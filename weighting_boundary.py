#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from settings import args
import time as time

import build_field

from field_calc import finite_difference as fdm
from field_calc import elecfield as field
from field_calc import setup_fdm

from settings import parameters
from settings import geometry as geo
from settings import filesIO

from physics import materials

print("\nField Calculation by iterative method !\n")


# -------------------------------
# Geometry and parameters
# -------------------------------
args = args.weighting() # Argument parsing
params = parameters.get_parameters(args.setup)

g = params['geometry']
w = params['weighting']
p = params['physics']

def extract_boundary_planes(field):
    return field[:, 0, :].copy(), field[:, -1, :].copy()

print("Welcome to the weighting field calculation")
print("\n")
print(40*"#")
print("\n")
print("Settings used:")
print("View: %s " %(args.view))
print("Step: %0.2f mm" %g['step'])
print("\n")
print(40*"#")
print("\n")

""" Initialization of the initial conditions --> set strip, view, boundary and dielectric """
#view, view2D = set_view(args.view)

pcb = geo.PCB(g['ny'] + 1, g['nz'] + 1, g['step'], g['r_hole'])

plane = pcb.hex_pattern(Ly = g['Ly'], Lz = g['Lz'], shift = 0)
plane_shifted = pcb.hex_pattern(Ly = g['Ly'], Lz = g['Lz'], shift = g['shift'])

builder_2D = build_field.PotentialField(plane, plane_shifted, w['nx_2D'], g['ny'] + 1, g['nz'] + 1, 
                               w['idx_shield_2D'], w['idx_ind1_2D'], w['idx_ind2_2D'], w['idx_coll_2D'], g['idx_drift'], g['idx_ground'],
                               materials.LiquidArgon.dielectric_permitivity, materials.FR4.dielectric_permitivity, materials.Copper.dielectric_permitivity)

weighting_potential_2D, diel_2D, boundary_2D = builder_2D.set_weighting_2D(w['n_strip_2D'], g['step'], g['shift'], args.view)


print("Weighting_2D shape = ", weighting_potential_2D.shape)
print("Diel_2D shape = ", diel_2D.shape)
print("Boundary_2D shape = ", boundary_2D.shape)


cfg_2D = setup_fdm.FDMConfig(
    fdm_func=fdm.fdm_weighting_2D,
    stopping_criterion=1e-10
)


""" Field calculation """
tstart=time.time()
weighting_potential_2D = setup_fdm.solve_fdm(weighting_potential_2D, boundary_2D, diel_2D, cfg_2D)
tstop=time.time()

print("Field calculation took %.2f s to run for a stopping convergence criterion -: %s V"%((tstop-tstart), args.conv))



builder_3D = build_field.PotentialField(plane, plane_shifted, g['nx'], g['ny'] + 1, g['nz'] + 1, 
                               g['idx_shield'], g['idx_ind1'], g['idx_ind2'], g['idx_coll'], g['idx_drift'], g['idx_ground'],
                               materials.LiquidArgon.dielectric_permitivity, materials.FR4.dielectric_permitivity, materials.Copper.dielectric_permitivity)

weighting_potential_3D, diel_3D, boundary_3D = builder_3D.set_weighting_3D(g['n_strip'], g['step'], g['shift'], args.view, weighting_potential_2D)

b_min, b_max = builder_3D.extract_boundary_planes(weighting_potential_3D)


cfg_3D = setup_fdm.FDMConfig(
    fdm_func=fdm.fdm_weighting_boundary,
    stopping_criterion=float(args.conv),
    b_min=b_min,
    b_max=b_max
)

#weighting_potential[:,-1,:] = y1

""" Field calculation """
tstart=time.time()
print("Field shape", weighting_potential_3D.shape)

weighting_potential_3D = setup_fdm.solve_fdm(weighting_potential_3D, boundary_3D, diel_3D, cfg_3D)
#weighting_potential = FDMiteration(weighting_potential, boundary, diel, fdm.fdm_weighting_boundary, stopping_criterion = np.float32(args.conv), b_min=b_min, b_max=b_max)
tstop=time.time()

print("Field calculation took %.2f s to run for a stopping convergence criterion -: %s V"%((tstop-tstart), args.conv))
#plot.plot_pattern(weighting_potential[params["idx_coll"], :, :], params)

shaper = geo.FieldShaper(g['n_strip'])
weighting_potential_3D = shaper.extend_volume(weighting_potential_3D, axis = 2)

setup_electric_field = build_field.ElectricField(g['hx'], g['hyz'])
E = setup_electric_field.compute_field(weighting_potential_3D, "weighting")

f_io = filesIO.FieldIO(params['paths']['base'])
f_io.save_field(weighting_potential_3D, E[0], E[1], E[2], namefile = args.namefile, type_field="weighting", view=args.view)

#save_field(weighting_potential, Ex, Ey, Ez, g['path'], "weighting", args.namefile, args.view)

#plot.plot_weighting(weighting_potential, args.namefile, view)
 