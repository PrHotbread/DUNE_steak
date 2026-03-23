import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import argparse
from plot_geo import plot_PCB

import plotting as plot

from physics import transportation
from physics import drift_charge
from physics import shockley_ramo
import time
import pandas as pd
from scipy import interpolate
from scipy import integrate


from settings import parameters
from settings import args
from settings import filesIO
from settings import geometry as geo

from physics.dielec import dielectric_tensor

from physics import materials

from settings import monte_carlo
""" ###################################################################################################

This module allows calculating the induced current on each view by using the Shockley-Ramo theorem.

 ################################################################################################### """


args = args.gen_signal() # Argument parsing

params = parameters.get_parameters('')
g = params['geometry']
p = params['physics']
w = params['weighting']
s = params['simulation']

        
        

######################## save part - csv/signal/traj #############################

def save_csv(namefile:str):
    df = pd.DataFrame(columns=["xi", "yi", "zi", "xf", "yf", "zf", "drift time anode (t_coll - t_shield)", "q induction1", "q induction2", "q collection", "collect v0", "collect v1", "collect v2", "collect shield", "fr4 capture"], index = np.linspace(1, ne, ne))
    for i in range(ne):
        df.loc[i+1, "xi"] = t[i,0,0] - e_drift[3].shape[2]/2*5e-2
        df.loc[i+1, "yi"] = t[i,1,0] - e_drift[3].shape[1]/2*5e-2
        df.loc[i+1, "zi"] = t[i,2,0]
        df.loc[i+1, "xf"] = t[i, 0][t[i, 0] != 0][-1] 
        df.loc[i+1, "yf"] = t[i, 1][t[i, 1] != 0][-1] - e_drift[3].shape[1]/2*5e-2
        df.loc[i+1, "zf"] = t[i, 2][t[i, 2] != 0][-1] - e_drift[3].shape[2]/2*5e-2
        df.loc[i+1, "drift time anode"] = t_drift[i]
        df.loc[i+1, "q induction1"] = q_view0[i]
        df.loc[i+1, "q induction2"] = q_view1[i]
        df.loc[i+1, "q collection"] = q_view2[i]
        df.loc[i+1, "collect v0"] = c[i,0]
        df.loc[i+1, "collect v1"] = c[i,1]
        df.loc[i+1, "collect v2"] = c[i,2]
        df.loc[i+1, "collect shield"] = c[i,3]
        df.loc[i+1, "fr4 capture"] = c[i,4]
    gen.check_path(params['paths']['results'])
    df.to_csv(params['paths']['results'] + "data_charge_tracking_" + namefile + ".csv")#, index = False)
    print('DataFrame is written to CSV File successfully.')
    return None
    
def save(namefile:str):
    np.save(params['paths']['results'] + "physics_params_" + namefile + ".npy", (field_mean, mu_mean, te_mean, le_mean, lc_mean, tc_mean,  s['drift_time']))
    np.save(params['paths']['results'] + "results_raw_signal_" + namefile + ".npy", (S_view0, S_view1, S_view2, s['drift_time']))
    np.save(params['paths']['results'] + "results_convolve_signal_" + namefile + ".npy", (Sconv_view0, Sconv_view1, Sconv_view2, t_conv))
    np.save(params['paths']['results'] + "results_DAQ_" + namefile + ".npy", (dQ_view0, dQ_view1, dQ_view2, sample))
    print('Numpy arrays containing induced signal are written to .npy File successfully.')
    return None

def save_traj(namefile:str):
    np.save(params['paths']['results'] + "charge_trajectories_" + namefile + ".npy", (t))
    return None
    
    
def extended_strip(s_view, ne, view):
    Sk = 0 
    Sk1 = 0
    Sk2 = 0
    nk = 0
    nk1 = 0
    nk2 = 0 
    for i in range(ne):
        yf = t[i, 1][t[i, 1] != 0][-1] - e_drift[3].shape[1]/2 * 5e-2
        zf = t[i, 2][t[i, 2] != 0][-1] - e_drift[3].shape[1]/2 * 5e-2 
        if view == "view2":
            in_central = (-w['L_coll_strip']/2 < yf < w['L_coll_strip']/2)
            in_extended = (-3/2 * w['L_coll_strip'] < yf <  3/2 * w['L_coll_strip']) and not in_central
            in_extextended = (- 5 /2 * w['L_coll_strip'] < yf < 5/2 * w['L_coll_strip']) and not (in_extended and in_central)
        else:
            in_central = (-w['L_coll_strip']/2 < yf < w['L_coll_strip'])
            in_extended = (-2 * w['L_coll_strip'] < yf <  5/2 * w['L_coll_strip']) and not in_central
            in_extextended = (- 4 * w['L_coll_strip'] < yf < 7/2 * w['L_coll_strip']) and not (in_extended and in_central)
        if in_central:
            if isinstance(Sk, int):
                Sk = s_view[i].copy()
            else:
                Sk += s_view[i]
                nk += 1
        elif in_extended:
            if isinstance(Sk1, int):
                Sk1 = s_view[i].copy() 
            else:
                Sk1 += s_view[i]
                nk1 += 1
        elif in_extextended:
            if isinstance(Sk2, int):
                Sk2 = s_view[i].copy() 
            else:
                Sk2 += s_view[i]
                nk2 += 1
    return Sk/nk, Sk1/nk1, Sk2/nk2

def save_extend(namefile:str):
    Skv0 = extended_strip(s_view0, ne, "view0")
    Skv1 = extended_strip(s_view1, ne, "view1")
    Skv2 = extended_strip(s_view2, ne, "view2")
    np.save(params['paths']['results'] + "signal_extended_v0_" + namefile + ".npy", (Skv0))
    np.save(params['paths']['results'] + "signal_extended_v1_" + namefile + ".npy", (Skv1))
    np.save(params['paths']['results'] + "signal_extended_v2_" + namefile + ".npy", (Skv2))
    return None





cfg_io = filesIO.FieldIO(params['paths']['base'])

""" File fields  """
e_drift = cfg_io.load_field("drift_Ex", "drift_Ey", "drift_Ez", "drift")

ew_view0 = cfg_io.load_field("weighting_Ex", "weighting_Ey", "weighting_Ez", "weighting", "view0")
ew_view1 = cfg_io.load_field("weighting_Ex", "weighting_Ey", "weighting_Ez", "weighting", "view1")
ew_view2 = cfg_io.load_field("weighting_Ex", "weighting_Ey", "weighting_Ez", "weighting", "view2")

print("field shape", e_drift[0].shape)

""" Set anode geometry"""
pcb = geo.PCB(g['ny'] + 1, g['nz'] + 1, g['step'], g['r_hole'])
plane = pcb.hex_pattern(Ly = g['Ly'], Lz = g['Lz'], shift =  0.0)
plane_shifted = pcb.hex_pattern(Ly = g['Ly'], Lz = g['Lz'], shift = g['shift'])

""" Set Dielectric matrix """
dm = dielectric_tensor(g["idx_shield"], g["idx_ind1"], g["idx_ind2"], g["idx_coll"],
                       g["nx"], g["ny"] + 1, g["nz"] + 1,
                       materials.LiquidArgon.dielectric_permitivity, materials.FR4.dielectric_permitivity, materials.Copper.dielectric_permitivity, 
                       plane, plane_shifted)


shaper = geo.FieldShaper(g['n_strip'])
dm = shaper.extend_volume(dm, axis = 1)
dm = shaper.extend_volume(dm, axis = 2)

print("dielec", dm.shape)

volume = pcb.volume(e_drift[3], g['hx'], g['hyz'])
print("volume :", volume.shape)
ne = np.int32(args.n)


cfg_mc = monte_carlo.MonteCarlo(ne = np.int32(args.n), volume = volume, 
                                step_x = g['hx'], step_yz = g['hyz'])

vector_position = cfg_mc.set_position(x = -4, y = 0, z = 0)
#vector_position = cfg_mc.set_hexagonal(ly = np.float32(g['Ly']), lz = np.float32(g['Ly']))

""" start simulation """
t1 = time.time()
t, v, field, mu, te, tc, lc, c = drift_charge.elec_gun(vector_position, ne, s['n_time_step'], e_drift, volume, dm, s['te'], 
                                                        g['x_shield'], g['x_ind1'], g['x_ind2'], g['x_coll'], 
                                                        g['hx'], g['hyz'], 
                                                        materials.LiquidArgon.temperature)


plot.plot_traj3D(t, ne)


s_view0, s_view1, s_view2 = shockley_ramo.induced_signal(ne, s['n_time_step'], volume, 
                                                             np.float32(t[:,0]), np.float32(t[:,1]), np.float32(t[:,2]), 
                                                             np.float32(v[:,0]), np.float32(v[:,1]), np.float32(v[:,2]), 
                                                             ew_view0,  ew_view1, ew_view2, g['hx'])


params = shockley_ramo.SimulationParams(ne = ne, td = s['drift_time'], 
                                        x =np.float32(t[:,0]), y = np.float32(t[:,1]), z = np.float32(t[:,2]))


"""Charge calculation """
q_view0 = params.charge(s_view0)
q_view1 = params.charge(s_view1)
q_view2 = params.charge(s_view2)

""" Drift time in anode """

drift_time = params.drift_time_anode(xs = g['x_shield'])

"""Start and End Point"""
xi, yi, zi = params.start_point()
xf, yf, zf = params.start_point()

""" Mean signal from simulation """
S_view0 = np.sum(s_view0, axis = 0)
S_view1 = np.sum(s_view1, axis = 0)
S_view2 = np.sum(s_view2, axis = 0)

cfg_processing = shockley_ramo.SignalProcessing(te = s['te'], ta = s['acquisition_time'], td = s['drift_time'], elec = 'bot')

tc, sc_view0 = cfg_processing.convolved(S_view0)
tc, sc_view1 = cfg_processing.convolved(S_view1)
tc, sc_view2 = cfg_processing.convolved(S_view2)


sample_time, sv0_output =  cfg_processing.daq(tc, sc_view0)
sample_time, sv1_output =  cfg_processing.daq(tc, sc_view1)
sample_time, sv2_output =  cfg_processing.daq(tc, sc_view2)



#q_view0, q_view1, q_view2 = signal_processing.charge(ne, s_view0, s_view1, s_view2)
te = s['drift_time']
print(te.shape)
print(s_view0.shape[0])
plt.plot(s['drift_time'], s_view0[0])
plt.show()
plt.plot(tc, sc_view0)
plt.show()


t2 = time.time()
print("Simulation  took: %f s" %(t2-t1))
""" end simulation """





plt.plot(s['drift_time'], S_view0)
plt.show()


Qtot = integrate.trapz(S_view2, s['drift_time'])/np.sum(c[:,2], axis = 0)
plot.plot_electron([S_view0, S_view1, S_view2], s['drift_time'], Qtot, tc, sc_view0, sc_view1, sc_view2, sample_time, sv0_output, sv1_output, sv2_output)
print("Charge total = ", Qtot)

field_mean = np.mean(field[c[:,2] == 1], axis = 0)
mu_mean = np.mean(mu[c[:,2] == 1], axis = 0)
te_mean = np.mean(te[c[:,2] == 1], axis = 0)
tc_mean = np.mean(tc[c[:,2] == 1], axis = 0)
lc_mean = np.mean(lc[c[:,2] == 1], axis = 0)
le_mean = lc_mean/tc_mean * te_mean
print(c[:,0])
print(c[:,1])
print(c[:,2])
print(c[:,3])
print(c[:,4])
T = np.sum(c[:,2]) / ( np.sum(c[:,0]) + np.sum(c[:,1]) + np.sum(c[:,2]) + np.sum(c[:,3]) + np.sum(c[:,4]))
print("Transparency", T)
#traj3D(t)


""" Save part """
namefile_results = "li"
save_csv(namefile_results)
save(namefile_results)
save_traj(namefile_results)
#save_extend("nominal")