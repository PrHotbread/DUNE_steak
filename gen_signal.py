import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import argparse
from plot_geo import plot_PCB

import plotting as plot

from physics import physics_parameters
from physics import drift_charge
from physics import signal_processing
import time
import pandas as pd
from scipy import interpolate
from scipy import integrate


import numpy as np
import pandas as pd
import numba as nb
import matplotlib.pyplot as plt


from settings import parameters
from settings import args
from settings import filesIO
from settings import geometry as geo

from physics.dielec import dielectric_tensor

from physics import materials
""" ###################################################################################################

This module allows calculating the induced current on each view by using the Shockley-Ramo theorem.

 ################################################################################################### """


args = args.gen_signal() # Argument parsing

params = parameters.get_parameters('')
g = params['geometry']
p = params['physics']
w = params['weighting']
s = params['simulation']
#@nb.jit(nopython = True, parallel = True)
def induced_signal(ne, n_step, vol, x, y, z, vx, vy, vz, ew_view0, ew_view1, ew_view2, step_x = np.float32(g['hx'])):
    Sind1 = np.zeros((ne, n_step), dtype = np.float32)
    Sind2 = np.zeros((ne, n_step), dtype = np.float32)
    S_view2 = np.zeros((ne, n_step), dtype = np.float32)
    print(Sind1.shape)
    print(x.shape)
    print(vx.shape)
    for e in nb.prange(ne):
        print(e)
        Sind1[e], Sind2[e], S_view2[e] = signal_processing.ramo(vol, x[e], y[e], z[e], vx[e], vy[e], vz[e], ew_view0, ew_view1, ew_view2, step_x, n_step)
    return Sind1, Sind2, S_view2
     
     
def charge(ne, ind1, ind2, coll):
    q_view0 = np.zeros(ne, dtype = np.float32)
    q_view1 = np.zeros(ne, dtype = np.float32)
    q_view2 = np.zeros(ne, dtype = np.float32)
    for e in range(ne):
        q_view0[e] = integrate.trapz(ind1[e], g['drift_time'])
        q_view1[e] = integrate.trapz(ind2[e], g['drift_time'])
        q_view2[e] = integrate.trapz(coll[e], g['drift_time'])
    return q_view0, q_view1, q_view2
        
        
def drift_time_anode(ne, drift_time, x):
    t_drift = np.zeros(ne, dtype = np.float32)
    for e in range(ne):
        x_drift = x[e][x[e] != 0]
        coll = drift_time[len(x_drift) -1]
        idx_shield = np.where(x[e]>g['x_shield'] - 0.55)
        shield = drift_time[idx_shield[0][0]]
        t_drift[e] = coll - shield
    return t_drift
    
    
def gen_weighting(path = params['paths']['base'], n_strip = np.int32(g['n_strip'])):
    ew_view0 = cfg_io.load_field("weighting_Ex", "weighting_Ey", "weighting_Ez", "weighting", "view0")
    ew_view1 = cfg_io.load_field("weighting_Ex", "weighting_Ey", "weighting_Ez", "weighting", "view1")
    ew_view2 = cfg_io.load_field("weighting_Ex", "weighting_Ey", "weighting_Ez", "weighting", "view2")
    return ew_view0, ew_view1, ew_view2
   
def set_position(position_vector):
    z0 = e_drift[3].shape[2]/2 * np.float32(g['hyz'])
    y0 = e_drift[3].shape[1]/2 * np.float32(g['hyz'])
    x0 = 1
    position_vector[:, 2] = z0 + 1 * np.float32(g['Lz']) * (2 * np.random.random(ne) - 1)
    position_vector[:, 1] = y0 + 2 * np.float32(g['Ly']) * (2 * np.random.random(ne) - 1)
    position_vector[:, 0] = x0
    theta = np.random.random( size = ne) * 2 * np.pi
    R =  np.random.random( size = ne ) * 1.7
    """     position_vector[:, 2] = R * np.cos(theta) + z0
    position_vector[:, 1] = R * np.sin(theta) + y0
    position_vector[:, 0] = np.random.random(size = ne) * 2 """
    return position_vector


def traj3D(t):
    fig, ax = plt.subplots()
    ax = plt.axes(projection='3d')
    for e in range(ne):
        ax.plot(t[e, 2][t[e, 2] != 0], t[e, 1][t[e, 1] != 0], t[e, 0][t[e, 0] != 0], linewidth = 0.5)
    ax.set_aspect('equal')
    plt.show()
    return None

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

print("field shape", e_drift[0].shape)
ew_view0, ew_view1, ew_view2 = gen_weighting()

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

ne = np.int32(args.n)
init_position = np.zeros((ne, 3), dtype  = np.float32)

init_position = set_position(init_position)


""" start simulation """
t1 = time.time()
t, v, field, mu, te, tc, lc, c = drift_charge.elec_gun(init_position, ne, s['n_time_step'], e_drift, volume, dm, s['te'], 
                                                        g['x_shield'], g['x_ind1'], g['x_ind2'], g['x_coll'], 
                                                        g['hx'], g['hyz'], 
                                                        materials.LiquidArgon.temperature)
traj3D(t)

s_view0, s_view1, s_view2 = induced_signal(ne, s['n_time_step'], volume, np.float32(t[:,0]), np.float32(t[:,1]), np.float32(t[:,2]), np.float32(v[:,0]), np.float32(v[:,1]), np.float32(v[:,2]), ew_view0,  ew_view1, ew_view2)

#s_view0, s_view1, s_view2 = signal_processing.ramo(volume, 
                                                   #np.float32(t[:,0]), np.float32(t[:,1]), np.float32(t[:,2]), 
                                                   #np.float32(v[:,0]), np.float32(v[:,1]), np.float32(v[:,2]), 
                                                   #ew_view0, ew_view1, ew_view2,
                                                   #g['step'], s['n_time_step'])

#q_view0, q_view1, q_view2 = signal_processing.charge(ne, s_view0, s_view1, s_view2)

t_drift = drift_time_anode(ne, s['drift_time'], t[:,0])

t2 = time.time()
print("Simulation  took: %f s" %(t2-t1))
""" end simulation """



""" Mean signal from simulation """
S_view0 = np.sum(s_view0, axis =0)
S_view1 = np.sum(s_view1, axis =0)
S_view2 = np.sum(s_view2, axis =0)

""" Convolved signals """

dQ_view0, sample, Sconv_view0, t_conv = signal_processing.conv(S_view0, "bot")
dQ_view1, sample, Sconv_view1, t_conv = signal_processing.conv(S_view1, "bot")
dQ_view2, sample, Sconv_view2, t_conv = signal_processing.conv(S_view2, "bot")


print("Charge integration = ", np.trapz(dQ_view2, sample))


Qtot = integrate.trapz(S_view2, s['drift_time'])/np.sum(c[:,2], axis = 0)
plot.plot_electron([S_view0, S_view1, S_view2], Qtot, t_conv, Sconv_view0, Sconv_view1, Sconv_view2, sample, dQ_view0, dQ_view1, dQ_view2)
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