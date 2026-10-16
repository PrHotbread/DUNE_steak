# settings/parameters.py
import numpy as np
import json
#import libRooFitJSONInterface

# Typing
f32 = np.float32
i32 = np.int32

def to_nearest(length, step):
   return np.round(np.round(length / step) * step,2)
   
def index(length, step):
    return i32(np.round(length/step))

# ----------------------------
# Default parameters
# ----------------------------
def default_parameters():
    return {
        "geometry": {
            "hole_radius": f32(1.2), 
            "hexagonal_side": f32(2.94),
            "shift": f32(0.5),
            "drift_length": f32(5.0),
            "pcb1_thickness": f32(3.2),
            "pcb2_thickness": f32(3.2),
            "gap_length": f32(10.0),
            "ground_length": f32(10.0),
            "step": f32(5e-2),
            "delta_x": f32(1.0),
            "delta_yz": f32(1.0),
            "n_strip": i32(7)
        },

        "physics": {
            "E_drift": f32(50.0), #V/mm
            "T_Ar": f32(87.0), #Liquide Argon Temperature
            "V_shield": f32(-1500.0), #Potential in V
            "V_ind1": f32(-500.0), #Potential in V
            "V_ind2": f32(0.0), #Potential in V
            "V_coll": f32(1000.0),#Potential in V
            "V_ground": f32(0.0),#Potential in V
        },

        "simulation": {
            "te": f32(0.001),
            "acquisition_time": f32(15.0),
            "conv_crit": f32(1e-1),
            "ne": i32(100),
        },

        "weighting" : {
        },

        "paths": {
            "base": "/Users/pinchault/Documents/work/research/DUNE_steak/steak/results_field/test_newconf",
            "drift": "/Users/pinchault/Documents/work/research/DUNE_steak/steak/results_field/nominal",
            "results": "/Users/pinchault/Documents/work/research/DUNE_steak/steak/results_param_drift/",
            "electronics":""
        }
    }

# ----------------------------
#  Secondary parameters
# ----------------------------
def compute_derived(geom):
    """Ajoute les grandeurs calculées automatiquement"""
    g = geom["geometry"]
    p = geom["physics"]
    w = geom['weighting']
    s = geom['simulation']

    # Lengths and indexes
    g["hx"] = g["step"] * g["delta_x"]
    g["hyz"] = g["step"] * g["delta_yz"]

    g["x_drift"]  = f32(0.0)
    g["x_shield"] = g["x_drift"] + g["drift_length"]
    g["x_ind1"]   = g["x_shield"] + g["pcb1_thickness"]
    g["x_ind2"]   = g["x_ind1"] + g["gap_length"]
    g["x_coll"]   = g["x_ind2"] + g["pcb2_thickness"]
    g["x_ground"] = g["x_coll"] + g["ground_length"]

    g['Lx'] = g['x_ground'] - g['x_drift']
    g['Ly'] = f32(to_nearest(g['hexagonal_side'] * np.cos(np.radians(30)), g['step']))
    g['Lz'] = f32(to_nearest(g['hexagonal_side'] * np.sin(np.radians(30)), g['step']))

    g['nx'] = i32(index(g['Lx'], g['step'] * g['delta_x']))
    g['ny'] = i32(index(g['Ly'], g['step'] * g['delta_yz']))
    g['nz'] = i32(index(g['Lz'], g['step'] * g['delta_yz']))
    g['idx_drift'] = i32(index(g['x_drift'], g['step'] * g['delta_x']))
    g['idx_shield'] = i32(index(g['x_shield'], g['step'] * g['delta_x']))
    g['idx_ind1'] = i32(index(g['x_ind1'], g['step'] * g['delta_x']))
    g['idx_ind2'] = i32(index(g['x_ind2'], g['step'] * g['delta_x']))
    g['idx_coll'] = i32(index(g['x_coll'], g['step'] * g['delta_x']))
    g['idx_ground'] = i32(index(g['x_ground'], g['step'] * g['delta_x']))

    # Drift potential
    p["V_drift"] = f32(p["V_shield"] - p["E_drift"] * (g["x_shield"] - g["x_drift"]))

    # Weighting parameters
    w['width_induct'] = f32(3 * g['Ly'])
    w['width_collect'] = f32(2 * g['Ly'])
    w['idx_ind_strip'] = i32(index(w['width_induct'], g['step']))
    w['idx_coll_strip'] = i32(index(w['width_collect'], g['step']))

    # 2D weighting field calculation
    w['drift_length_2D'] = f32(5.)
    w['x_drift_2D'] = f32(0.)
    w['x_shield_2D'] = f32(w['x_drift_2D']  + w['drift_length_2D'])
    w['x_ind1_2D']   = f32(w['x_shield_2D'] + g['pcb1_thickness'])
    w['x_ind2_2D']   = f32(w['x_ind1_2D'] + g['gap_length'])
    w['x_coll_2D']   = f32(w['x_ind2_2D'] + g['pcb2_thickness'])
    w['x_ground_2D'] = f32(w['x_coll_2D'] + g['ground_length'])

    w['Lx_2D'] = w['x_ground_2D'] - w['x_drift_2D']

    w['nx_2D'] = i32(index(w['Lx_2D'], g['step'] * g['delta_x']))

    w['idx_drift_2D'] = i32(index(w['x_drift_2D'], g['step'] * g['delta_x']))
    w['idx_shield_2D'] = i32(index(w['x_shield_2D'], g['step'] * g['delta_x']))
    w['idx_ind1_2D'] = i32(index(w['x_ind1_2D'], g['step'] * g['delta_x']))
    w['idx_ind2_2D'] = i32(index(w['x_ind2_2D'], g['step'] * g['delta_x']))
    w['idx_coll_2D'] = i32(index(w['x_coll_2D'], g['step'] * g['delta_x']))
    w['idx_ground_2D'] = i32(index(w['x_ground_2D'], g['step'] * g['delta_x']))
    w['n_strip_2D'] = i32(g['n_strip'] + 6) 

    # Electron drift parameters
    s['n_time_step'] = np.int32(np.round(s['acquisition_time'])/s['te'])
    s['drift_time'] = np.linspace(0, s['acquisition_time'], s['n_time_step'], dtype = np.float32 )
    return geom


# ----------------------------
#  Update parameters json file
# ----------------------------
def set_param(key, val, ref):
    """Met à jour récursivement un dictionnaire de paramètres."""
    if key in ref:
        if isinstance(val, dict):
            for k, v in val.items():
                set_param(k, v, ref[key])
        else:
            ref[key] = val
    else:
        print(f"Unknown parameter '{key}'")

def update_from_file(geom, json_path):
    """Charge un fichier JSON et met à jour les valeurs par défaut"""
    with open(json_path, 'r') as f:
        user_params = json.load(f)
    for k, v in user_params.items():
        set_param(k, v, geom)
    return geom

def get_parameters(json_file=None):
    """ Return a complet dictionnary for the simulation """
    geom = default_parameters()
    if json_file:
        geom = update_from_file(geom, json_file)
    geom = compute_derived(geom)
    return geom