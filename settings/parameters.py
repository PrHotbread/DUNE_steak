# settings/parameters.py
import numpy as np
import json

# Typage court pour la lisibilité
f32 = np.float32
i32 = np.int32

# ----------------------------
# 1️⃣  Définitions par défaut
# ----------------------------
def default_parameters():
    return {
        "geometry": {
            "r_hole": f32(1.2),
            "d_hole": f32(2.94),
            "shift": f32(0.0),
            "L_drift": f32(5.0),
            "L_pcb1": f32(3.2),
            "L_pcb2": f32(3.2),
            "L_gap": f32(10.0),
            "L_ground": f32(10.0),
            "step": f32(5e-2),
            "delta_x": f32(1.0),
            "delta_yz": f32(1.0),
            "n_strip": i32(7)
        },

        "physics": {
            "E_drift": f32(50.0),
            "e_Ar": f32(1.62),
            "e_fr4": f32(4.0),
            "e_Cu": f32(9999.0),
            "T_Ar": f32(87.0),
            "V_shield": f32(-1500.0),
            "V_ind1": f32(-500.0),
            "V_ind2": f32(0.0),
            "V_coll": f32(1000.0),
            "V_ground": f32(0.0),
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
        }
    }

# ----------------------------
# 2️⃣  Fonction de mise à jour
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
        print(f"⚠️  Unknown parameter '{key}'")

def update_from_file(geom, json_path):
    """Charge un fichier JSON et met à jour les valeurs par défaut"""
    with open(json_path, 'r') as f:
        user_params = json.load(f)
    for k, v in user_params.items():
        set_param(k, v, geom)
    return geom

def to_nearest(length, step):
   return np.round(np.round(length / step) * step,2)
   
def index(length, step):
    return i32(np.round(length/step))
# ----------------------------
# 3️⃣  Calculs géométriques dérivés
# ----------------------------
def compute_derived(geom):
    """Ajoute les grandeurs calculées automatiquement"""
    g = geom["geometry"]
    p = geom["physics"]
    w = geom['weighting']
    # Longueurs et indices
    g["hx"] = g["step"] * g["delta_x"]
    g["hyz"] = g["step"] * g["delta_yz"]

    g["x_drift"]  = f32(0.0)
    g["x_shield"] = g["x_drift"] + g["L_drift"]
    g["x_ind1"]   = g["x_shield"] + g["L_pcb1"]
    g["x_ind2"]   = g["x_ind1"] + g["L_gap"]
    g["x_coll"]   = g["x_ind2"] + g["L_pcb2"]
    g["x_ground"] = g["x_coll"] + g["L_ground"]

    g['Lx'] = g['x_ground'] - g['x_drift']
    g['Ly'] = f32(to_nearest(g['d_hole'] * np.cos(np.radians(30)), g['step']))
    g['Lz'] = f32(to_nearest(g['d_hole'] * np.sin(np.radians(30)), g['step']))

    g['nx'] = i32(index(g['Lx'], g['step'] * g['delta_x']))
    g['ny'] = i32(index(g['Ly'], g['step'] * g['delta_yz']))
    g['nz'] = i32(index(g['Lz'], g['step'] * g['delta_yz']))
    g['idx_drift'] = i32(index(g['x_drift'], g['step'] * g['delta_x']))
    g['idx_shield'] = i32(index(g['x_shield'], g['step'] * g['delta_x']))
    g['idx_ind1'] = i32(index(g['x_ind1'], g['step'] * g['delta_x']))
    g['idx_ind2'] = i32(index(g['x_ind2'], g['step'] * g['delta_x']))
    g['idx_coll'] = i32(index(g['x_coll'], g['step'] * g['delta_x']))
    g['idx_ground'] = i32(index(g['x_ground'], g['step'] * g['delta_x']))

    # Potentiel de drift
    p["V_drift"] = f32(p["V_shield"] - p["E_drift"] * (g["x_shield"] - g["x_drift"]))

    # Weighting parameters
    w['L_ind_strip'] = f32(3 * g['Ly'])
    w['L_coll_strip'] = f32(2 * g['Ly'])
    w['idx_ind_strip'] = i32(index(w['L_ind_strip'], g['step']))
    w['idx_coll_strip'] = i32(index(w['L_coll_strip'], g['step']))

    # 2D weighting field calculation
    w['L_drift_2D'] = f32(5.)
    w['x_drift_2D'] = f32(0.)
    w['x_shield_2D'] = f32(w['x_drift_2D']  + w['L_drift_2D'])
    w['x_ind1_2D']   = f32(w['x_shield_2D'] + g['L_pcb1'])
    w['x_ind2_2D']   = f32(w['x_ind1_2D'] + g['L_gap'])
    w['x_coll_2D']   = f32(w['x_ind2_2D'] + g['L_pcb2'])
    w['x_ground_2D'] = f32(w['x_coll_2D'] + g['L_ground'])

    w['Lx_2D'] = w['x_ground_2D'] - w['x_drift_2D']

    w['nx_2D'] = i32(index(w['Lx_2D'], g['step'] * g['delta_x']))

    w['idx_drift_2D'] = i32(index(w['x_drift_2D'], g['step'] * g['delta_x']))
    w['idx_shield_2D'] = i32(index(w['x_shield_2D'], g['step'] * g['delta_x']))
    w['idx_ind1_2D'] = i32(index(w['x_ind1_2D'], g['step'] * g['delta_x']))
    w['idx_ind2_2D'] = i32(index(w['x_ind2_2D'], g['step'] * g['delta_x']))
    w['idx_coll_2D'] = i32(index(w['x_coll_2D'], g['step'] * g['delta_x']))
    w['idx_ground_2D'] = i32(index(w['x_ground_2D'], g['step'] * g['delta_x']))

    w['n_strip_2D'] = i32(g['n_strip'] + 6) 
    return geom

# ----------------------------
# 4️⃣  Fonction principale
# ----------------------------
def get_parameters(json_file=None):
    """Retourne un dictionnaire complet de paramètres pour la simulation"""
    geom = default_parameters()
    if json_file:
        geom = update_from_file(geom, json_file)
    geom = compute_derived(geom)
    return geom
