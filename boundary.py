import numpy as np
from settings import geometry as geo

def set_drift_boundary(f, p, p_shifted, idx_s, idx_i1, idx_i2, idx_c):
    # Define boundary condition mask
    boundary = np.ones(f.shape, dtype=np.float32)
    boundary[:,:,0] = 2
    boundary[:,:,-1] = 3
    boundary[:,0,:] = 4
    boundary[:,-1,:] = 5
    boundary[:,0,0] = 6
    boundary[:,0,-1] = 7
    boundary[:,-1,0] = 8
    boundary[:,-1,-1] = 9
    
    # Apply boundary mask to electrodes
    boundary[idx_s] *= p
    boundary[idx_i1] *= p_shifted
    boundary[idx_i2] *= p
    boundary[idx_c] *= p
    return boundary


def set_weighting_boundary(p, p_shifted, idx_s, idx_i1, idx_i2, idx_c, nx, ny, nz, shaper):
    # Define boundary condition mask
    
    boundary = np.ones((  nx,
                    ny,
                    nz), dtype = np.float32)

    # Apply boundary mask to electrodes
    boundary[idx_s] *= p
    boundary[idx_i1] *= p_shifted
    boundary[idx_i2] *= p
    boundary[idx_c] *= p
    boundary = shaper.extend_volume(boundary, axis=1)
    #boundary = geo.shape(boundary, n_strip, axis = "1")
    boundary[:,:,0][boundary[:,:,0] == 1] = 2
    boundary[:,:,-1][boundary[:,:,-1] == 1] = 3
    boundary[:,0,:][boundary[:,0,:] == 1] = 4
    boundary[:,-1,:][boundary[:,-1,:] == 1] = 5
    boundary[:,0,0] = 6
    boundary[:,0,-1] = 7
    boundary[:,-1,0] = 8
    boundary[:,-1,-1] = 9
    return boundary

