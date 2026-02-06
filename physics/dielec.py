import numpy as np
#import materials 

def dielectric_tensor(idx_shield, idx_ind1, idx_ind2, idx_coll,
                      nx, ny, nz,
                      e_Ar, e_fr4, e_Cu,
                      plane,
                      plane_shifted=None,
                      hole_shifted=False):

    # Initialize dielectric tensor with liquid argon permittivity
    Matrix = e_Ar * np.ones((nx, ny, nz))

    # Compute midpoints between PCB indices
    idx_half_pcb_1 = round((idx_ind1 + idx_shield) / 2)
    idx_half_pcb_2 = round((idx_coll + idx_ind2) / 2)


    # Assign dielectric regions (FR4)
    Matrix[idx_shield + 1:idx_half_pcb_1, plane == 0] = e_fr4
    Matrix[idx_ind2 + 1:idx_half_pcb_2, plane == 0] = e_fr4
    Matrix[idx_half_pcb_2:idx_coll, plane == 0] = e_fr4

    # Assign copper conductor layers
    Matrix[idx_shield, plane == 0] = e_Cu
    Matrix[idx_ind2, plane == 0] = e_Cu
    Matrix[idx_coll, plane == 0] = e_Cu

    # Select which plane to use depending on the hole_shifted option
    if hole_shifted:
        if plane_shifted is None:
            raise ValueError("plane_shifted must be provided when hole_shifted=True")

        # Assign shifted dielectric and conductor regions
        Matrix[idx_half_pcb_1:idx_ind1][plane_shifted == 0] = e_fr4
        Matrix[idx_ind1][plane_shifted == 0] = e_Cu

    else:
        # Assign nominal dielectric and conductor regions
        Matrix[idx_half_pcb_1:idx_ind1, plane == 0] = e_fr4
        Matrix[idx_ind1, plane ==0] = e_Cu

    return Matrix