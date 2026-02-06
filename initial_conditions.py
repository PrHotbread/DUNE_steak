import numpy as np


class DriftInitialCondition:
    """
    Build initial electric potential for drift field simulation.
    """

    def __init__(self, p, p_shifted,
                 ny, nz,
                 idx_d, idx_s, idx_i1, idx_i2, idx_c, idx_g,
                 u_d, u_s, u_i1, u_i2, u_c, u_g):

        self.plane = p
        self.plane_shifted = p_shifted
        self.ny = ny
        self.nz = nz

        self.idx_d = idx_d
        self.idx_s = idx_s
        self.idx_i1 = idx_i1
        self.idx_i2 = idx_i2
        self.idx_c = idx_c
        self.idx_g = idx_g

        self.u_d = u_d
        self.u_s = u_s
        self.u_i1 = u_i1
        self.u_i2 = u_i2
        self.u_c = u_c
        self.u_g = u_g

    # -------------------------
    # Fixed potentials
    # -------------------------
    @staticmethod
    def electrode_mask(value, pattern):
        mask = np.zeros(pattern.shape)
        mask[pattern == 0] = value
        return mask

    @staticmethod
    def set_dirichlet(field, u_bot, u_top):
        field[0] = u_bot
        field[-1] = u_top
        return field

    # -------------------------
    # Linear Potential Profile
    # -------------------------
    def build_linear_profile(self):
        V = np.concatenate([
            np.linspace(self.u_d, self.u_s,  self.idx_s - self.idx_d),
            np.linspace(self.u_s, self.u_i1, self.idx_i1 - self.idx_s),
            np.linspace(self.u_i1, self.u_i2, self.idx_i2 - self.idx_i1),
            np.linspace(self.u_i2, self.u_c,  self.idx_c - self.idx_i2),
            np.linspace(self.u_c, self.u_g,  self.idx_g - self.idx_c),
        ])
        return V

    def apply_electrodes(self, field):
        field[self.idx_s]  = field[self.idx_s]  * self.plane         + self.electrode_mask(self.u_s,  self.plane)
        field[self.idx_i1] = field[self.idx_i1] * self.plane         + self.electrode_mask(self.u_i1, self.plane)
        field[self.idx_i2] = field[self.idx_i2] * self.plane         + self.electrode_mask(self.u_i2, self.plane)
        field[self.idx_c]  = field[self.idx_c]  * self.plane         + self.electrode_mask(self.u_c,  self.plane)
        return field

    # -------------------------
    # Field Building
    # -------------------------
    def build(self):
        """
        Returns
        -------
        field : numpy.ndarray
            Initialized 3D electric potential
        """
        V = self.build_linear_profile()

        field = np.tile(V, (self.nz, self.ny, 1)).T
        field = self.set_dirichlet(field, self.u_d, self.u_g)
        field = self.apply_electrodes(field)

        return field









class WeightingInitialCondition:
    def __init__(self, p, p_shifted, idx_s, idx_i1, idx_i2, idx_c, nx, ny, nz, shaper, step, shift, view):
        self.plane = p
        self.plane_shifted = p_shifted
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.idx_s = idx_s
        self.idx_i1 = idx_i1
        self.idx_i2 = idx_i2
        self.idx_c = idx_c

        self.shaper = shaper
        self.view = view

        self.step = step
        self.shift = shift

    
    def _set_view(self, view: str):
        if view == "view0":
            return self.idx_i1
            #return self.g['idx_ind1'], self.w['idx_ind1_2D']
        elif view == "view1":
            #return self.g['idx_ind2'], self.w['idx_ind2_2D']
            return self.idx_i2
        elif view == "view2":
            return self.idx_c
            #return self.g['idx_coll'], self.w['idx_coll_2D']
        else:
            raise ValueError(f"Unknown view: {view}")
        
    def readout_strip(self, field, view_idx, strip_shifted=False):
        """
        Parameters
        ----------
        field: numpy.ndarray
        A 3D matrix that contains the volume dimension and the hexagon pattern
        
        view: string
        A string object that correspond to the view where the weighting field will be calculated 
        
        strip_shifted: booleen
        If true: allows to shift the pcb with a definited value
        
        Returns
        ----------
        strip_mask: numpy.ndarray
        A 2D matrix that contains the mask of the charge readout strip
        """
        pcb_mask = field[view_idx]
        if view_idx in (
            self.idx_i1,
            self.idx_i2
        ):
            lminStrip = int(round(field.shape[1] / 2 - self.ny))
            lmaxStrip = int(round(field.shape[1] / 2 + 2 * self.ny))

        elif view_idx in (
            self.idx_c,
        ):
            lminStrip = int(field.shape[1] / 2 - self.ny)
            lmaxStrip = int(field.shape[1] / 2 + self.ny)

        else:
            raise ValueError("Invalid view index for strip definition")

        if strip_shifted:
            idx_shift = int(round(self.shift / self.step))
            lminStrip += idx_shift
            lmaxStrip += idx_shift

        strip_mask = np.zeros_like(pcb_mask)
        strip_mask[lminStrip:lmaxStrip + 1, :][
            pcb_mask[lminStrip:lmaxStrip + 1, :] == 0
        ] = 1

        return strip_mask
    
    def lift_2D_to_3D(self, f_2D: np.ndarray, f_3D: np.ndarray):
        fixed_planes = [self.idx_s, self.idx_i1, self.idx_i2, self.idx_c]
        nx_2D, ny_2D = f_2D.shape
        nx_3D, ny_3D, nz_3D = f_3D.shape

        idx_ymin = (ny_2D - ny_3D) // 2
        idx_ymax = (ny_2D + ny_3D) // 2
        
        idx_xmin = nx_2D - nx_3D

        core_2D = f_2D[idx_xmin:, idx_ymin:idx_ymax]


        mask = np.ones(nx_3D, dtype=bool)
        mask[fixed_planes] = False

        f_3D[mask, :, :] = core_2D[mask, :, None]
        return f_3D

    def build(self):
        """
        Parameters
        ----------
        n_strip: int32
        Integer number giving the strip number that is calculated
        Returns
        ----------
        field: numpy.ndarray
        A 3D matrix that contains the volume dimension and the hexagon pattern
        
        boundary: numpy.ndarray
        A 3D matrix that contains the boundary conditions for the weighting field calculation
        """

        field = np.zeros((  self.nx,
                            self.ny,
                            self.nz), dtype = np.float32)
        field[self.idx_s] = self.plane
        field[self.idx_i1] = self.plane_shifted
        field[self.idx_i2] = self.plane
        field[self.idx_c] = self.plane

        field = self.shaper.extend_volume(field, axis=1)

        view_idx = self._set_view(self.view)

        strip_mask = self.readout_strip(field, view_idx)
        field[view_idx] += strip_mask

        return field