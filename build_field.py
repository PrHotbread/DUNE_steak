import numpy as np 

from initial_conditions import DriftInitialCondition
from initial_conditions import WeightingInitialCondition

from boundary import set_drift_boundary
from boundary import set_weighting_boundary

from physics.dielec import dielectric_tensor
from settings import geometry as geo

class PotentialField:
    def __init__(self, p:np.ndarray, p_shifted:np.ndarray, 
                 nx:int, ny:int, nz:int, 
                 idx_s:int, idx_i1:int, idx_i2:int, idx_c:int, idx_d:int, idx_g:int, 
                 e_Ar:float, e_fr4:float, e_Cu:float):
        
        self.p = p
        self.p_shifted = p_shifted
        
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        self.idx_d = idx_d
        self.idx_s = idx_s
        self.idx_i1 = idx_i1
        self.idx_i2 = idx_i2
        self.idx_c = idx_c
        self.idx_g = idx_g

        self.e_Ar = e_Ar
        self.e_fr4 = e_fr4
        self.e_Cu = e_Cu


    def set_drift(self, u_d:float, u_s:float, u_i1:float, u_i2:float, u_c:float, u_g:float):
        drift_ic = DriftInitialCondition(
            self.p, self.p_shifted,
            self.ny, self.nz,
            self.idx_d, self.idx_s, self.idx_i1, self.idx_i2, self.idx_c, self.idx_g,
            u_d, u_s, u_i1, u_i2, u_c, u_g,
        )
        
        f = drift_ic.build()

        d = dielectric_tensor(self.idx_s, self.idx_i1, self.idx_i2, self.idx_c,
                            self.nx, self.ny, self.nz,
                            self.e_Ar, self.e_fr4, self.e_Cu, 
                            self.p, self.p_shifted)
        
        b = set_drift_boundary(f, self.p, self.p_shifted, 
                            self.idx_s, self.idx_i1, self.idx_i2, self.idx_c)
        
        return f, d, b


    def set_weighting(self, n_strip:int, step:float, hole_shifted:float, view:str):

        shaper = geo.FieldShaper(n_strip)
        self.weighting_ic = WeightingInitialCondition(
            self.p, self.p_shifted, 
            self.idx_s, self.idx_i1, self.idx_i2, self.idx_c, 
            self.nx, self.ny, self.nz,
            shaper, step, hole_shifted, view
        )
        f = self.weighting_ic.build()

        d = dielectric_tensor(self.idx_s, self.idx_i1, self.idx_i2, self.idx_c,
                            self.nx, self.ny, self.nz,
                            self.e_Ar, self.e_fr4, self.e_Cu, 
                            self.p, self.p_shifted)
        
        d = shaper.extend_volume(d, axis=1)
        b = set_weighting_boundary(self.p, self.p_shifted, self.idx_s, self.idx_i1, self.idx_i2, self.idx_c, self.nx, self.ny, self.nz, shaper)

        #f_2D, d_2D, b_2D = f[:, :, -1], d[:,:,-1], b[:,:,-1]

        return f, d, b
    
    def set_weighting_2D(self, n_strip:int, step:float, hole_shifted:float, view:str):
        f, d, b = self.set_weighting(n_strip, step, hole_shifted, view)
        return f[:,:,-1], d[:,:,-1], b[:,:,-1]


    def set_weighting_3D(self, n_strip:int, step:float, hole_shifted:float, view:str, f_2D:np.ndarray):
        f_3D, d_3D, b_3D = self.set_weighting(n_strip, step, hole_shifted, view)
        return self.weighting_ic.lift_2D_to_3D(f_2D, f_3D), d_3D, b_3D

    def extract_boundary_planes(self, f_3D:np.ndarray):
        return f_3D[:, 0, :].copy(), f_3D[:, -1, :].copy()
    





class ElectricField:
    def __init__(self, step_x:float, step_yz:float):
        self.step_x = step_x
        self.step_yz = step_yz

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    def _set_drift_boundary(self, V, Ex, Ey, Ez):
        """Boundary conditions for drift field"""

        # X boundaries (non-mirror)
        Ex[0, :, :]  = - (V[1, :, :]  - V[0, :, :])  / self.step_x
        Ex[-1, :, :] = - (V[-1, :, :] - V[-2, :, :]) / self.step_x

        # Y boundaries (mirror)
        Ey[:, 0, :]  = 0.0
        Ey[:, -1, :] = 0.0

        # Z boundaries (mirror)
        Ez[:, :, 0]  = 0.0
        Ez[:, :, -1] = 0.0

        # Global sign convention (drift direction)
        return -Ex, -Ey, -Ez
    
    def _set_weighting_boundary(self, V, Ex, Ey, Ez):
        """Boundary conditions for weighting field"""

        # X boundaries (non-mirror)
        Ex[0, :, :]  = - (V[1, :, :]  - V[0, :, :])  / self.step_x
        Ex[-1, :, :] = - (V[-1, :, :] - V[-2, :, :]) / self.step_x

        # Y boundaries (non-mirror)
        Ey[:, 0, :]  = - (V[:, 1, :]  - V[:, 0, :])  / self.step_yz
        Ey[:, -1, :] = - (V[:, -1, :] - V[:, -2, :]) / self.step_yz

        # Z boundaries (mirror)
        Ez[:, :, 0]  = 0.0
        Ez[:, :, -1] = 0.0

        return Ex, Ey, Ez

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------
    def compute_field(self, field_potential: np.ndarray, type_field: str):
        """
        Compute electric field components:
            E = -∇V
        """

        Ex = np.zeros_like(field_potential)
        Ey = np.zeros_like(field_potential)
        Ez = np.zeros_like(field_potential)

        print("Electric field calculation : $\\vec{E} = -\\nabla V$")

        # Interior points (centered finite differences)
        Ex[1:-1, :, :] = - (field_potential[2:, :, :] - field_potential[:-2, :, :]) / (2 * self.step_x)
        Ey[:, 1:-1, :] = - (field_potential[:, 2:, :] - field_potential[:, :-2, :]) / (2 * self.step_yz)
        Ez[:, :, 1:-1] = - (field_potential[:, :, 2:] - field_potential[:, :, :-2]) / (2 * self.step_yz)

        # Boundary conditions
        if type_field == "drift":
            Ex, Ey, Ez = self._set_drift_boundary(field_potential, Ex, Ey, Ez)
        elif type_field == "weighting":
            Ex, Ey, Ez = self._set_weighting_boundary(field_potential, Ex, Ey, Ez)
        else:
            raise ValueError("type_field must be 'drift' or 'weighting'")
        return Ex, Ey, Ez