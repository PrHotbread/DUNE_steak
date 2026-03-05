import numpy as np

class MonteCarlo:

    def __init__(self, ne, volume, step_x, step_yz):
        self.ne = ne
        _, _, ny, nz = volume.shape
        self.center = np.array([
            5, #set up the shileding plane at 0 
            ny / 2 * step_yz,
            nz / 2 * step_yz
        ], dtype=np.float32)
    
    @staticmethod
    def initial_position(self):
        return np.tile(self.center, (self.ne, 1))
    
    def set_position(self, x, y, z):
        p = self.initial_position()
        p[:, 0] += x
        p[:, 1] += y
        p[:, 2] += z
        return p
    
    def set_hexagonal(self, ly, lz):
        p = self.initial_position()
        p[:, 2] += 1 * lz * (2 * np.random.random(self.ne) - 1)
        p[:, 1] += 2 * ly * (2 * np.random.random(self.ne) - 1)
        p[:, 0] += -4
        return p
    
    def cylinder(self, radius, weight):
        p = self.initial_position()
        R =  radius * np.random.random( size = self.ne)
        theta = 2 * np.pi * np.random.random(size = self.ne)
        p[:, 2] += R * np.cos(theta)
        p[:, 1] += R * np.sin(theta)
        p[:, 0] += weight * np.random.random(size = self.ne)
        return p

    def spheric(self, radius):
        p = self.initial_position()
        R =  radius * np.random.random( size = self.ne)
        theta = np. pi * np.random.random(size = self.ne)
        phi = 2 * np.pi * np.random.random(size = self.ne)
        p[:, 2] += R * np.cos(theta)
        p[:, 1] += R * np.sin(theta) * np.cos(phi)
        p[:, 0] += R * np.sin(theta) * np.sin(phi)
        return p