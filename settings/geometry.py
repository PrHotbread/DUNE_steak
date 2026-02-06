import numpy as np


class DetectorGeometry:
    """
    Base class for detector geometries PCB
    """

    def __init__(self, ny, nz, step):
        self.ny = ny
        self.nz = nz
        self.step = step

    def out_circle(self, mesh, z, y, r):

        Y, Z = np.meshgrid(
            np.arange(self.ny) * self.step,
            np.arange(self.nz) * self.step,
            indexing = 'ij'
        )
        mask = (Z - z)**2 + (Y - y)**2 <= r**2
        mesh[mask] = 1
        
        return mesh
    
    def empty_plane(self):
        return np.zeros((self.ny, self.nz))

    def circle_hole(self, mesh, y, z, r):
        return self.out_circle(mesh, z, y, r)

    def plot(self, mesh, title="Geometry"):
        plot_2d_nodes(mesh, self.step, title)


class PCB(DetectorGeometry):
    """
    PCB with circular holes
    """

    def __init__(self, ny, nz, step, r_hole):
        super().__init__(ny, nz, step)
        self.r_hole = r_hole

    def single_hole(self, y, z):
        plane = self.empty_plane()
        return self.circle_hole(plane, y, z, self.r_hole)

    def hex_pattern(self, Ly, Lz, shift):
        plane = self.empty_plane()
        plane = self.circle_hole(plane, Ly + shift, Lz, self.r_hole)
        plane = self.circle_hole(plane, 0 + shift, 0, self.r_hole)
        print("Ly = %0.2f, Lz = %0.2f, ny = %i, nz = %i" %(Ly, Lz, self.ny, self.nz))
        return plane




class FieldShaper:
    """
    Utility class to extend electric potential fields
    by exploiting symmetry and periodicity.

    Supports:
    - 3D volume extension
    - 2D plane extension
    """

    def __init__(self, n_strip: int):
        self.n_strip = n_strip

    @staticmethod
    def _mirror_and_concat(array, axis: int):
        """
        Mirror an array along a given axis and concatenate it,
        removing duplicated boundary planes.
        """
        mirrored = np.flip(array, axis=array.ndim - 1)

        # remove duplicated boundaries
        slicer = [slice(None)] * array.ndim
        slicer[axis] = slice(1, -1)
        mirrored = mirrored[tuple(slicer)]

        return np.concatenate((array, mirrored), axis=axis)


    def extend_volume(self, field: np.ndarray, axis: int):
        """
        Extend a 3D field potential along a given axis
        using mirror symmetry and periodic repetition.
        """

        if field.ndim != 3:
            raise ValueError("extend_volume expects a 3D array")

        # Build one periodic strip
        strip = self._mirror_and_concat(field, axis)

        # Repeat the strip n_strip times
        extended = strip
        for _ in range(self.n_strip - 1):
            extended = np.concatenate((extended, strip), axis=axis)

        return extended

    def extend_plane(self, field: np.ndarray, axis: int):
        """
        Extend a 2D field potential using mirror symmetry
        and periodic tiling.
        """

        if field.ndim != 2:
            raise ValueError("extend_plane expects a 2D array")

        # Build a symmetric unit
        unit = self._mirror_and_concat(field, axis)

        # Duplicate once (hex / periodic pattern)
        extended = np.concatenate((unit, unit), axis=axis)

        return extended





def shape(field_potential, n_strip: int, axis: str):
    """
    Extend a 3D field potential computed on a small window
    by mirroring and concatenating it to build a larger volume.

    The idea is to:
    - compute the electric field on a reduced domain (unit cell)
    - exploit symmetry / periodicity
    - replicate the field to simulate a larger detector volume
    """

    if axis == "1":
        # -------------------------------------------------------------
        # Extension along axis=1 (typically Y direction)
        # -------------------------------------------------------------

        # Mirror the field along the Z direction
        # This enforces a symmetric continuation of the potential
        field_potential_reverse = field_potential[:, :, ::-1]

        # Remove duplicated boundary planes to avoid overlapping nodes
        field_potential_reverse = np.delete(field_potential_reverse, 0, axis=1)
        field_potential_reverse = np.delete(field_potential_reverse, -1, axis=1)

        # Concatenate original + mirrored field to build one periodic strip
        strip = np.concatenate(
            (field_potential, field_potential_reverse),
            axis=1
        )

        # Initialize the extended window
        window = strip.copy()

        # Repeat the strip n_strip times to enlarge the domain
        for i in range(n_strip - 1):
            window = np.concatenate((window, strip), axis=1)

        return window

    elif axis == "2":
        # -------------------------------------------------------------
        # Extension along axis=2 (typically Z direction)
        # -------------------------------------------------------------

        # Mirror the field along the Z direction
        field_potential_reverse = field_potential[:, :, ::-1]

        # Remove duplicated boundary planes to ensure continuity
        field_potential_reverse = np.delete(field_potential_reverse, 0, axis=2)
        field_potential_reverse = np.delete(field_potential_reverse, -1, axis=2)

        # Create a first extended window using mirror symmetry
        mini_hex_window = np.concatenate(
            (field_potential, field_potential_reverse),
            axis=2
        )

        # Duplicate the window to simulate a larger periodic structure
        # (e.g. hexagonal or tiled geometry)
        hex_window = np.concatenate(
            (mini_hex_window, mini_hex_window),
            axis=2
        )

        return hex_window





def plot_2d_nodes(plane, step, title="2D Geometry (nodes)", s=10):
    import matplotlib.pyplot as plt
    """
    Plot a 2D geometry where each mesh node is displayed as a point.

    Parameters
    ----------
    plane : ndarray (ny, nz)
        2D geometry array
    step : float
        Spatial resolution
    title : str
        Plot title
    s : float
        Marker size
    """

    ny, nz = plane.shape

    # Physical coordinates of the nodes
    Y, Z = np.meshgrid(
        np.arange(ny) * step,
        np.arange(nz) * step,
        indexing="ij"
    )

    plt.figure(figsize=(5, 5))

    # Plot empty nodes
    mask_empty = plane == np.min(plane)
    plt.scatter(
        Y[mask_empty],
        Z[mask_empty],
        s=s,
        c="orange",
        marker=".",
        label="Material"
    )

    # Plot hole nodes
    mask_hole = plane == np.max(plane)
    plt.scatter(
        Y[mask_hole],
        Z[mask_hole],
        s=s,
        c="royalblue",
        marker=".",
        label="Hole"
    )

    plt.xlabel("z [mm]")
    plt.ylabel("y [mm]")
    plt.title(title)
    plt.axis("equal")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.show()

"""
import parameters

params = parameters.get_parameters()
g = params['geometry']
p = params['physics']


pcb = PCB(g['ny'] + 1, g['nz'] + 1, g['step'], g['r_hole'])

plane = pcb.hex_pattern(Ly = g['Ly'], Lz = g['Lz'], shift =  0.0)
plane_shifted = pcb.hex_pattern(Ly = g['Ly'], Lz = g['Lz'], shift = g['shift'])

pcb.plot(plane, "PCB – Base Pattern")

shaper = FieldShaper(3)

field_2D_ext = shaper.extend_plane(plane_shifted, axis=0)
field_2D_ext_2 = shaper.extend_plane(field_2D_ext, axis=1)

pcb.plot(field_2D_ext_2, "PCB – Shifted Pattern")
"""
def volume(field, params):
    """
    Generate a 3D coordinate grid corresponding to the field volume.

    Parameters
    ----------
    field : ndarray (ny, nx, nz)
        3D field or potential map
    params : dict
        Spatial discretization parameters

    Returns
    -------
    vol : ndarray
        3D meshgrid containing physical coordinates
    """

    vol = np.round(
        np.meshgrid(
            np.arange(0, field.shape[1] * params['hyz'], params['hyz']),
            np.arange(0, field.shape[0] * params['hx'],  params['hyz']),
            np.arange(0, field.shape[2] * params['hyz'], params['hyz'])
        ),
        3
    )
