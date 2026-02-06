import numpy as np

def set_drift_boundary(field_potential, Ex, Ey, Ez, step_x, step_yz):
    """
    Apply boundary conditions for the *drift field*.

    Boundary behavior:
    - X direction: non-mirror (Dirichlet-type) boundary condition
    - Y direction: mirror (Neumann-type) boundary condition
    - Z direction: mirror (Neumann-type) boundary condition

    Parameters
    ----------
    field_potential : numpy.ndarray
        Electric potential field.
    Ex, Ey, Ez : numpy.ndarray
        Electric field components to be modified at the boundaries.
    step_x : float
        Grid spacing along x.
    step_yz : float
        Grid spacing along y and z.

    Returns
    -------
    Ex, Ey, Ez : numpy.ndarray
        Updated electric field components with boundary conditions applied.
    """

    # --- X boundaries (non-mirror condition) ---
    Ex[0, :, :]  = - (field_potential[1, :, :]  - field_potential[0, :, :])  / step_x
    Ex[-1, :, :] = - (field_potential[-1, :, :] - field_potential[-2, :, :]) / step_x

    # --- Y boundaries (mirror: infinite PCB plane) ---
    Ey[:, 0, :]  = 0 # - (field_potential[:,1,:] - field_potential[:,1,:]) / (2 * step_yz)
    Ey[:, -1, :] = 0 #- (field_potential[:,1,:] - field_potential[:,1,:]) / (2 * step_yz)

    # --- Z boundaries (mirror: infinite PCB plane) ---
    Ez[:, :, 0]  = 0 #- (field_potential[:,1,:] - field_potential[:,1,:]) / (2 * step_yz)
    Ez[:, :, -1] = 0 #- (field_potential[:,1,:] - field_potential[:,1,:]) / (2 * step_yz)

    return Ex, Ey, Ez



def set_weighting_boundary(field_potential, Ex, Ey, Ez, step_x, step_yz):
    """
    Apply boundary conditions for the *weighting field*.

    Boundary behavior:
    - X direction: non-mirror (Dirichlet-type) boundary condition
    - Y direction: non-mirror (Dirichlet-type) boundary condition
    - Z direction: mirror (Neumann-type) boundary condition

    Parameters
    ----------
    field_potential : numpy.ndarray
        Weighting potential field.
    Ex, Ey, Ez : numpy.ndarray
        Electric field components to be modified at the boundaries.
    step_x : float
        Grid spacing along x.
    step_yz : float
        Grid spacing along y and z.

    Returns
    -------
    Ex, Ey, Ez : numpy.ndarray
        Updated electric field components with boundary conditions applied.
    """

    # --- X boundaries (non-mirror condition) ---
    Ex[0, :, :]  = - (field_potential[1, :, :]  - field_potential[0, :, :])  / step_x
    Ex[-1, :, :] = - (field_potential[-1, :, :] - field_potential[-2, :, :]) / step_x

    # --- Y boundaries (non-mirror condition: reach to 0 far away from readout strip) ---
    Ey[:, 0, :]  = - (field_potential[:, 1, :]  - field_potential[:, 0, :])  / step_yz
    Ey[:, -1, :] = - (field_potential[:, :, -1] - field_potential[:, :, -2]) / step_yz

    # --- Z boundaries (mirror: infinite readout strip) ---
    Ez[:, :, 0]  = 0 # - (field_potential[:, :, 1]  - field_potential[:, :, 0])  / step_yz
    Ez[:, :, -1] = 0 # - (field_potential[:, :, -1] - field_potential[:, :, -2]) / step_yz

    return Ex, Ey, Ez



def field_component(field_potential, type_field: str, step_x, step_yz):
    """
    Compute the electric field components from a scalar potential field.

    The electric field is obtained via finite differences:
        E = -∇V

    Interior points use centered finite differences.
    Boundary points are handled separately depending on the field type.

    Parameters
    ----------
    field_potential : numpy.ndarray
        Scalar electric potential field.
    type_field : str
        Type of electric field to compute:
        - "drift"     : physical drift field
        - otherwise  : weighting field
    step_x : float
        Grid spacing along x.
    step_yz : float
        Grid spacing along y and z.

    Returns
    -------
    Ex, Ey, Ez : numpy.ndarray
        Electric field components.
    """

    # Initialize field components
    Ex = np.zeros(field_potential.shape)
    Ey = np.zeros(field_potential.shape)
    Ez = np.zeros(field_potential.shape)

    print("Electric field calculation : $\\vec{E} = -\\nabla V$")

    # --- Interior points: centered finite differences ---
    Ex[1:-1, :, :] = - (field_potential[2:, :, :]   - field_potential[:-2, :, :])   / (2 * step_x)
    Ey[:, 1:-1, :] = - (field_potential[:, 2:, :]   - field_potential[:, :-2, :])   / (2 * step_yz)
    Ez[:, :, 1:-1] = - (field_potential[:, :, 2:]   - field_potential[:, :, :-2])   / (2 * step_yz)

    # --- Boundary conditions depending on field type ---
    if type_field == "drift":
        Ex, Ey, Ez = set_drift_boundary(field_potential, Ex, Ey, Ez, step_x, step_yz)
        # Global sign convention for drift field (follow up the drift velocity direction)
        Ex, Ey, Ez = -Ex, -Ey, -Ez
    else:
        Ex, Ey, Ez = set_weighting_boundary(field_potential, Ex, Ey, Ez, step_x, step_yz)

    return Ex, Ey, Ez
