import numpy as np
from numba import njit


"""
Electron mobility and diffusion models.

The transport coefficients implemented here are derived from the global fit
presented in:

Yichen Li, Thomas Tsang, Craig Thorn, Xin Qian, Milind Diwan, Jyoti Joshi,
Steve Kettell, William Morse, Triveni Rao, James Stewart, Wei Tang, Brett Viren,
"Measurement of longitudinal electron diffusion in liquid argon",
Nuclear Instruments and Methods in Physics Research A,
Volume 816 (2016), Pages 160–170.
https://doi.org/10.1016/j.nima.2016.01.094
"""


""" Mobility and diffusion calculations """

@njit
def mobility(E, T):
    """
    Electron mobility in liquid argon.

    Parameters
    ----------
    E : float
        Electric field magnitude [kV/cm]
    T : float
        Liquid argon temperature [K]

    Returns
    -------
    float
        Electron mobility [cm^2 / (V·s)]
    """
    return (a0 + a1 * E + a2 * np.power(E, 3/2) + a3 * np.power(E, 5/2)) / \
           (1 + (a1/a0) * E + a4 * np.power(E, 2) + a5 * np.power(E, 3)) * \
           np.power(T/T0, -3/2)


@njit
def longitudinal_electron_energy(E, T):
    """
    Effective longitudinal electron energy.

    Parameters
    ----------
    E : float
        Electric field magnitude [kV/cm]
    T : float
        Temperature [K]

    Returns
    -------
    float
        Effective electron energy (dimensionless scaling quantity used in
        the diffusion model).
    """
    return (b0 + b1 * E + b2 * np.power(E, 2)) / \
           (1 + (b1 / b0) * E + b3 * np.power(E, 2)) * (T/T1)


@njit
def coeffs_diff(E, T):
    """
    Compute longitudinal and transverse diffusion coefficients.

    The longitudinal diffusion coefficient follows directly from the
    Einstein relation using the effective electron energy.

    The transverse diffusion coefficient is derived from the field
    dependence of the electron mobility.

    Parameters
    ----------
    E : float
        Electric field [kV/cm]
    T : float
        Temperature [K]

    Returns
    -------
    Dt : float
        Transverse diffusion coefficient
    Dl : float
        Longitudinal diffusion coefficient
    """

    Dl = mobility(E, T) * longitudinal_electron_energy(E, T)

    u = a0 + a1 * E + a2 * np.power(E, 3/2) + a3 * np.power(E, 5/2)
    uprime = a1 + 3/2 * a2 * np.power(E, 1/2) + 5/2 * a3 * np.power(E, 3/2)

    w = 1 + (a1/a0) * E + a4 * np.power(E, 2) + a5 * np.power(E, 3)
    wprime = (a1/a0) + 2 * a4 * E + 3 * a5 * np.power(E, 2)

    delta_mu = np.power((T/T0), -3/2) * (uprime * w - u * wprime) / np.power(w, 2)

    Dt = Dl / (1 + E / mobility(E, T) * delta_mu)

    return Dt, Dl


@njit
def coeffs_Astra(E, T):
    """
    Alternative diffusion model used the Astrazev model.

    Parameters
    ----------
    E : float
        Electric field [kV/cm]
    T : float
        Temperature [K]

    Returns
    -------
    Dt : float
        Transverse diffusion coefficient
    Dl : float
        Longitudinal diffusion coefficient
    """

    epsilonT = 0.8 * T * (E/Eh) * 1.38e-23 / 1.6e-19
    epsilonL = 0.5 * epsilonT

    mu = mobility(E, T)

    Dt = epsilonT * mu
    Dl = epsilonL * mu

    return Dt, Dl


@njit
def coeffs_translate(E):
    """
    Polynomial parameterization of diffusion coefficients.

    Parameters
    ----------
    E : float
        Electric field [V/cm]

    Returns
    -------
    Dt : float
        Transverse diffusion coefficient
    Dl : float
        Longitudinal diffusion coefficient
    """

    Elog = np.log10(E + 1e-5)

    Dt = c0t * np.power(Elog, 4) + c1t * np.power(Elog, 3) + \
         c2t * np.power(Elog, 2) + c3t * Elog + c4t

    Dl = c0l * np.power(Elog, 5) + c1l * np.power(Elog, 4) + \
         c2l * np.power(Elog, 3) + c3l * np.power(Elog, 2) + \
         c4l * Elog + c5l

    return Dt, Dl


@njit
def transverse_diffusion(Dt, E, e, T, t, p):
    """
    Simulate transverse diffusion of an electron during one time step.

    The displacement is sampled from a Gaussian distribution with
    variance determined by the transverse diffusion coefficient.

    The displacement is applied in the plane perpendicular to the
    electric field direction.

    Parameters
    ----------
    Dt : float
        Transverse diffusion coefficient
    E : float
        Electric field magnitude
    e : array
        Electric field vector
    T : float
        Temperature [K]
    t : float
        Time step [s]
    p : array
        Transverse displacement vector

    Returns
    -------
    p : array
        Updated transverse displacement vector
    Dt : float
        Transverse diffusion coefficient
    """

    a = np.zeros(3, dtype=np.float32)

    sigma = np.sqrt(2 * Dt * 1e2 * t)

    gt1 = np.random.normal(loc=0.0, scale=sigma)
    gt2 = np.random.normal(loc=0.0, scale=sigma)

    # Choose a vector not parallel to the electric field
    if e[0] != 0:
        a[1] = 1
    elif e[1] != 0:
        a[2] = 1
    elif e[2] != 0:
        a[0] = 1

    # Build orthogonal basis perpendicular to the field
    u1 = np.cross(e, a)
    u2 = np.cross(e, u1)

    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 / np.linalg.norm(u2)

    p[0] = u1[0] * gt1 + u2[0] * gt2
    p[1] = u1[1] * gt1 + u2[1] * gt2
    p[2] = u1[2] * gt1 + u2[2] * gt2

    return p, Dt


@njit
def longitudinal_diffusion(E, T, t, s, Dl):
    """
    Apply longitudinal diffusion along the drift direction.

    The displacement is sampled from a Gaussian distribution with
    variance determined by the longitudinal diffusion coefficient.

    Parameters
    ----------
    E : float
        Electric field
    T : float
        Temperature [K]
    t : float
        Time step [s]
    s : float
        Deterministic drift step
    Dl : float
        Longitudinal diffusion coefficient

    Returns
    -------
    float
        Updated longitudinal step including diffusion
    float
        Longitudinal diffusion coefficient
    float
        Effective electron energy
    """

    sigma = np.sqrt(2 * Dl * 1e2 * t)

    return s + np.random.normal(loc=0.0, scale=sigma), Dl, longitudinal_electron_energy(E, T)


""" 
The dielectric matrix contains the dielectric constant values
of each voxel of the simulation volume.
"""


""" Fit parameters """
T0 = np.float32(89)
a0 = np.float32(551.6) # electron fit_BNL in cm2/V/s at E=0V/cm, 89K
a1 = np.float32(7158.3)
a2 = np.float32(4440.43)
a3 = np.float32(4.29)
a4 = np.float32(43.63)
a5 = np.float32(0.2053)

b0 = np.float32(0.0075)
b1 = np.float32(742.9)
b2 = np.float32(3269.6)
b3 = np.float32(31678.2)
T1 = np.float32(87)

Eh = 0.18

c0t = 0.36699705
c1t = -1.76518699
c2t = 2.50247415 
c3t = -0.91067945
c4t = 5.3908162

c0l = 0.01834564
c1l = 0.10165009
c2l = -0.64824901
c3l = -0.52937704
c4l = 3.49993323
c5l = 3.19617058