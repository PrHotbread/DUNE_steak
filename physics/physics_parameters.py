import numpy as np
from numba import njit
     



""" 
Mobility and diffusion calculation is obtained from the global fit given on: 
"
Yichen Li, Thomas Tsang, Craig Thorn, Xin Qian, Milind Diwan, Jyoti Joshi, Steve Kettell, William Morse, Triveni Rao, James Stewart, Wei Tang, Brett Viren,
Measurement of longitudinal electron diffusion in liquid argon,
Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment,
Volume 816,
2016,
Pages 160-170,
ISSN 0168-9002,
https://doi.org/10.1016/j.nima.2016.01.094. "
"""

""" Diffusion and mobility calculation """
@njit
def mobility(E,T):
    return (a0 + a1 * E + a2 * np.power(E, 3/2) + a3 * np.power(E, 5/2)) / (1 + (a1/a0) * E + a4 * np.power(E, 2) + a5 * np.power(E, 3)) * np.power(T/T0, -3/2)
    
@njit
def longitudinal_electron_energy(E, T):
    return  (b0 + b1 * E + b2 * np.power(E, 2)) / ( 1 + (b1 / b0) * E + b3 * np.power(E, 2)) * (T/T1)


@njit 
def coeffs_diff(E, T):
    Dl =  (mobility(E,T) * longitudinal_electron_energy(E, T))
    u = a0 + a1 * E + a2 * np.power(E, 3/2) + a3 * np.power(E, 5/2)
    uprime = a1 + 3/2 * a2 * np.power(E, 1/2) + 5/2 * a3 * np.power(E, 3/2)
    
    w = 1 + (a1/a0) * E + a4 * np.power(E, 2) + a5 * np.power(E, 3)
    wprime = (a1/a0) + 2 * a4 * E + 3 * a5 * np.power(E, 2)
    
    delta_mu = np.power((T/T0), -3/2) * (uprime * w - u * wprime)/np.power(w, 2)
    
    Dt = Dl / (1 + E / mobility(E,T) * delta_mu)
    return Dt, Dl

@njit
def coeffs_Astra(E, T): # Attention E {kV/cm}
    epsilonT = 0.8 * T * (E/Eh) * 1.38e-23/1.6e-19
    epsilonL = 0.5 * epsilonT 
    mu = mobility(E, T)
    Dt = epsilonT * mu
    Dl = epsilonL * mu
    return Dt, Dl

@njit
def coeffs_translate(E):
    Elog = np.log10(E + 1e-5) 
    Dt = c0t * np.power(Elog, 4) + c1t * np.power(Elog, 3) + c2t * np.power(Elog, 2) + c3t * Elog + c4t
    Dl = c0l * np.power(Elog, 5) + c1l * np.power(Elog, 4) + c2l * np.power(Elog, 3) + c3l * np.power(Elog, 2) + c4l * Elog + c5l
    return Dt, Dl




@njit
def transverse_diffusion(Dt, E, e, T, t, vector_position):
#    a = vector_base[0]
#    b = vector_base[1]
#    u1 = vector_base[2
#    u2 = vector_base[3]
    #e = np.array([ex, ey, ez], dtype = np.float32)
    #Dt = coeff_transverse(Dl, E, T)
    a = np.zeros(3, dtype = np.float32)
    
    sigma = np.sqrt(2 * Dt * 1e2 * t)
    
    gt1 = np.random.normal(loc=0.0, scale=sigma)
    gt2 = np.random.normal(loc=0.0, scale=sigma)
    """ Coordonnées initiales """
    
    if e[0] != 0:
        a[1] = 1  # Si v[0] n'est pas nul, choisir a = (0, 1, 0)
    elif e[1] != 0:
        a[2] = 1  # Si v[1] n'est pas nul, choisir a = (1, 0, 0)
    elif e[2] != 0:
        a[0] = 1  # Si v est colinéaire à (0, 0, z), choisir a = (1, 0, 0)
 
    u1 = np.cross(e, a)
    
    u2 = np.cross(e, u1)
    
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 / np.linalg.norm(u2)
    
    vector_position[0] = u1[0] * gt1 + u2[0] * gt2
    vector_position[1] = u1[1] * gt1 + u2[1] * gt2
    vector_position[2] = u1[2] * gt1 + u2[2] * gt2
    return vector_position, Dt
        
@njit
def longitudinal_diffusion(E, T, t, s, Dl):
    sigma = np.sqrt(2 * Dl * 1e2 * t)
    return s + np.random.normal(loc = 0.0, scale = sigma), Dl, longitudinal_electron_energy(E, T)


    

""" Dielectric matrix contains the dielectric constant values """

 

"""Fit parameters"""
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