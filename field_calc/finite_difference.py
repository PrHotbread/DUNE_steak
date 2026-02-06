# -*- coding: utf-8 -*
import numba as nb

################################################################################
# Finite Difference Methods (FDM) – Numba-compiled kernels
#
# These routines solve Laplace / Poisson equations for electric potential
# using finite difference schemes on a structured grid.
#
# Boundary handling is controlled by the integer mask B:
#   B = 0 : equipotential (Dirichlet)
#   B = 1 : interior point
#   B = 2–5 : edges (Neumann: ∂E/∂n = 0, reflected boundary)
#   B = 6–9 : corners (Neumann: ∂E/∂n = 0, reflected boundary)
#
# Dielectric effects are included through D (permittivity or weights).
################################################################################



""" Allow to compilate the part of the code """
@nb.jit(nopython = True)
def fdm_drift(F1, B):
    """
    Finite difference solver for drift potential (Laplace equation).

    - Uses a 3D 6-point stencil
    - Neumann boundary conditions on edges and corners
    - Dirichlet conditions enforced via mask B
    """
    F2 = F1.copy()
    for i in range(1, F1.shape[0]-1): # Dirichlet conditions - let field[0] = a; field[-1] = b
        for j in range(F1.shape[1]):
            for k in range(F1.shape[2]):
                """ Equipotential case """
                if B[i,j,k] == 0:
                    continue
                    """ general case"""
                elif B[i,j,k] == 1:
                    F2[i,j,k] = (F1[i+1,j,k] + F1[i,j+1,k] + F1[i,j,k+1] +  F1[i-1,j,k] +  F1[i,j-1,k] + F1[i,j,k-1])/6
                    """ edges reflected boundary conditions  - Newman conditions dE/dn = 0"""
                elif B[i,j,k] == 2:
                    F2[i,j,0] = (F1[i+1,j,0] + F1[i-1,j,0] + F1[i,j+1,0] + F1[i,j-1,0] + 2 * F1[i,j, 1])/6
                    
                elif B[i,j,k] == 3:
                    F2[i,j,-1] = (F1[i+1,j,-1] + F1[i,j+1,-1] + 2 * F1[i,j,-2] +  F1[i-1,j,-1] + F1[i,j-1,-1]) /6
                    
                elif B[i,j,k] == 4:
                    F2[i,0,k] = (F1[i+1,0,k] + F1[i-1,0,k] + F1[i,1,k] + F1[i,-2,-k-1] +   F1[i,0,k+1] + F1[i,0,k-1] )/6
                    
                elif B[i,j,k] == 5:
                    F2[i,-1,k] = (F1[i+1,-1,k] + F1[i-1,-1,k] + F1[i,-2,k] + F1[i,1,-k-1] +  F1[i,-1, k+1]  + F1[i,-1,k-1] )/6
                    
                    """ corners reflected boundary conditions - Newman conditions dE/dn = 0"""
                elif B[i,j,k] == 6:
                    F2[i,0,0] = (F1[i+1, 0, 0] + F1[i, 1, 0] + F1[i,-1,-2] + F1[i, 0, 1] +  F1[i,-2,-1] +  F1[i-1, 0, 0])/6
                    
                elif B[i,j,k] == 7:
                    F2[i, 0, -1] = ( F1[i+1, 0, -1] + F1[i, 1, -1] + F1[i,-1,1] + F1[i, 0, -2] +  F1[i,-2,0] +  F1[i-1, -2,0])/6
                    
                elif B[i,j,k] == 8:
                    F2[i, -1, 0] = (F1[i+1, -1, 0] +  F1[i, -2, 0] + F1[i,0,-2] + F1[i, -1, 1] +  F1[i, 1,-1] +  F1[i-1, -1, 0])/6
                    
                elif B[i,j,k] == 9:
                    F2[i, -1, -1] = (F1[i+1, -1, -1] + F1[i, -2, -1] + F1[i,0,1] + F1[i, -1, -2] +  F1[i,1,0] +  F1[i-1, -1, -1])/6
    return  F2




""" Allow to compilate the part of the code """
@nb.jit(nopython = True)
def fdm_drift_dielec(F1, B, D):
    """
    Finite difference solver for drift potential (Laplace equation).

    - Uses a 3D 6-point stencil
    - Neumann boundary conditions on edges and corners
    - Dirichlet conditions enforced via mask B
    - Uses dielectric tensor
    """
    F2 = F1.copy()
    for i in range(1, F1.shape[0]-1): # Dirichlet conditions - let field[0] = a; field[-1] = b
        for j in range(F1.shape[1]):
            for k in range(F1.shape[2]):
                """ equipotential """
                if B[i,j,k] == 0:
                    continue
                    """ general case"""
                elif B[i,j,k] == 1:
                    F2[i,j,k] = (D[i+1,j,k]  * F1[i+1,j,k] + D[i,j+1,k] * F1[i,j+1,k] + D[i,j,k+1] * F1[i,j,k+1] + D[i-1,j,k] * F1[i-1,j,k] + D[i,j-1,k] * F1[i,j-1,k] + D[i,j,k-1] * F1[i,j,k-1]) / (D[i+1,j,k] + D[i,j+1,k] + D[i,j,k+1] + D[i-1,j,k] + D[i,j-1,k] + D[i,j,k-1])
                    """ edges reflected boundary conditions  - Newman conditions dE/dn = 0"""
                elif B[i,j,k] == 2:
                    F2[i,j,0] = (D[i+1,j,0] * F1[i+1,j,0] + D[i-1,j,0] * F1[i-1,j,0] + D[i,j+1,0] * F1[i,j+1,0] + D[i,j-1,0] * F1[i,j-1,0] + 2 * D[i,j, 1] * F1[i,j, 1]) / (D[i+1,j,0] + D[i,j+1,0] + 2 * D[i,j, 1] + D[i-1,j,0] + D[i,j-1,0])
                    
                elif B[i,j,k] == 3:
                    F2[i,j,-1] = (D[i+1,j,-1] * F1[i+1,j,-1] + D[i,j+1,-1] * F1[i,j+1,-1] + 2 * D[i,j,-2] * F1[i,j,-2] + D[i-1,j,-1] * F1[i-1,j,-1] + D[i,j-1,-1] * F1[i,j-1,-1]) / (D[i+1,j,-1] + D[i,j+1,-1] + 2 * D[i,j,-2] + D[i-1,j,-1] + D[i,j-1,-1])
                    
                elif B[i,j,k] == 4:
                    F2[i,0,k] = (D[i+1,0,k] * F1[i+1,0,k] + D[i-1,0,k] * F1[i-1,0,k] + D[i,1,k] * F1[i,1,k] + D[i,-2,-k-1] * F1[i,-2,-k-1] +  D[i,0,k+1] * F1[i,0,k+1] + D[i,0,k-1] * F1[i,0,k-1] ) / (D[i+1,0,k] + D[i,1,k] + D[i,-2,-k-1] + D[i, 0, k+1] + D[i-1,0,k]  + D[i,0,k-1])
                    
                elif B[i,j,k] == 5:
                    F2[i,-1,k] = (D[i+1,-1,k] * F1[i+1,-1,k] + D[i-1,-1,k] * F1[i-1,-1,k] + D[i,-2,k] * F1[i,-2,k] + D[i,1,-k-1] * F1[i,1,-k-1] + D[i,-1, k+1] * F1[i,-1, k+1]  + D[i,-1,k-1] * F1[i,-1,k-1] ) / (D[i+1,-1,k] + D[i,-2,k] + D[i,1,-k-1] + D[i,-1, k+1] + D[i-1,-1,k] + D[i,-1,k-1])
                    
                    """ corners reflected boundary conditions - Newman conditions dE/dn = 0"""
                elif B[i,j,k] == 6:
                    F2[i,0,0] = (D[i+1, 0, 0] * F1[i+1, 0, 0] + D[i, 1, 0] * F1[i, 1, 0] + D[i,-1,-2] * F1[i,-1,-2] + D[i, 0, 1] * F1[i, 0, 1] + D[i,-2,-1] * F1[i,-2,-1] + D[i-1, 0, 0] * F1[i-1, 0, 0])/ (D[i+1, 0, 0] + D[i, 1, 0] + D[i,-1,-2] + D[i, 0, 1] + D[i,-2,-1] + D[i-1, 0, 0])
                    
                elif B[i,j,k] == 7:
                    F2[i, 0, -1] = (D[i+1, 0, -1] * F1[i+1, 0, -1] + D[i, 1, -1] * F1[i, 1, -1] + D[i,-2,1] * F1[i,-1,1] + D[i, 0, -2] * F1[i, 0, -2] + D[i,-2,0] * F1[i,-2,0] + D[i-1, -2,0] * F1[i-1, -2,0])/(D[i+1, 0, -1]+ D[i, 1, -1] + D[i,-1,1] + D[i, 0, -2] + D[i,-2,0] + D[i-1, -2,0])
                    
                elif B[i,j,k] == 8:
                    F2[i, -1, 0] = (D[i+1, -1, 0] * F1[i+1, -1, 0] + D[i, -2, 0] * F1[i, -2, 0] + D[i,0,-2]  * F1[i,0,-2] + D[i, -1, 1] * F1[i, -1, 1] + D[i, 1,-1] * F1[i, 1,-1] + D[i-1, -1, 0] * F1[i-1, -1, 0])/(D[i+1, -1, 0] + D[i, -2, 0] + D[i,0,-2]  + D[i, -1, 1] + D[i, 1,-1] + D[i-1, -1, 0])
                    
                elif B[i,j,k] == 9:
                    F2[i, -1, -1] = (D[i+1, -1, -1] * F1[i+1, -1, -1] + D[i, -2, -1] * F1[i, -2, -1] + D[i,0,1] * F1[i,0,1] + D[i,-1,-2] * F1[i, -1, -2] + D[i,1,0] * F1[i,1,0] + D[i-1, -1, -1] * F1[i-1, -1, -1])/(D[i+1, -1, -1] + D[i, -2, -1] + D[i,0,1] + D[i, -1, -2] + D[i,1,0] + D[i-1, -1, -1])
    return  F2


@nb.jit(nopython = True)
def fdm_weighting_dielec(F1, B, D):
    """
    Finite difference solver for drift potential (Laplace equation).

    - Uses a 3D 6-point stencil
    - Neumann boundary conditions on edges and corners
    - Dirichlet conditions enforced via mask B
    """
    F2 = F1.copy()
    for i in range(1, F1.shape[0]-1): # Dirichlet conditions - let field[0] = a; field[-1] = b
        for j in range(F1.shape[1]):
            for k in range(F1.shape[2]):
                """ Equipotential case """
                if B[i,j,k] == 0:
                    continue
                    """ general case"""
                elif B[i,j,k] == 1:
                    F2[i,j,k] = (D[i+1,j,k]  * F1[i+1,j,k] + D[i,j+1,k] * F1[i,j+1,k] + D[i,j,k+1] * F1[i,j,k+1] + D[i-1,j,k] * F1[i-1,j,k] + D[i,j-1,k] * F1[i,j-1,k] + D[i,j,k-1] * F1[i,j,k-1]) / (D[i+1,j,k] + D[i,j+1,k] + D[i,j,k+1] + D[i-1,j,k] + D[i,j-1,k] + D[i,j,k-1])
                    """ edges reflected boundary conditions  - Newman conditions dE/dn = 0"""
                elif B[i,j,k] == 2:
                    F2[i,j,0] = (D[i+1,j,0]  * F1[i+1,j,0] + D[i,j+1,0] * F1[i,j+1,0] + 2 * D[i,j, 1] * F1[i,j, 1] + D[i-1,j,0] * F1[i-1,j,0] + D[i,j-1,0] * F1[i,j-1,0]) / (D[i+1,j,0] + D[i,j+1,0] + 2 * D[i,j, 1] + D[i-1,j,0] + D[i,j-1,0])
                    
                elif B[i,j,k] == 3:
                    F2[i,j,-1] = (D[i+1,j,-1] * F1[i+1,j,-1] + D[i,j+1,-1] * F1[i,j+1,-1] + 2 * D[i,j,-2] * F1[i,j,-2] + D[i-1,j,-1] * F1[i-1,j,-1] + D[i,j-1,-1] * F1[i,j-1,-1]) / (D[i+1,j,-1] + D[i,j+1,-1] + 2 * D[i,j,-2] + D[i-1,j,-1] + D[i,j-1,-1])
                    
                elif B[i,j,k] == 4:
                    F2[i,0,k] = 0 #(F1[i+1,0,k] + 2 * F1[i,1,k] +  F1[i,0, k+1 ] + F1[i-1,0,k] + F1[i,0,k-1] )/6
                    
                elif B[i,j,k] == 5:
                    F2[i,-1,k] = 0 #(F1[i+1,-1,k] + 2 * F1[i,-2,k] +  F1[i,-1, k+1] +F1[i-1,-1,k] + F1[i,-1,k-1] )/6
                    
                    """ corners reflected boundary conditions - Newman conditions dE/dn = 0"""
                elif B[i,j,k] == 6:
                    F2[i,0,0] = 0 #(F1[i+1, 0, 0] + 2 * F1[i, 1, 0] + 2 * F1[i, 0, 1] + F1[i-1, 0, 0])/6
                    
                elif B[i,j,k] == 7:
                    F2[i, 0, -1] = 0 #(F1[i+1, 0, -1] + 2 * F1[i, 1, -1] + 2 * F1[i, 0, -2] + F1[i-1, 0, -1])/6
                    
                elif B[i,j,k] == 8:
                    F2[i, -1, 0] = 0# (F1[i+1, -1, 0] + 2 * F1[i, -2, 0] + 2 * F1[i, -1, 1] + F1[i-1, -1, 0])/6
                    
                elif B[i,j,k] == 9:
                    F2[i, -1, -1] = 0# (F1[i+1, -1, -1] + 2 * F1[i, -2, -1] + 2 * F1[i, -1, -2] + F1[i-1, -1, -1])/6
    return F2


@nb.jit(nopython = True)
def fdm_weighting_boundary(F1, B, D, b_min, b_max):
    """
    Finite difference solver for drift potential (Laplace equation).

    - Uses a 3D 6-point stencil
    - Neumann boundary conditions on edges and corners
    - Dirichlet conditions enforced via mask B
    - Fixe the boundary condition via the 2D calculation
    """
    F2 = F1.copy()
    for i in range(1, F1.shape[0]-1): # Dirichlet conditions - let field[0] = a; field[-1] = b
        for j in range(F1.shape[1]):
            for k in range(F1.shape[2]):
                """ Equipotential case """
                if B[i,j,k] == 0:
                    continue
                    """ general case"""
                elif B[i,j,k] == 1:
                    F2[i,j,k] = (D[i+1,j,k]  * F1[i+1,j,k] + D[i,j+1,k] * F1[i,j+1,k] + D[i,j,k+1] * F1[i,j,k+1] + D[i-1,j,k] * F1[i-1,j,k] + D[i,j-1,k] * F1[i,j-1,k] + D[i,j,k-1] * F1[i,j,k-1]) / (D[i+1,j,k] + D[i,j+1,k] + D[i,j,k+1] + D[i-1,j,k] + D[i,j-1,k] + D[i,j,k-1])
                    """ edges reflected boundary conditions  - Newman conditions dE/dn = 0"""
                elif B[i,j,k] == 2:
                    F2[i,j,0] = (D[i+1,j,0]  * F1[i+1,j,0] + D[i,j+1,0] * F1[i,j+1,0] + 2 * D[i,j, 1] * F1[i,j, 1] + D[i-1,j,0] * F1[i-1,j,0] + D[i,j-1,0] * F1[i,j-1,0]) / (D[i+1,j,0] + D[i,j+1,0] + 2 * D[i,j, 1] + D[i-1,j,0] + D[i,j-1,0])
                    
                elif B[i,j,k] == 3:
                    F2[i,j,-1] = (D[i+1,j,-1] * F1[i+1,j,-1] + D[i,j+1,-1] * F1[i,j+1,-1] + 2 * D[i,j,-2] * F1[i,j,-2] + D[i-1,j,-1] * F1[i-1,j,-1] + D[i,j-1,-1] * F1[i,j-1,-1]) / (D[i+1,j,-1] + D[i,j+1,-1] + 2 * D[i,j,-2] + D[i-1,j,-1] + D[i,j-1,-1])
               
                elif B[i,j,k] == 4:
                    F2[i,0,k] = b_min[i,k] #(F1[i+1,0,k] + 2 * F1[i,1,k] +  F1[i,0, k+1 ] + F1[i-1,0,k] + F1[i,0,k-1] )/6
                    
                elif B[i,j,k] == 5:
                    F2[i,-1,k] = b_max[i,k] #(F1[i+1,-1,k] + 2 * F1[i,-2,k] +  F1[i,-1, k+1] +F1[i-1,-1,k] + F1[i,-1,k-1] )/6
                    
                    """ corners reflected boundary conditions - Newman conditions dE/dn = 0"""
                elif B[i,j,k] == 6:
                    F2[i,0,0] = b_min[i,0] #y0[i,0] #(F1[i+1, 0, 0] + 2 * F1[i, 1, 0] + 2 * F1[i, 0, 1] + F1[i-1, 0, 0])/6
                    
                elif B[i,j,k] == 7:
                    F2[i, 0, -1] = b_min[i,-1] #y0[i,-1] #(F1[i+1, 0, -1] + 2 * F1[i, 1, -1] + 2 * F1[i, 0, -2] + F1[i-1, 0, -1])/6
                    
                elif B[i,j,k] == 8:
                    F2[i, -1, 0] = b_max[i,0]# y1[i,0]# (F1[i+1, -1, 0] + 2 * F1[i, -2, 0] + 2 * F1[i, -1, 1] + F1[i-1, -1, 0])/6
                    
                elif B[i,j,k] == 9:
                    F2[i, -1, -1] = b_max[i,-1] #y0[i,-1]# (F1[i+1, -1, -1] + 2 * F1[i, -2, -1] + 2 * F1[i, -1, -2] + F1[i-1, -1, -1])/6
    return F2




@nb.jit(nopython = True)
def fdm_weighting_2D(F1, B, D):
    """
    Finite difference solver for drift potential (Laplace equation).

    - Uses a 2D 4-point stencil
    - Neumann boundary conditions on edges and corners
    - Dirichlet conditions enforced via mask B
    - Allow to fixe the boundary condition for the 3D calculation field 
    """
    F2 = F1.copy()
    for i in range(1, F1.shape[0] -1): # Dirichlet conditions - let field[0] = a; field[-1] = b
        for j in range(F1.shape[1]):
                """ Equipotential case """
                if B[i,j] == 0:
                    continue
                    """ general case"""
                elif B[i,j] == 3 or B[i,j] == 1:
                    F2[i,j] = (D[i+1,j]  * F1[i+1,j] + D[i,j+1] * F1[i,j+1]  + D[i-1,j] * F1[i-1,j] + D[i,j-1] * F1[i,j-1]) / (D[i+1,j] + D[i,j+1] + D[i-1,j] + D[i,j-1])
                    """ fixe to 0 far away from the readout strip"""
                elif B[i,j] == 4 or B[i,j] == 5 or B[i,j] == 6 or B[i,j] == 7 or B[i,j] == 8 or B[i,j] == 9:
                    F2[i,0] = 0
    return F2