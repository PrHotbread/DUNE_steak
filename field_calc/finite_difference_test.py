# -*- coding: utf-8 -*
import numba as nb

############################ Compilate part using for loops  #######################################@

""" Hardcoded to test 50x25x25 micro step """
@nb.jit(nopython = True)
def fdm_drift_diff_scale(F1, B, **bound_kwargs):
    F2 = F1.copy()
    """ general case """
    for i in range(1, F1.shape[0]-1):
        for j in range(F1.shape[1]):
            for k in range(F1.shape[2]):
                if B[i,j,k] == 0:
                    continue
                elif B[i,j,k] == 1:
                    F2[i,j,k] = 1./18 * ( 4 * (F1[i,j+1,k] +  F1[i,j-1,k] + F1[i,j,k+1] + F1[i,j,k-1] ) + F1[i+1,j,k] +  F1[i-1,j,k])
                elif B[i,j,k] == 2:
                    F2[i,j,0] = 1./18 * ( 4 * (F1[i,j+1,0] +  F1[i,j-1,0] + F1[i,j,1] + F1[i,j,1] ) + F1[i+1,j,0] +  F1[i-1,j,0])
                elif B[i,j,k] ==3:
                    F2[i,j,-1] = 1./18 * ( 4 * (F1[i,j+1,-1] +  F1[i,j-1,-1] + F1[i,j,-2] + F1[i,j,-2] ) + F1[i+1,j,-1] +  F1[i-1,j,-1])
                elif B[i,j,k] ==4:
                    F2[i,0,k] = 1./18 * ( 4 * (F1[i,1,k] +  F1[i,1,k] + F1[i,0,k+1] + F1[i,0,k-1] ) + F1[i+1,0,k] +  F1[i-1,0,k])
                elif B[i,j,k] ==5:
                    F2[i,-1,k] = 1./18 * ( 4 * (F1[i,-2,k] +  F1[i,-2,k] + F1[i,-1,k+1] + F1[i,-1,k-1] ) + F1[i+1,-1,k] +  F1[i-1,-1,k])
                elif B[i,j,k] == 6:
                    F2[i,0,0] = 1./18 * ( 4 * (F1[i,1,0] +  F1[i,1,0] + F1[i,0,1] + F1[i,0,1] ) + F1[i+1,0,0] +  F1[i-1,0,0])
                elif B[i,j,k] ==7:
                    F2[i,0,-1] = 1./18 * ( 4 * (F1[i,1,-1] +  F1[i,1,-1] + F1[i,0,-2] + F1[i,0,-2] ) + F1[i+1,0,-1] +  F1[i-1,0,-1])
                elif B[i,j,k] ==8:
                    F2[i,-1,0] = 1./18 * ( 4 * (F1[i,-2,0] +  F1[i,-2,0] + F1[i,-1,1] + F1[i,-1,1] ) +F1[i+1,-1,0] +  F1[i-1,-1,0])
                elif B[i,j,k] ==9:
                    F2[i,-1,-1] = 1./18 * ( 4 * (F1[i,-2,-1] +  F1[i,-2,-1] + F1[i,-1,-2] + F1[i,-1,-2] ) + F1[i+1,-1,-1] +  F1[i-1,-1,-1])


    
    """ edges reflected boundary conditions  """
    


                         
    """ corners reflected boundary conditions """

    return  F2



@nb.jit(nopython = True)
def fdm_drift_for(F1, dx, dy, dz, B):
    F2 = F1.copy()
    """ general case """
    for i in range(1, F1.shape[0]-1):
        for j in range(F1.shape[1]):
            for k in range( F1.shape[2]):
                if B[i,j,k] == 0:
                    continue
                elif B[i,j,k] == 1:
                    F2[i,j,k] = (1./( 1./(dx[i+1] * dx[i-1]) + 1./(dy[j+1] * dy[j-1]) + 1./(dz[k+1] * dz[k-1]))) * ( 1./(dx[i+1] + dx[i-1]) * (F1[i+1,j,k]/dx[i+1] + F1[i-1,j,k]/dx[i-1])  +
                                                                                                              1./(dy[j+1] + dy[j-1]) * (F1[i,j+1,k]/dy[j+1] + F1[i,j-1,k]/dy[j-1]) +
                                                                                                              1./(dz[k+1] + dz[k-1]) * (F1[i,j,k+1]/dz[k+1] + F1[i,j,k-1]/dz[k-1]))
                
                elif B[i,j,k] == 2:
                    F2[i,j,0] = 1./( 1./(dx[i+1] * dx[i-1]) + 1./(dy[j+1] * dy[j-1]) + 1./(dz[1] * dz[1])) * ( 1./(dx[i+1] + dx[i-1]) * (F1[i+1,j,0]/dx[i+1] + F1[i-1,j,0]/dx[i-1])  +
                                                                                                               1./(dy[j+1] + dy[j-1]) * (F1[i,j+1,0]/dy[j+1] + F1[i,j-1,0]/dy[j-1]) +
                                                                                                               1./(dz[1] + dz[1]) * (F1[i,j,1]/dz[1] + F1[i,j,1]/dz[1]))
                                                                                                          
                elif B[i,j,k] == 3:
                    F2[i,j,-1] = 1./( 1./(dx[i+1] * dx[i-1]) + 1./(dy[j+1] * dy[j-1]) + 1./(dz[-2] * dz[-2])) * ( 1./(dx[i+1] + dx[i-1]) * (F1[i+1,j,-1]/dx[i+1] + F1[i-1,j,-1]/dx[i-1])  +
                                                                                                          1./(dy[j+1] + dy[j-1]) * (F1[i,j+1,-1]/dy[j+1] + F1[i,j-1,-1]/dy[j-1]) +
                                                                                                          1./(dz[-2] + dz[-2]) * (F1[i,j,-2]/dz[-2] + F1[i,j,-2]/dz[-2]))
                                                                                                              
                elif B[i,j,k] == 4:
                    F2[i,0,k] = 1./( 1./(dx[i+1] * dx[i-1]) + 1./(dy[1] * dy[1]) + 1./(dz[k+1] * dz[k-1])) * (    1./(dx[i+1] + dx[i-1]) * (F1[i+1,0,k]/dx[i+1] + F1[i-1,0,k]/dx[i-1])  +
                                                                                                          1./(dy[1] + dy[1]) * (F1[i,1,k]/dy[1] + F1[i,1,k]/dy[1]) +
                                                                                                          1./(dz[k+1] + dz[k-1]) * (F1[i,0,k+1]/dz[k+1] + F1[i,0,k-1]/dz[k-1]))
                                                                                                          
                elif B[i,j,k] == 5:
                    F2[i,-1,k] = 1./( 1./(dx[i+1] * dx[i-1]) + 1./(dy[-2] * dy[-2]) + 1./(dz[k+1] * dz[k-1])) * ( 1./(dx[i+1] + dx[i-1]) * (F1[i+1,-1,k]/dx[i+1] + F1[i-1,-1,k]/dx[i-1])  +
                                                                                                          1./(dy[-2] + dy[-2]) * (F1[i,-2,k]/dy[-2] + F1[i,-2,k]/dy[-2]) +
                                                                                                          1./(dz[k+1] + dz[k-1]) * (F1[i,-1,k+1]/dz[k+1] + F1[i,-1,k-1]/dz[k-1]))

                elif B[i,j,k] == 6:
                    F2[i,0,0] =   1./( 1./(dx[i+1] * dx[i-1]) + 1./(dy[1] * dy[1]) + 1./(dz[1] * dz[1])) *   (            1./(dx[i+1] + dx[i-1]) * (F1[i+1,0,0]/dx[i+1] + F1[i-1,0,0]/dx[i-1]) +
                                                                                                              1./(dy[1] + dy[1]) * (F1[i,1,0]/dy[1] + F1[i,1,0]/dy[1]) +
                                                                                                              1./(dz[1] + dz[1]) * (F1[i,0,1]/dz[1] + F1[i,0,1]/dz[1]))
                                                                                                              
                elif B[i,j,k] == 7:
                    F2[i, 0, -1] = 1./( 1./(dx[i+1] * dx[i-1]) + 1./(dy[1] * dy[1]) + 1./(dz[-2] * dz[-2])) * (           1./(dx[i+1] + dx[i-1]) * (F1[i+1,0,-1]/dx[i+1] + F1[i-1,0,-1]/dx[i-1])  +
                                                                                                              1./(dy[1] + dy[1]) * (F1[i,1,-1]/dy[1] + F1[i,1,-1]/dy[1]) +
                                                                                                              1./(dz[-2] + dz[-2]) * (F1[i,0,-2]/dz[-2] + F1[i,0,-2]/dz[-2]))
                                                                                                              
                                                                                                              
                elif B[i,j,k] == 8:
                    F2[i, -1, 0] =  1./( 1./(dx[i+1] * dx[i-1]) + 1./(dy[-2] * dy[-2]) + 1./(dz[1] * dz[1])) * (          1./(dx[i+1] + dx[i-1]) * (F1[i+1,-1,0]/dx[i+1] + F1[i-1,-1,0]/dx[i-1])  +
                                                                                                              1./(dy[-2] + dy[-2]) * (F1[i,-2,0]/dy[-2] + F1[i,-2,0]/dy[-2]) +
                                                                                                              1./(dz[1] + dz[1]) * (F1[i,-1,1]/dz[1] + F1[i,-1,1]/dz[1]))
                                                                                                              
                elif B[i,j,k] == 9:
                    F2[i,-1,-1] =   1./( 1./(dx[i+1] * dx[i-1]) + 1./(dy[-2] * dy[-2]) + 1./(dz[-2] * dz[-2])) * (        1./(dx[i+1] + dx[i-1]) * (F1[i+1,-1,-1]/dx[i+1] + F1[i-1,-1,-1]/dx[i-1])  +
                                                                                                              1./(dy[-2] + dy[-2]) * (F1[i,-2,-1]/dy[-2] + F1[i,-2,-1]/dy[-2]) +
                                                                                                              1./(dz[-2] + dz[-2]) * (F1[i,-2,-1]/dz[-2] + F1[i,-2,-1]/dz[-2]))
    
    """ edges reflected boundary conditions  """
    


                         
    """ corners reflected boundary conditions """

    return  F2


""" Allow to compilate the part of the code """
@nb.jit(nopython = True)
def fdm_drift_same_scale(F1, B):
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
def fdm_drift_same_scale_dielec(F1, B, D, **bound_kwargs):
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





""" Allow to compilate the part of the code """
@nb.jit(nopython = True)
def fdm_weighting_same_scale(F1, B):
    F2 = F1.copy()
    for i in range(1, F1.shape[0]-1): # Dirichlet conditions - let field[0] = a; field[-1] = b
        for j in range(F1.shape[1]):
            for k in range(F1.shape[2]):
            
                """ Equipotential case """
                if B[i,j,k] == 0:
                    continue
                    """ general case"""
                elif B[i,j,k] == 1:
                    F2[i,j,k] = (F1[i+1,j,k] + F1[i,j+1,k] + F1[i,j,k+1] + F1[i-1,j,k] + F1[i,j-1,k] + F1[i,j,k-1])/6
                    """ edges reflected boundary conditions  - Newman conditions dE/dn = 0"""
                elif B[i,j,k] == 2:
                    F2[i,j,0] = (F1[i+1,j,0] + F1[i,j+1,0] +   2 * F1[i,j, 1] + F1[i-1,j,0] + F1[i,j-1,0] )/6
                    
                elif B[i,j,k] == 3:
                    F2[i,j,-1] = (F1[i+1,j,-1] + F1[i,j+1,-1]  + 2 * F1[i,j,-2] + F1[i-1,j,-1] + F1[i,j-1,-1] )/6
                    
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
                    F2[i, -1, -1] =0# (F1[i+1, -1, -1] + 2 * F1[i, -2, -1] + 2 * F1[i, -1, -2] + F1[i-1, -1, -1])/6
    return  F2



@nb.jit(nopython = True)
def fdm_weighting_dielec(F1, B, D):
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
def fdm_weighting_boundary(F1, B, D, y0, y1):
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
                    F2[i,0,k] = y0[i,k] #(F1[i+1,0,k] + 2 * F1[i,1,k] +  F1[i,0, k+1 ] + F1[i-1,0,k] + F1[i,0,k-1] )/6
                    
                elif B[i,j,k] == 5:
                    F2[i,-1,k] = y1[i,k] #(F1[i+1,-1,k] + 2 * F1[i,-2,k] +  F1[i,-1, k+1] +F1[i-1,-1,k] + F1[i,-1,k-1] )/6
                    
                    """ corners reflected boundary conditions - Newman conditions dE/dn = 0"""
                elif B[i,j,k] == 6:
                    F2[i,0,0] = y0[i,0] #y0[i,0] #(F1[i+1, 0, 0] + 2 * F1[i, 1, 0] + 2 * F1[i, 0, 1] + F1[i-1, 0, 0])/6
                    
                elif B[i,j,k] == 7:
                    F2[i, 0, -1] = y0[i,-1] #y0[i,-1] #(F1[i+1, 0, -1] + 2 * F1[i, 1, -1] + 2 * F1[i, 0, -2] + F1[i-1, 0, -1])/6
                    
                elif B[i,j,k] == 8:
                    F2[i, -1, 0] = y1[i,0]# y1[i,0]# (F1[i+1, -1, 0] + 2 * F1[i, -2, 0] + 2 * F1[i, -1, 1] + F1[i-1, -1, 0])/6
                    
                elif B[i,j,k] == 9:
                    F2[i, -1, -1] = y1[i,-1] #y0[i,-1]# (F1[i+1, -1, -1] + 2 * F1[i, -2, -1] + 2 * F1[i, -1, -2] + F1[i-1, -1, -1])/6
    return F2




@nb.jit(nopython = True)
def fdm_weighting_2D(F1, B, D):
    F2 = F1.copy()
    for i in range(1, F1.shape[0] -1): # Dirichlet conditions - let field[0] = a; field[-1] = b
        for j in range(F1.shape[1]):
                """ Equipotential case """
                if B[i,j] == 0:
                    continue
                    """ general case"""
                elif B[i,j] == 3 or B[i,j] == 1:
                    F2[i,j] = (D[i+1,j]  * F1[i+1,j] + D[i,j+1] * F1[i,j+1]  + D[i-1,j] * F1[i-1,j] + D[i,j-1] * F1[i,j-1]) / (D[i+1,j] + D[i,j+1] + D[i-1,j] + D[i,j-1])
                    """ edges reflected boundary conditions  - Newman conditions dE/dn = 0"""
                elif B[i,j] == 4 or B[i,j] == 5 or B[i,j] == 6 or B[i,j] == 7 or B[i,j] == 8 or B[i,j] == 9:
                    F2[i,0] = 0
    return F2






@nb.jit(nopython = True)
def fdm_drift_dielec_for(F1, dx, dy, dz):
    F2 = F1.copy()
    """ general case """
    for i in range(1, F1.shape[0]-1):
        for j in range(1, F1.shape[1]-1):
            for k in range(1, F1.shape[2]-1):
                F2[i,j,k] = (dx[i+1,j,k] * F1[i+1,j,k] + dy[i,j+1,k] * F1[i,j+1,k] + dz[i,j,k+1] * F1[i,j,k+1] + dx[i-1,j,k] * F1[i-1,j,k] + dy[i,j-1,k] * F1[i,j-1,k] + dz[i,j,k-1] * F1[i,j,k-1])/(dx[i+1,j,k] + dy[i,j+1,k] + dz[i,j,k+1] + dx[i-1,j,k] + dy[i,j-1,k] + dz[i,j,k-1])
    """ edges reflected boundary conditions  """
    
    for i in range(1, F1.shape[0]-1):
        for j in range(1, F1.shape[1]-1):
            
            F2[i,j,0] = (dx[i+1,j,0] * F1[i+1,j,0] + dy[i,j+1,0] * F1[i,j+1,0] + 2 * dz[i,j,1] * F1[i,j, 1] + dx[i-1,j,0] * F1[i-1,j,0] + dy[i,j-1,0] * F1[i,j-1,0]) / (dx[i+1,j,0] + dy[i,j+1,0] + 2 * dz[i,j,1] + dx[i-1,j,0] + dy[i,j-1,0])
            
            
            F2[i,j,-1] = (dx[i+1,j,-1] * F1[i+1,j,-1] + dy[i,j+1,-1] * F1[i,j+1,-1] + 2 * dz[i,j,-2] * F1[i,j, -2] + dx[i-1,j,-1] * F1[i-1,j,-1] + dy[i,j-1,-1] * F1[i,j-1,-1]) / (dx[i+1,j,-1] + dy[i,j+1,-1] + 2 * dz[i,j,-2] + dx[i-1,j,-1] + dy[i,j-1,-1])
    
    
    for i in range(1, F1.shape[0]-1):
        for k in range(1, F1.shape[2]-1):
        
        
        
            F2[i,0,k] = (dx[i+1,0,k] * F1[i+1,0,k] + 2 * dy[i,1,k] * F1[i,1,k] +  dz[i,0, k+1] * F1[i,0, k+1 ] + dx[i-1,0,k] * F1[i-1,0,k] + dz[i,0, k-1] * F1[i,0,k-1] )/( dx[i+1,0,k] + 2 * dy[i,1,k] + dz[i,0, k+1] + dx[i-1,0,k] + dz[i,0, k-1])
            
            
            F2[i,-1,k] = (dx[i+1,-1,k] * F1[i+1,-1,k] + 2 * dy[i,-2,k] * F1[i,-2,k] +  dz[i,-1, k+1] * F1[i,-1, k+1] + dx[i-1,-1,k] * F1[i-1,-1,k] + dz[i,-1,k-1] * F1[i,-1,k-1] )/((dx[i+1,-1,k] + 2 * dy[i,-2,k] + dz[i,-1, k+1] + dx[i-1,-1,k] + dz[i,-1,k-1] ))

                         
    """ corners reflected boundary conditions """
    for i in range(1, F1.shape[0]-1):
        F2[i,0,0] = (dx[i+1,0,0] * F1[i+1, 0, 0] + 2 * dy[i,1,0] * F1[i, 1, 0] + 2 * dz[i,0,1] * F1[i, 0, 1] + dz[i-1,0,0] * F1[i-1, 0, 0])/(dx[i+1,0,0] + 2 * dy[i,1,0] + 2 * dz[i,0,1] + dz[i-1,0,0] )
        
        F2[i, 0, -1] = (dx[i+1,0,0] * F1[i+1, 0, -1] + 2 * dy[i,1,-1] * F1[i, 1, -1] + 2 * dz[i,0,-2] * F1[i, 0, -2] + dx[i-1, 0, -1] * F1[i-1, 0, -1])/(dx[i+1,0,0] + 2 * dy[i,1,-1] + 2 * dz[i,0,-2] + dx[i-1,0,-1])
        
        F2[i, -1, 0] = (dx[i+1,-1,0] * F1[i+1, -1, 0] + 2 * dy[i,-2,0] * F1[i, -2, 0] + 2 * dz[i,-1,1] * F1[i, -1, 1] + dx[i-1,-1,0] * F1[i-1, -1, 0])/(dx[i+1,-1,0] + 2 * (dy[i,-2,0] + dz[i,-1,1]) + dx[i-1,-1,0])
        
        F2[i, -1, -1] = (dx[i+1,-1,-1] * F1[i+1, -1, -1] + 2 * dy[i,-2,-1] * F1[i, -2, -1] + 2 * dz[i,-1,-2] * F1[i, -1, -2] + dx[i-1,-1,-1] * F1[i-1, -1, -1])/(dx[i+1,-1,-1] +2 * (dy[i,-2,-1] + dz[i,-1,-2]) + dx[i-1,-1,-1] )
    return  F2

""" Finite element with dielectric for drift calculation only """











""" Allow to compilate the part of the code """
@nb.jit(nopython = True)
def fdm_drift_density(F1, B, D, rho):
    F2 = F1.copy()
    for i in range(1, F1.shape[0]-1): # Dirichlet conditions - let field[0] = a; field[-1] = b
        for j in range(F1.shape[1]):
            for k in range(F1.shape[2]):
                """ equipotential """
                if B[i,j,k] == 0:
                    continue
                    """ general case"""
                elif B[i,j,k] == 1:
                    F2[i,j,k] = (D[i+1,j,k]  * F1[i+1,j,k] + D[i,j+1,k] * F1[i,j+1,k] + D[i,j,k+1] * F1[i,j,k+1] + D[i-1,j,k] * F1[i-1,j,k] + D[i,j-1,k] * F1[i,j-1,k] + D[i,j,k-1] * F1[i,j,k-1]) / (D[i+1,j,k] + D[i,j+1,k] + D[i,j,k+1] + D[i-1,j,k] + D[i,j-1,k] + D[i,j,k-1]) - rho[i,j,k]/(6 * 8.854e-12)
                    """ edges reflected boundary conditions  - Newman conditions dE/dn = 0"""
                elif B[i,j,k] == 2:
                    F2[i,j,0] = (D[i+1,j,0] * F1[i+1,j,0] + D[i-1,j,0] * F1[i-1,j,0] + D[i,j+1,0] * F1[i,j+1,0] + D[i,j-1,0] * F1[i,j-1,0] + 2 * D[i,j, 1] * F1[i,j, 1]) / (D[i+1,j,0] + D[i,j+1,0] + 2 * D[i,j, 1] + D[i-1,j,0] + D[i,j-1,0]) - rho[i,j,k]/(6 * 8.854e-12)
                    
                elif B[i,j,k] == 3:
                    F2[i,j,-1] = (D[i+1,j,-1] * F1[i+1,j,-1] + D[i,j+1,-1] * F1[i,j+1,-1] + 2 * D[i,j,-2] * F1[i,j,-2] + D[i-1,j,-1] * F1[i-1,j,-1] + D[i,j-1,-1] * F1[i,j-1,-1]) / (D[i+1,j,-1] + D[i,j+1,-1] + 2 * D[i,j,-2] + D[i-1,j,-1] + D[i,j-1,-1])  - rho[i,j,k]/(6 * 8.854e-12)
                    
                elif B[i,j,k] == 4:
                    F2[i,0,k] = (D[i+1,0,k] * F1[i+1,0,k] + D[i-1,0,k] * F1[i-1,0,k] + D[i,1,k] * F1[i,1,k] + D[i,-2,-k-1] * F1[i,-2,-k-1] +  D[i,0,k+1] * F1[i,0,k+1] + D[i,0,k-1] * F1[i,0,k-1] ) / (D[i+1,0,k] + D[i,1,k] + D[i,-2,-k-1] + D[i, 0, k+1] + D[i-1,0,k]  + D[i,0,k-1]) - rho[i,j,k]/(6 * 8.854e-12)
                    
                elif B[i,j,k] == 5:
                    F2[i,-1,k] = (D[i+1,-1,k] * F1[i+1,-1,k] + D[i-1,-1,k] * F1[i-1,-1,k] + D[i,-2,k] * F1[i,-2,k] + D[i,1,-k-1] * F1[i,1,-k-1] + D[i,-1, k+1] * F1[i,-1, k+1]  + D[i,-1,k-1] * F1[i,-1,k-1] ) / (D[i+1,-1,k] + D[i,-2,k] + D[i,1,-k-1] + D[i,-1, k+1] + D[i-1,-1,k] + D[i,-1,k-1]) - rho[i,j,k]/(6 * 8.854e-12)
                    
                    """ corners reflected boundary conditions - Newman conditions dE/dn = 0"""
                elif B[i,j,k] == 6:
                    F2[i,0,0] = (D[i+1, 0, 0] * F1[i+1, 0, 0] + D[i, 1, 0] * F1[i, 1, 0] + D[i,-1,-2] * F1[i,-1,-2] + D[i, 0, 1] * F1[i, 0, 1] + D[i,-2,-1] * F1[i,-2,-1] + D[i-1, 0, 0] * F1[i-1, 0, 0])/ (D[i+1, 0, 0] + D[i, 1, 0] + D[i,-1,-2] + D[i, 0, 1] + D[i,-2,-1] + D[i-1, 0, 0]) - rho[i,j,k]/(6 * 8.854e-12)
                    
                elif B[i,j,k] == 7:
                    F2[i, 0, -1] = (D[i+1, 0, -1] * F1[i+1, 0, -1] + D[i, 1, -1] * F1[i, 1, -1] + D[i,-2,1] * F1[i,-1,1] + D[i, 0, -2] * F1[i, 0, -2] + D[i,-2,0] * F1[i,-2,0] + D[i-1, -2,0] * F1[i-1, -2,0])/(D[i+1, 0, -1]+ D[i, 1, -1] + D[i,-1,1] + D[i, 0, -2] + D[i,-2,0] + D[i-1, -2,0]) - rho[i,j,k]/(6 * 8.854e-12)
                    
                elif B[i,j,k] == 8:
                    F2[i, -1, 0] = (D[i+1, -1, 0] * F1[i+1, -1, 0] + D[i, -2, 0] * F1[i, -2, 0] + D[i,0,-2]  * F1[i,0,-2] + D[i, -1, 1] * F1[i, -1, 1] + D[i, 1,-1] * F1[i, 1,-1] + D[i-1, -1, 0] * F1[i-1, -1, 0])/(D[i+1, -1, 0] + D[i, -2, 0] + D[i,0,-2]  + D[i, -1, 1] + D[i, 1,-1] + D[i-1, -1, 0]) - rho[i,j,k]/(6 * 8.854e-12)
                    
                elif B[i,j,k] == 9:
                    F2[i, -1, -1] = (D[i+1, -1, -1] * F1[i+1, -1, -1] + D[i, -2, -1] * F1[i, -2, -1] + D[i,0,1] * F1[i,0,1] + D[i,-1,-2] * F1[i, -1, -2] + D[i,1,0] * F1[i,1,0] + D[i-1, -1, -1] * F1[i-1, -1, -1])/(D[i+1, -1, -1] + D[i, -2, -1] + D[i,0,1] + D[i, -1, -2] + D[i,1,0] + D[i-1, -1, -1]) - rho[i,j,k]/(6 * 8.854e-12)
    return  F2
