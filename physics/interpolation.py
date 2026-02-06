from numba import njit

#@njit
def trilinear(x, y, z, x0, y0, z0, e000, e100, e010, e001, e110, e101, e011, e111, step_x, step_yz):

    """ 3D-Linear interpolation is identical to 2 2D-linear interpolation combined with a 1D-linear interpolation """
    
    """ Smaller coordinate related """
    xr = ( x - x0 ) / step_x
    yr = ( y - y0 ) / step_yz
    zr = ( z - z0 ) / step_yz
    
    """ Linear interpolation along 2-axis """
    e00 = e000 * ( 1 - xr ) + e100 * xr
    e01 = e010 * ( 1 - xr ) + e110 * xr
    e10 = e001 * ( 1 - xr ) + e101 * xr
    e11 = e011 * ( 1 - xr ) + e111 * xr

    """ Linear interpolation along 1-axis based on the previous result """
    e0 = e00 * ( 1 - yr ) + e01 * yr
    e1 = e10 * ( 1 - yr ) + e11 * yr
    
    """ Linear interpolation along the last axis (0-axis) """
    e = e0 * ( 1 - zr ) + e1 * zr
    
    return e
    

#@njit
def linear_interp_field(x, y, z, vol, i, j, k, step_x, step_yz, *field):

    """ Define drift volume to interpolate """
    """
    vol_x = vol[0][i,j,k]
    vol_y = vol[1][i,j,k]
    vol_z = vol[2][i,j,k]
    """
    vol_x = vol[1,i,j,k]
    vol_y = vol[0,i,j,k]
    vol_z = vol[2,i,j,k]
    """  Field linear interpolation """
    if len(field) == 3:

        Ex, Ey, Ez = field
        Ez0 = trilinear(       x,y,z,
                               vol_x, vol_y, vol_z,
                               Ez[i,j,k],
                               Ez[i,j,k+1], Ez[i,j+1,k], Ez[i+1,j,k],
                               Ez[i,j+1,k+1], Ez[i+1,j,k+1], Ez[i+1,j+1,k],
                               Ez[i+1,j+1,k+1],
                               step_x, step_yz                                  )


        Ey0 = trilinear(       x,y,z,
                               vol_x, vol_y, vol_z,
                               Ey[i,j,k],
                               Ey[i,j,k+1], Ey[i,j+1,k], Ey[i+1,j,k],
                               Ey[i,j+1,k+1], Ey[i+1,j,k+1], Ey[i+1,j+1,k],
                               Ey[i+1,j+1,k+1],
                               step_x, step_yz                                  )


        Ex0 = trilinear(       x,y,z,
                               vol_x, vol_y, vol_z,
                               Ex[i,j,k],
                               Ex[i,j,k+1],Ex[i,j+1,k],Ex[i+1,j,k],
                               Ex[i,j+1,k+1],Ex[i+1,j,k+1],Ex[i+1,j+1,k],
                               Ex[i+1,j+1,k+1],
                               step_x, step_yz                                  )
        
        return -Ex0, -Ey0, -Ez0

    else :
        """ Potiential linear interpolation """
        phi = field[0]
        
        phi0=trilinear(        x,y,z,
                               vol_x, vol_y, vol_z,
                               phi[i,j,k],
                               phi[i,j,k+1],phi[i,j+1,k],phi[i+1,j,k],
                               phi[i,j+1,k+1],phi[i+1,j,k+1],phi[i+1,j+1,k],
                               phi[i+1,j+1,k+1],
                               step_x, step_yz                                  )
    return phi0


