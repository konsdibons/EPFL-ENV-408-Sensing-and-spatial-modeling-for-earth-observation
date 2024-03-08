# -*- coding: utf-8 -*-

import numpy as np

def reprojectPoints(P, PI_prime):
    '''
    Reproject 3D points given a projection matrix
    
    Parameters
    ----------
    P : np.array(n x 3), coordinates of the 3d points in the world frame
    PI_prime : np.array(3 x 4), projection matrix
    
    Returns
    -------
    np.array(n x 2), coordinates of the reprojected 2d points
    '''
    #Express P in homogeneous coordinates, i.e. [x y z 1], by adding a column of ones
    P_hom = np.hstack((P, np.ones((P.shape[0],1))))

    # TODO : reproject points
    P_c = None

    # Here we divide the homogeneous coordinates by the third element to account for the depth.
    # TODO: Important to understand why
    P_c_homogeneous = P_c[:,:2]/P_c[:,2].reshape(-1,1)

    return P_c_homogeneous