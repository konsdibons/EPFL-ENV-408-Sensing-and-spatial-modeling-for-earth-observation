# -*- coding: utf-8 -*-

import numpy as np


def estimatePoseDLT(Q):
    """
    Estimate PI_prime, which encodes the transformation 
    that maps points from the world frame to the camera frame

    Parameters
    ----------
    Q : np.array(2n x 12), Q matrix for the DLT algorithm

    Returns
    -------
    PI_prime : np.array(3 x 4), projection matrix
    """

    # TODO : step 2.1 to 2.4
   
    return 

def buildQ(p, P):
    '''
    Build the Q matrix for the DLT algorithm.

    Parameters
    ----------
    p : np.array(n x 2), 2D coordinates of the points in the image plane
    P : np.array(n x 3), 3D coordinates of the points in the world frame

    Returns
    -------
    Q : np.array(2n x 12), Q matrix for the DLT algorithm
    '''
    N_pts = p.shape[0]
    Q = np.zeros((2*N_pts,12))

    #Express P in homogeneous coordinates, i.e. [x y z 1]
    P_hom = np.hstack((P, np.ones((N_pts,1))))

    # TODO : Fill in Q matrix

    return
