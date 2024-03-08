# -*- coding: utf-8 -*-

import numpy as np

def topleft2perspective(uv, cam_p):
    '''
    Project from top left to perspective image coordinate 
    Equivalent to apply inv(K) to uv measurements
    
    Parameters
    ----------
    uv : np.array(Nx2), the array of measurements to project
        
    cam_p : dict, the camera calibration parameters
            
    Return
    ------
    xy : np.array(Nx2), the reprojected array 
    '''
    #Parse camera parameters
    h, w = cam_p['h'], cam_p['w']
    cx, cy, c = cam_p['cx'], cam_p['cy'], cam_p['c']

    #convert coordinates
    xy = np.empty(uv.shape)
    xy[:,0] = (uv[:,0] - (.5*(w-1) + cx))/c
    xy[:,1] = (uv[:,1] - (.5*(h-1) + cy))/c
    
    return xy

def perspective2topleft(xy, cam_p):
    '''
    Project from perspective to top left image coordinate 
    Equivalent to apply K to xy measurements
    
    Parameters
    ----------
    xy : np.array(Nx2), the array of measurements to project
        
    h, w : image height and width
         
    cx, cy, c : corresponding camera calibration parameters
            
    Return
    ------
    uv : np.array(Nx2), the reprojected array 
    '''
    #Parse camera parameters
    h, w = cam_p['h'], cam_p['w']
    cx, cy, c = cam_p['cx'], cam_p['cy'], cam_p['c']

    #convert coordinates
    uv = np.empty(xy.shape)
    uv[:,0] = xy[:,0]*c + (.5*(w-1) + cx)
    uv[:,1] = xy[:,1]*c + (.5*(h-1) + cy)
    
    return uv