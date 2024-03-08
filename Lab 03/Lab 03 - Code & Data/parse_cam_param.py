import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm

def parse_cam_param(cam_param_file):
    """ 
    Parse camera parameters from a text file.

    Parameters
    ----------
    cam_param_file : str
        Path to the camera parameter file
    Returns
    -------
    cam_param : dict
        Dictionary containing camera parameters
        
    """

    cam_param = {}
    with open(cam_param_file) as f:
        for line in f:
            if line[0] == '#':
                continue
            key, value = line.split(';')
            if key == 'height':
                cam_param['h'] = int(value)
            elif key == 'width':
                cam_param['w'] = int(value)
            elif key == 'c':
                cam_param['c'] = float(value)
            elif key == 'cx':
                cam_param['cx'] = float(value)
            elif key == 'cy':
                cam_param['cy'] = float(value)
            elif key == 'k1':
                cam_param['k1'] = float(value)
            elif key == 'k2':
                cam_param['k2'] = float(value)
            elif key == 'p1':
                cam_param['p1'] = float(value)
            elif key == 'p2':
                cam_param['p2'] = float(value)

    return cam_param