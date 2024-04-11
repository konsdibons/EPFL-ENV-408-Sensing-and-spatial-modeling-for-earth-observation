import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm
import copy

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
                cam_param['height'] = int(value)
            elif key == 'width':
                cam_param['width'] = int(value)
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

def scale_measurements(p_down,h_raw, h_downscaled, w_raw, w_downscaled):
    """ Scale measurements from lab 2 to the original image size

    Parameters
    ----------
    p : np.ndarray(N,4)
        Points to scale : u_L,v_L,u_R,v_R
    h0 : int
        Original image height
    h_down : int
        Downsampled image height
    w0 : int
        Original image width
    w_down : int
        Downsampled image width

    Returns
    -------
    p0 : np.ndarray(N,4)
        Scaled points
    """
    p0 = copy.deepcopy(p_down)

    ratio_h = h_raw/h_downscaled
    ratio_w = w_raw/w_downscaled

    p0[:,0] *= ratio_w
    p0[:,1] *= ratio_h
    p0[:,2] *= ratio_w
    p0[:,3] *= ratio_h

    return p0

def skewMatrix(x):
    """ Retun the skew matrix M corresponding to a 3 by 1 vector x such that M*y = cross(x,y))

    Parameters
    ----------
       - x np.ndarray(3,1) : vector

    Return
    ------
       - M np.ndarray(3,3) : skew matrix matrix
    """
    return np.array([[    0, -x[2],   x[1]], 
                     [ x[2],     0,  -x[0]],
                     [-x[1],  x[0],      0]])

def plot_pipeline(P, T2, R2, P_check=None, units='relative'):
    T2 = -T2  # -T2 because eq 3.19, x2 = R2*x1 + T2 thus x1 = R2.T * (x2 - T2)
    dist = np.linalg.norm(T2)

    fig = plt.figure()
    #size to large
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[0,:], P[1,:], P[2,:], c='r', marker='o',label='Triangulated 3D points')
    if P_check is not None:
        ax.scatter(P_check[0,:], P_check[1,:], P_check[2,:], c='#FFA500', marker='x',label='Check 3D points',alpha=0.5)

    ax.scatter(0, 0, 0, c='cyan', marker='o',label='Camera 1')
    ax.scatter(T2[0], T2[1], T2[2], c='magenta', marker='o',label='Camera 2')
    ax.legend()

    #Ensure equal proportion of axis w.o. using axis('equal')
    ax.set_xlim3d(-2.5*dist, 2.5*dist)
    ax.set_ylim3d(-2.5*dist, 2.5*dist)
    ax.set_zlim3d(0 , 5*dist)
    
    r2_x = R2.T @ np.array([1,0,0])
    r2_y = R2.T @ np.array([0,1,0])
    r2_z = R2.T @ np.array([0,0,1])
    ax.quiver(0, 0, 0, 1, 0, 0, length=dist/3, color='r')
    ax.quiver(0, 0, 0, 0, 1, 0, length=dist/3, color='g')
    ax.quiver(0, 0, 0, 0, 0, 1, length=dist/3, color='b')
    ax.quiver(T2[0], T2[1], T2[2], r2_x[0], r2_x[1], r2_x[2], length=dist/3, color='r')
    ax.quiver(T2[0], T2[1], T2[2], r2_y[0], r2_y[1], r2_y[2], length=dist/3, color='g')
    ax.quiver(T2[0], T2[1], T2[2], r2_z[0], r2_z[1], r2_z[2], length=dist/3, color='b')
    ax.plot_trisurf(P[0,:], P[1,:], P[2,:]+dist/200, edgecolor='w', linewidth =0.2, cmap=cm.jet, alpha=0.4)

    if units == 'meters':
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]]')
        ax.set_zlabel('Z [m]')

    else:
        ax.set_xlabel('X [-]')
        ax.set_ylabel('Y [-]')
        ax.set_zlabel('Z [-]')

    ax.set_title('3D reconstruction in camera 1 frame')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_zaxis()
    plt.show()

     

