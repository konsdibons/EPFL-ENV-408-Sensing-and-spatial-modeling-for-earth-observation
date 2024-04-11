 # %%

# TODO: uncomment line below if you want to use autoreload with interactive python
# (reload your imported function after modifications) 
%load_ext autoreload


import os, sys
import numpy as np

# TODO:lines below takes care of importing the functions you have to complete
# If import do not work, uncomment line 14 and replace PATH with your absolute path to main.py
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
#sys.path.append('PATH')

from parse_cam_param import parse_cam_param

from coordinates import *
from DLT import *
from plot import *
from reprojectPoints import reprojectPoints


# %%-----------------------------------------------------#
#                   Data Preparation                     #
#--------------------------------------------------------# 
img_id = 1092311568

#TODO: Define here the root directory for input files.
data_path = 'data/'

gcps_path = data_path + 'gcps_local.txt'
cam_param_path = data_path + 'cam_param.txt'
xy_path = data_path + 'id_xy_corrected.txt'

xy = np.loadtxt(xy_path, delimiter=',', skiprows=1)
gcps = np.loadtxt(gcps_path,delimiter=',',skiprows=1)
cam_p = parse_cam_param(cam_param_path)
# %%-----------------------------------------------------#
#              Task 4 :  Test points subset              #
#--------------------------------------------------------# 

#Modify this part only for the last question of the assignment :)

#set_id =  [140,142,143,147,150,151] #-> uncomment this to use minimum set of GCPs
#set_id = [143,144,146,147,149,150] #-> uncomment this to use degenerate set of GCPs   
set_id = gcps[:,0] #-> uncomment this to use all GCPs

xy = xy[ np.isin(xy[:,0], set_id), : ]
gcps = gcps[ np.isin(gcps[:,0], set_id), : ]


with np.printoptions(precision=3, suppress=True):
        print(f'Using {xy.shape[0]} points for the DLT algorithm')
        print(f'IDs: {xy[:,0]}')
        print(f'Sample xy = \n{xy[:3,1:]}\n XYZ =\n {gcps[:3,1:]}')

# TODO: uncomment the 3 lines below to emulate a faulty measurement of 10 pixel per axis on GCP 149
# faulty_gcp_id = 149 
# index = np.where(xy[:,0] == faulty_gcp_id)
# xy[index,:] += 10 / cam_p['c'] # 10 px error on both axis

# %%-----------------------------------------------------#
#           Task 1 :  Build DLT's Eq. System             #
#--------------------------------------------------------# 
p = xy[:,1:]
P = gcps[:,1:]

#TODO: Implement the buildQ function in DLT.py
Q = buildQ(p, P)
# %%-----------------------------------------------------#
#           Task 2 :  Solve DLT's Eq. System             #
#--------------------------------------------------------# 
#TODO: Implement the estimatePoseDLT function in DLT.py
PI_prime = estimatePoseDLT(Q)
with np.printoptions(precision=2, suppress=True):
        print(f'PI_prime = {PI_prime}',)

# %%-----------------------------------------------------#
#       Task 3 :  Reproject points on the image          #
#--------------------------------------------------------# 
#TODO: Implement the reprojectPoints function in reprojectPoints.py     
p_reprojected = reprojectPoints(P, PI_prime)

# Transform both points and reprojected points in top left coordinates to estimate errors in pixels
uv = perspective2topleft(p, cam_p)
uv_reprojected = perspective2topleft(p_reprojected, cam_p)
reprojection_error = uv - uv_reprojected

# Plot reprojection error per point
plot_reprojection_error(reprojection_error, xy[:,0], img_id)
plot_reprojection_error_norm(np.linalg.norm( (reprojection_error), axis=1), xy[:,0], img_id)

# Extract camera rotation and position from PI_prime and compare with the ground truth
R_hat = PI_prime[:3,:3]
t_hat = PI_prime[:3,3]

# Hard question: Why do we need to transpose R_hat and negate cam_rot@t_hat to get camera coordinates ?
cam_rot = R_hat.T
cam_pos = -cam_rot @ t_hat
np.set_printoptions(precision=2)

# P were shifted close to the origin to avoid DLT unstability.
# Here we just need to shift back toestimate the camera location in swiss coordinate system
T = [ 2569000.0, 1094000.0, 2000.0 ]
cam_pos = cam_pos + T

cam_pos_groundtruth = np.array([2569444.71, 1094733.44, 3881.67])
print( 'DLT estimated postion = ', cam_pos, 'm' )
print( 'Groundtruth camera position = ', cam_pos_groundtruth, 'm' )
print( 'Error = ', np.linalg.norm(cam_pos - cam_pos_groundtruth), 'm' )

# %%
