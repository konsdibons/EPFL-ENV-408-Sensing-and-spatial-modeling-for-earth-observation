# %%
import os
import sys
import numpy as np

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import *
from epip2err import epip2err
# %%-----------------------------------------------------#
#                      Data setup                        #
#--------------------------------------------------------#
# Camera parameters
cam_p = parse_cam_param('raw_data/cam_param.txt')
# Load correspondences
p1 = np.loadtxt('raw_data/xy_corrected_left.txt', delimiter=';',skiprows=2) 
p2 = np.loadtxt('raw_data/xy_corrected_right.txt', delimiter=';',skiprows=2)
# Express in homogenous coordinates
p1 = np.hstack((p1, np.ones((p1.shape[0], 1)))).T
p2 = np.hstack((p2, np.ones((p2.shape[0], 1)))).T

# %%-----------------------------------------------------#
#   Task 1 : Implement triangulation function & test     #
#--------------------------------------------------------#
def triangulation(p1, p2, Rt_1, Rt_2):
    """ Linear Triangulation, see lecture 06 for reminder
     Input:
      - p1, p2 np.ndarray(3, N): homogeneous coordinates in image 1 and 2
      - Rt_1. Rt_2 np.ndarray(3, 4): projection matrix corresponding to camera 1 and 2

     Returns:
      - P np.ndarray(4, N): homogeneous coordinates of 3-D points
    """
    P = np.empty((4, p1.shape[1]))
    # Build matrix of linear homogeneous equations per point

    for i in range(p1.shape[1]):
        A1 = np.dot(skewMatrix(p1[:, i]), Rt_1)
        A2 = np.dot(skewMatrix(p2[:, i]), Rt_2)
        A = np.r_[A1, A2]

        # Solve the system, see lecture 06 for reminder
        _, _, VT = np.linalg.svd(A)
        P[:, i] = VT[-1, :]

    # Scale back to homogeneous coordinates (divide by last coordinates) and return
    P /= P[3]
    return P 

# %%---------------------Test task 1------------------------#
P_test = np.random.rand(4,10)
P_test[3, :] = 1

Rt_1_test = np.array([[1, 0, 0,0],
                      [0, 1, 0,0],
                      [0, 0, 1,0]])
Rt_2_test = np.array([[-1, 0, 0,-1],
                      [ 0,-1, 0, 0],
                      [ 0, 0, 1, 0]])

p1_test = Rt_1_test @ P_test
p2_test = Rt_2_test @ P_test

P_est = triangulation(p1_test, p2_test, Rt_1_test, Rt_2_test)
np.testing.assert_allclose(P_est, P_test,
                            atol=1e-9,
                            err_msg="Test 1 : Triangulation -> FAILED")
print("Test 1 : Triangulation -> PASSED")
print(f"Mean triangulation error : {np.mean(np.abs(P_est - P_test))} below threshold (1e-9)")

# %%-----------------------------------------------------#
#        Task 2 : Estimate Essential Matrix E            #
#--------------------------------------------------------#
def eightpoints(p1, p2):
    """
    Estimate the essential matrix E from corresponding points using the 8-point algorithm.

    Parameters
    ----------
    p1,p2 : np.array(3xN), points in image 1, image 2

    Returns
    -------
    E : np.array(3x3), essential matrix
    """
    Q = np.zeros((p1.shape[1],9))
    # TODO : Compute the matrix Q using kronecker products
    for i in range(p1.shape[1]):
        Q[i,:] = np.kron(p1[:,i], p2[:,i]).T

    # TODO : Perform SVD(Q) & extract E from V last column
    _, _, VT_Q = np.linalg.svd(Q)          # np.linalg.svd outputs V.H ~ V.T, not V 
    E = VT_Q[-1,:].reshape(3,3).T

    # TODO : Enforce the rank 2 constraint on E: average lambda 1 and 2 and set lambda 3=0
    U, s, VT = np.linalg.svd(E)
    s[0] = s[1] = np.mean(s[:2])
    s[2] = 0
    E = U @ np.diag(s) @ VT

    return E
# %%
E = eightpoints(p1, p2)
#%%---------------------Test task 2------------------------#
np.testing.assert_allclose(epip2err(E, p1, p2),0,
                            atol=1e-2,
                            err_msg="Test 2: Essential Martix Estimation -> FAILED")
print("Test 2: Essential Martix Estimation -> PASSED")
print(f"Reprojection error : {epip2err(E, p1, p2)} below threshold (1e-2)")

# %%-----------------------------------------------------#
#       Task 3 : Find R | t from Essential matrix        #
#--------------------------------------------------------#
def decomposeEssential(E):
    """ 
    Decompose essential matrix into R1, R2, t1, t2

    Parameters
    ----------
    E : np.array(3x3), essential matrix

    Returns
    -------
    R1, R2 : np.array(3x3), rotation matrix 1 and 2
    t1, t2 : np.array(3x1), translation vector 1 and 2

    """
    # TODO : Perform SVD(E), extract t vectors from last column of U and build rotation matrices
    U, _, VT = np.linalg.svd(E)

    # Extract translation : last column of U
    t = U[:, -1]
    t1 = t / np.linalg.norm(t)
    t2 = -t1

    # Estimate rotations
    W = np.array([ [0, -1,  0],
                    [1,  0,  0],
                    [0,  0,  1]])
    # TODO : Build rotation matrices R1 and R2 and check the determinant
    R1 = np.dot(U, np.dot(W, VT))
    R2 = np.dot(U, np.dot(W.T, VT))

    # TODO : if ||t|| != 1, normalize t to unit norm
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    return R1, R2, t1, t2 

def checkRelativePose(Rs, ts,p1,p2):
    """ 
    Find correct pose (R,t) with the most points in front of both cameras.

    Parameters
    ----------
    Rt : [np.array(3x3), np.array(3x3)], list of rotation matrices in list
    ts : [np.array(3xN), np.array(3xN)], list of translation vectors

    Returns
    -------
    R_out, t_out : np.array(3x3) and np.array(3x1), valid rotation matrix and translation vector
    """
   
    Rt_1 = np.eye(3,4)  # Projection matrix of camera 1

    front_best = 0
    # TODO : Loop over all possible combinations of R and t to estimate Rt_2 and triangulate points

    for R in Rs:  
        for t in ts:
            #estimate transformation from camera 1 to camera 2
            Rt_2 = np.zeros((3, 4))
            Rt_2[:, :3] = R
            Rt_2[:, 3] = t

            #triangulate in camera 1 coordinate system
            P1 = triangulation(p1, p2, Rt_1, Rt_2)

            #reproject in camera 2 coordinates system
            P2 = np.dot(Rt_2, P1)
            
            #Estimate nb points in front of cameras
            front = np.sum(P1[2, :] > 0) + np.sum(P2[2, :] > 0)
                              
            if (front > front_best):
                front_best = front
                R_out, t_out = R, t
                
    return R_out, t_out

# %%
R1, R2, t1, t2 = decomposeEssential(E)

R, t = checkRelativePose([R1,R2], [t1, t2], p1, p2)

Rt1 = np.c_[np.eye(3), np.zeros((3, 1))]
Rt2 = np.c_[R, t]

# Triangulate your points using the chosen R|t matrices and plot the 3D scene
P = triangulation(p1, p2, Rt1, Rt2)

#%%---------------------Test task 3------------------------#

cam__dist = 211.92 # Metric distance between the two cameras to scale the plot
P_metric = cam__dist * P[:3,:]
t_metric = cam__dist * t
P_check = np.loadtxt('raw_data/P_check.txt', delimiter=';',skiprows=2).T

rmse_3d = np.sqrt(np.sum((P_metric - P_check)**2)/P_check.shape[1])
np.testing.assert_array_less(np.abs(rmse_3d),10,
                            err_msg="Test 3: Complete Pipeline -> FAILED")
print("Test 3: Complete Pipeline -> PASSED")
print(f"Reprojection error : {rmse_3d:.3f} [m] below threshold (10 m)")

plot_pipeline(P[:3,:], t, R,units='relative')
plot_pipeline(P_metric, t_metric, R, P_check=P_check, units='meters')
# %%
