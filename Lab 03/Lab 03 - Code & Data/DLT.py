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
    # 2.1 Perform the SVD decomposition of Q to obtain Π˜s and reshape Π˜s into Π˜ of shape (3x4)
    # Hint numpy.linalg.svd()3 will perform the svd with S values sorted by descending order. np.reshape is usefull to recover the matrix form
    U, S, Vt = np.linalg.svd(Q)
    PI_tilde = Vt[-1, :].reshape(3, 4)
    # 2.2 Enforce det(R) = 1 property: Implement an if condition that multiply Π˜ by -1 if tz = Π˜_34 <0
    if PI_tilde[2, 3] < 0:
        PI_tilde = -PI_tilde

    # 2.3 Extract rotation matrix R: Perform SVD(R˜) = UΣV^T. You can then estimate Rˆ = UIV^T = UV^T which is equivalent to forcing R˜ eigenvalue to one.

    R_tilde = PI_tilde[:, :3]
    U, S, Vt = np.linalg.svd(R_tilde)
    R_hat = U @ Vt


    # 2.4 Recover the scale µ: The scale is defined by µ =||Rˆ||/||R˜|| , where || · || is any matrix norm, such as the Frobenius norm.
    # Hint numpy.linalg.norm() using Frobenius norm is available to you for this task.
    scale = np.linalg.norm(R_hat) / np.linalg.norm(R_tilde)

    # Output
    # Πˆ = [Rˆ|tˆ], 3x4 matrix. The rotation matrix and translation vector that projects 3D points into the image plane
    PI_prime = np.zeros((3, 4))
    PI_prime[:, :3] = R_hat
    PI_prime[:, 3] = PI_tilde[:, 3] * scale

    # Check that the resulting R (within PI_prime) is a valid rotation matrix (i.e. det(R) = 1 and RT R = I).
    # Hint .T (matrice transpose operator) and np.det() function might prove useful.

    print(np.linalg.det(PI_prime[:, :3]))

    assert np.isclose(
        np.linalg.det(PI_prime[:, :3]), 1
    ), "R is not a valid rotation matrix"
    assert np.allclose(
        PI_prime[:, :3].T @ PI_prime[:, :3], np.eye(3)
    ), "R is not a valid rotation matrix"

    return PI_prime


def buildQ(p, P):
    """
    Build the Q matrix for the DLT algorithm.

    Parameters
    ----------
    p : np.array(n x 2), 2D coordinates of the points in the image plane
    P : np.array(n x 3), 3D coordinates of the points in the world frame

    Returns
    -------
    Q : np.array(2n x 12), Q matrix for the DLT algorithm
    """
    N_pts = p.shape[0]
    Q = np.zeros((2 * N_pts, 12))

    # Express P in homogeneous coordinates, i.e. [x y z 1]
    P_hom = np.hstack((P, np.ones((N_pts, 1))))

    # TODO : Fill in Q matrix
    for i in range(N_pts):
        Q[2 * i, 0:4] = P_hom[i]
        Q[2 * i, 8:12] = -p[i, 0] * P_hom[i]
        Q[2 * i + 1, 4:8] = P_hom[i]
        Q[2 * i + 1, 8:12] = -p[i, 1] * P_hom[i]

    return Q
