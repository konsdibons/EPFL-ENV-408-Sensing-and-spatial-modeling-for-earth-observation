import numpy as np

def epip2err(E, p1, p2):
    """
    Compute the reprojection error for a given essential matrix E and set of tie points.

    Parameters
    ----------
    E : np.array(3x3), essential matrix
    p1, p2 : np.array(3xN), points in image 1, 2

    Returns
    -------
    err : np.array(N), reprojection error
    """
    return np.sqrt(np.sum((np.sum(p2*(E@p1),axis=0))**2)/p1.shape[1])