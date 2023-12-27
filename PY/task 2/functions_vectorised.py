import numpy as np
from typing import Tuple


def sum_non_neg_diag(X: np.ndarray) -> int:
    tmp = np.diag(X)
    tmp = np.delete(tmp, np.where(tmp < 0))
    if(tmp.shape == (0,)):
        return -1
    return np.sum(tmp)



def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    tmp_x = np.sort(x, axis = None)
    tmp_y = np.sort(y, axis = None)
    return np.all(tmp_x == tmp_y)


def max_prod_mod_3(x: np.ndarray) -> int:
    if(x.shape == (0,)):
        return -1
    y = x.copy()
    y = np.delete(y, 0)
    z = x.copy()
    z = np.delete(z, -1)
    y = y * z
    y = np.delete(y, np.where(y % 3 != 0))
    if(y.shape == (0,)):
        return -1
    return np.amax(y)


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.dot(image, weights)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    if np.sum(x, axis=0)[1] != np.sum(y, axis=0)[1]:
        return -1
    newx = x.ravel()
    newx = np.repeat(newx[0::2], newx[1::2])
    newy = y.ravel()
    newy = np.repeat(newy[0::2], newy[1::2])
    return np.dot(newx, newy)


def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xdiag = np.diag(np.dot(x, x.T))
    ydiag = np.diag(np.dot(y, y.T))
    scal = np.dot(x, y.T)
    norma = (xdiag**(0.5)) * ((ydiag.T)**(0.5))[:, None]
    norma = norma.T
    mask = np.where(norma == 0)
    norma[norma == 0] = 1
    ans = scal/norma
    ans[mask] = 1
    return ans