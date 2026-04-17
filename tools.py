import numpy as np
from scipy.special import softmax

PRECISION = 1e-16

# === DISTRIBUTIONS ===

def marginal(pXY, axis=1):
    """:return pY (axis = 0) or pX (default, axis = 1)"""
    return pXY.sum(axis)

def joint(pY_X, pX):
    """:return  pXY """
    return pY_X * pX[:, None]

# === INFORMATIONAL MEASURES ===

def xlogx(v):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(v > PRECISION, v * np.log2(v), 0)
    
def xlogy(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x > PRECISION, x * np.log2(y), 0)

def H(p, axis=None):
    """ Entropy """
    return -xlogx(p).sum(axis=axis)


def MI(pXY):
    """ mutual information, I(X;Y) """
    return H(pXY.sum(axis=0)) + H(pXY.sum(axis=1)) - H(pXY)

def complexity(pX, pZ_X):
    """ I(X;Z) """
    return MI(joint(pZ_X, pX))

def accuracy(pX, pZ_X, pY_X):
    """ I(Z;Y) """
    pXZ = joint(pZ_X, pX)
    pZY = pXZ.T @ pY_X
    return MI(pZY)

def DKL(p, q, axis=None):
    """ KL divergences, D[p||q] """
    return (xlogx(p) - np.where(p > PRECISION, p * np.log2(q + PRECISION), 0)).sum(axis=axis)

def M_HAT(pM: np.ndarray, pU_M: np.ndarray, pW_M: np.ndarray):
    """
    :param pM: prior distribution over meanings
    :param pU_M: meaning similairities
    :param pW_M: encoder
    :return: the optimal Bayesian posterior-decoder  m̂ = p(u∣w)

    """
    pMW = pW_M * pM[:, None]
    pW = pMW.sum(axis=0)[:, None]
    pM_W = np.where(
        pW > PRECISION, 
        pMW.T / pW, 
        pM # avoid division by zero, fallback to prior
    ) 
    return pM_W @ pU_M

# === Reversed Deterministic Annealing ===

def BA_iterations(
        pM: np.ndarray, 
        pU_M: np.ndarray, 
        q_init: np.ndarray, 
        beta: float, 
        num_iter: int = 50, 
        temperature: float = 1):
    """
    pM : Distribution on M, of shape M.
    pU_M : Conditional distribution on U given M, of shape M x U.
    q_init : Initial conditional distribution on W given M, of shape M x W.
    beta : A non-negative scalar value.
    Output: 
    pW_M : Conditional distribution on W given M, of shape M x W.
    """

    pW_M = q_init

    # Blahut-Arimoto iteration to find the minimizing encoder p(w|m)
    for _ in range(num_iter):
        pMW = joint(pW_M, pM)
        pW = marginal(pMW, axis=0)
        pW = np.clip(pW, PRECISION, None)
        pW /= pW.sum()
        pU_W = M_HAT(pM, pU_M, pW_M)

        # casting on the same space to compute DKL
        cast_pU_M = pU_M[:, None, :] # shape M x 1 x U
        cast_pU_W = pU_W[None, :, :] # shape 1 x W x U
        dkl = DKL(cast_pU_M, cast_pU_W, axis=-1)
        dkl_nats = dkl * np.log(2)

        # update pW_M 
        pW_M = softmax(np.log(pW + PRECISION) - beta * dkl_nats, axis=-1)

    return pW_M