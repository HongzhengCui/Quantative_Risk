import numpy as np
import pandas as pd

def near_psd(a, epsilon=0.0):
    n = a.shape[0]

    invSD = None
    out = a.copy()

    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / (vecs ** 2 @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out

def Frobenius(input):
    result = 0
    for i in range(len(input)):
        for j in range(len(input)):
            result += input[i][j]**2
    return result

def higham_nearestPSD(input):
    weight = np.identity(len(input))
        
    norml = np.inf
    Yk = input.copy()
    Delta_S = np.zeros_like(Yk)
    
    invSD = None
    if np.count_nonzero(np.diag(Yk) == 1.0) != input.shape[0]:
        invSD = np.diag(1 / np.sqrt(np.diag(Yk)))
        Yk = invSD @ Yk @ invSD
    
    Y0 = Yk.copy()

    for i in range(1000):
        Rk = Yk - Delta_S
        # PS
        Xk = np.sqrt(weight)@ Rk @np.sqrt(weight)
        vals, vecs = np.linalg.eigh(Xk)
        vals = np.where(vals > 0, vals, 0)
        Xk = np.sqrt(weight)@ vecs @ np.diagflat(vals) @ vecs.T @ np.sqrt(weight)
        Delta_S = Xk - Rk
        #PU
        Yk = Xk.copy()
        np.fill_diagonal(Yk, 1)
        norm = Frobenius(Yk-Y0)
        
        min_val = np.real(np.linalg.eigvals(Yk)).min()
        if abs(norm - norml) < 1e-8 and min_val > -1e-9:
            break
        else:
            norml = norm
    
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        Yk = invSD @ Yk @ invSD
    return Yk
