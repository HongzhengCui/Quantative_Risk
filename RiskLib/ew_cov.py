import numpy as np

def ewCovar(x, lam):
    m, n = x.shape
    w = np.empty(m)
    xm = np.mean(x, axis=0)
    x = x - xm
    for i in range(m):
        w[i] = (1 - lam) * lam**(m - i - 1)
    w = w / np.sum(w)

    return ((w[:, np.newaxis] * x).T @ x).values

def ew_correlation(x, lam):
    cout = ewCovar(x, lam)
    sd = 1 / np.sqrt(np.diag(cout))
    cout = np.diag(sd) @ cout @ np.diag(sd)
    return cout

def ew_cov_var_corr(x, lam_var, lam_corr):
    cout = ewCovar(x, lam_corr)
    sd1 = np.sqrt(np.diag(cout))

    cout = ewCovar(x, lam_var)
    sd = 1 / np.sqrt(np.diag(cout))

    cout = np.diag(sd1) @ np.diag(sd) @ cout @ np.diag(sd) @ np.diag(sd1)
    return cout