import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, t
from scipy.optimize import minimize

def fit_normal_distribution(data):
    values = data.iloc[:, 0]
    mean = values.mean()
    std_sample = values.std(ddof=1)    
    return mean, std_sample

def fit_t_distribution(data):
    values = data.iloc[:, 0].dropna()
    nu, mu, sigma = t.fit(values)
    
    return mu, sigma, nu

def MLE_t(X, Y):
    X = sm.add_constant(X)
    def ll_t(params):
        df = params[0]
        s = params[1]
        b = params[2:]
        e = Y - np.dot(X, b)
        ll = np.sum(t.logpdf(e, df=df, loc=0, scale=s))
        return -ll
    beta = np.zeros(X.shape[1])
    s = np.std(Y - np.dot(X, beta))
    df = 1
    params = [df, s]
    for i in beta:
        params.append(i)
    bnds = ((0, None), (1e-9, None), (None, None), (None, None), (None, None), (None, None))
    res = minimize(ll_t, params, bounds=bnds, options={"disp": False})
    beta_mle = res.x[2:]
    return beta_mle

def fit_t_regression(data):
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    betas = MLE_t(X, Y)
    X = sm.add_constant(X)
    e = Y - np.dot(X, betas)
    df, loc, scale = t.fit(e)
    return loc, scale, df, betas[0], betas[1], betas[2], betas[3]