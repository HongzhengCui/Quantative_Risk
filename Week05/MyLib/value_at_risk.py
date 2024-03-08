import numpy as np
import pandas as pd
from scipy.stats import norm, t
from statsmodels.tsa.ar_model import AutoReg
from scipy.integrate import quad

def var_normal(data, alpha=0.05):
    mean = data.mean()
    std = data.std()
    var = norm.ppf(alpha, mean, std)
    return -var, mean - var

def var_normal_ew(data, alpha=0.05, lambda_=0.94):
    mean = data.mean()
    ew_var = lambda_ * data.var() + (1 - lambda_) * data.std() ** 2
    return norm.ppf(alpha, mean, np.sqrt(ew_var))

def var_t_dist(data, alpha=0.05):
    params = t.fit(data)
    var = t.ppf(alpha, *params)
    mean = data.mean()
    var_diff = mean - var
    return -var, var_diff.item()

def var_ar(data, alpha=0.05, lags=1):
    model = AutoReg(data, lags).fit()
    forecast = model.predict(start=len(data), end=len(data), dynamic=False)
    resid = model.resid
    std = resid.std()
    return norm.ppf(alpha, forecast, std)

def var_historic(data, alpha=0.05):
    var = np.percentile(data, alpha * 100)
    var_diff = data.mean() - var
    return var, var_diff

def var_simulation(data, alpha=0.05, iterations=10000):
    returns = data.iloc[:, 0]
    simulated_returns = np.random.choice(returns, size=iterations)
    
    var = np.percentile(simulated_returns, (alpha * 100))
    var_diff = var - returns.mean()
    return abs(var), abs(var_diff)

def ES_normal(data, alpha=0.05):
    mean = data.mean()
    std_dev = data.std()
    es = mean - std_dev * norm.pdf(norm.ppf(alpha)) / alpha
    return abs(es), abs(std_dev * norm.pdf(norm.ppf(alpha)) / alpha)

def ES_t(data, alpha=0.05):
    df, loc, scale = t.fit(data)
    var = t.ppf(alpha, df, loc, scale)
    def integrand(x):
        return x * t.pdf(x, df, loc, scale)
    
    es_integral, _ = quad(integrand, np.NINF, var)
    es = es_integral / alpha
    return abs(es), abs(es-data.mean())

def ES_simulation(data, alpha=0.05, simulations=10000):
    returns = data.iloc[:, 0]
    simulated_returns = np.random.choice(returns, size=simulations, replace=True)
    sorted_returns = np.sort(simulated_returns)
    var_position = int(alpha * simulations)
    es = np.mean(sorted_returns[:var_position])
    return abs(es), abs(es-data.mean())