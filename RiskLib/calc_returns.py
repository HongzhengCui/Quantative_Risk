import numpy as np
import pandas as pd

def return_calculate(prices, method="DISCRETE", date_column="Date"):
    if date_column not in prices.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame: {prices.columns}")

    vars = [col for col in prices.columns if col != date_column]
    n_vars = len(vars)
    p = prices[vars].values
    n, m = p.shape

    p2 = np.zeros((n-1, m))
    for i in range(n-1):
        for j in range(m):
            p2[i, j] = p[i+1, j] / p[i, j]

    if method.upper() == "DISCRETE":
        p2 -= 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in ('LOG','DISCRETE')")

    out = pd.DataFrame(p2, columns=vars)
    out.insert(0, date_column, prices[date_column].values[1:])

    return out