import numpy as np

def skip_missing_rows(data):
    cleaned_data = data[~np.isnan(data).any(axis=1)]    
    matrix = np.cov(cleaned_data, rowvar=False)    
    
    return matrix

def pairwise_cov(data):
    data = data.cov()
    return data

def pairwise_corr(data):
    data = data.corr()
    return data