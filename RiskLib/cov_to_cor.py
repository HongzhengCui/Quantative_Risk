import numpy as np

def covariance_to_correlation(cov_matrix):
    std_dev = np.sqrt(np.diag(cov_matrix))
    correlation_matrix = cov_matrix / np.outer(std_dev, std_dev)
    return correlation_matrix