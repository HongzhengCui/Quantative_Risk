import numpy as np
from .non_psd_fix import higham_nearestPSD

def direct_simulation(cov_mat, draw_num=100000):
    chol_matrix = np.linalg.cholesky(cov_mat)
    simulated_data = chol_matrix @ np.random.normal(size=(cov_mat.shape[0], draw_num))
    return np.cov(simulated_data)

def pca_simulation(df, N=100000, mean=None, seed=1234, pctExp=0.99):
    # Error Checking
    m, n = df.shape
    if n != m:
        raise ValueError(f"Covariance Matrix is not square ({n},{m})")
    
    # Initialize the output
    out = np.zeros((N, n))
    
    # Set mean
    if mean is None:
        mean = np.zeros(n)
    else:
        if len(mean) != n:
            raise ValueError(f"Mean ({len(mean)}) is not the size of cov ({n},{n})")
    
    eigenvalues, eigenvectors = np.linalg.eig(df)
    
    # Get the indices that would sort eigenvalues in descending order
    indices = np.argsort(eigenvalues)[::-1]
    # Sort eigenvalues
    eigenvalues = eigenvalues[indices]
    # Sort eigenvectors according to the same order
    eigenvectors = eigenvectors[:, indices]
    
    tv = np.sum(eigenvalues)
    posv = np.where(eigenvalues >= 1e-8)[0]
    if pctExp <= 1:
        nval = 0
        pct = 0.0
        # How many factors needed
        for i in posv:
            pct += eigenvalues[i] / tv
            nval += 1
            if pct >= pctExp:
                break
    
     # If nval is less than the number of positive eigenvalues, truncate posv
    if nval < len(posv):
        posv = posv[:nval]
        
    # Filter eigenvalues based on posv
    eigenvalues = eigenvalues[posv]
    eigenvectors = eigenvectors[:, posv]
    
    B = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    
    np.random.seed(seed)
    rand_normals = np.random.normal(0.0, 1.0, size=(N, len(posv)))
    out = np.dot(rand_normals, B.T) + mean
    
    return out.T