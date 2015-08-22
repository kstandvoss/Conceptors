import numpy as np
import numpy.random
import scipy.sparse
import scipy.linalg

def sprandn(m, n, density):
    nnz = max(0, min(int(m*n*density), m*n))
    seq = np.random.permutation(m*n)[:nnz]
    data = np.random.randn(nnz)
    return scipy.sparse.csr_matrix((data, (seq/n,seq%n)), shape=(m,n)).todense()

def getSpecRad(M):
    N = M.shape[1]
    specRad, largestEigenvec = np.abs(scipy.linalg.eigh(M, eigvals=(N-1, N-1)))
    return specRad[0]

def ridgeRegression(A, b, alpha):
    aI = alpha * np.eye(A.shape[1])
    first = np.linalg.inv(np.dot(A.T, A) + np.dot(aI.T, aI))
    return np.dot(first , np.dot(A.T, b))

def NRMSE(output, target):
    error = target - output
    
    # ptp = range
    peakToPeak = np.ptp(target, axis=1)
    rmse = np.sqrt(np.mean(error**2, axis=1))
    nrmse = rmse / peakToPeak
    return nrmse