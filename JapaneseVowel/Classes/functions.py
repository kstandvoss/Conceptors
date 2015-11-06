import numpy as np
import numpy.random
import scipy as sp
import scipy.sparse
import scipy.linalg
import scipy.interpolate
import scipy.sparse.linalg

# def sprandn(m, n, density):
# 	nnz = max(0, min(int(m*n*density), m*n))
# 	seq = np.random.permutation(m*n)[:nnz]
# 	data = np.random.randn(nnz)
# 	return scipy.sparse.csr_matrix((data, (seq/n,seq%n)), shape=(m,n)).toarray()

def sprandn(m, n, connectivity):
	m = sp.sparse.rand(m, n, density=connectivity, format='lil')
	nz = m.nonzero()
	m[nz] = np.random.randn(len(nz[0]))
	return m

def ridgeRegression(A, b, alpha):
	aI = alpha * np.eye(A.shape[1])
	return np.linalg.inv(A.T @ A + aI.T @ aI) @ A.T @ b

def NRMSE(output, target):
	output = np.squeeze(output)
	target = np.squeeze(target)
	error = target - output
	
	# ptp = range
	peakToPeak = np.ptp(target)
	rmse = np.sqrt(np.mean(error**2))
	nrmse = rmse / peakToPeak
	return nrmse

def interpolateAndShift(driver, recall, sampleRate = 20):
	plotRange = len(driver)
	recallLength = len(recall)
	xVals = np.linspace(0, plotRange-1, plotRange*sampleRate)

	fD = sp.interpolate.interp1d(range(plotRange), driver, kind='linear')
	fR = sp.interpolate.interp1d(range(recallLength), recall, kind='linear')

	driverIP = fD(xVals)
	recallIP = fR(np.linspace(0, recallLength-1, recallLength*sampleRate))

	dL = len(driverIP)
	rL = len(recallIP)

	phaseDifference = [np.linalg.norm(driverIP - recallIP[s:s+dL]) for s in range(rL - dL)]

	pos = np.argmin(phaseDifference)

	driverIPShifted = driverIP
	recallIPShifted = recallIP[pos:pos+dL]

	return xVals, driverIPShifted, recallIPShifted

