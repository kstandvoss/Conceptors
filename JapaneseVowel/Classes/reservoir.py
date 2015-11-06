import functions

import numpy as np
import numpy.linalg
import numpy.random

import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

from matplotlib import pyplot as plt

class Reservoir:

	def __init__(	self,
					N = 100,
					connectivity = 0.1,
					inputDim = 1,
					outputDim = 1,
					spectralScaling = 1.5,
					inputScaling = 1.5,
					biasScaling = 0.2):

		# setup parameters
		self.N = N
		self.connectivity = connectivity
		self.inputDim = inputDim
		self.outputDim = outputDim
		self.spectralScaling = spectralScaling
		self.inputScaling = inputScaling
		self.biasScaling = biasScaling

		# setup other variables
		self.conceptors = None
		self.nrmse_load = None
		self.nrmse_out = None
		
		# setup weight matrix
		self.W_res = np.zeros((self.N, self.N))

		# fill weight matrix with values and calculate spectral tradius
		# spectral radius calculation can fail so gotta try until it works
		specRad = None
		while specRad is None:
			try:
				self.W_res = functions.sprandn(self.N, self.N, connectivity)
				specRad, _ = np.abs(sp.sparse.linalg.eigs(self.W_res, 1))
			except:
				pass

		# now we can get rid of sparse description (will have a dense matrix anyways after loading)
		# also apply spectral scaling
		self.W_res = self.W_res.toarray() * spectralScaling / np.squeeze(specRad)

		# setup other matrices
		self.W_in = self.inputScaling * np.random.randn(self.N, self.inputDim)
		self.W_bias = self.biasScaling * np.random.randn(self.N, 1)
		self.W_out = np.zeros((self.outputDim, self.N))



	def batchLoadingWithConceptorsPost(	self,
										patterns,
										t_washout = 100,
										t_learn = 1000,
										alpha_wout = 0.01,
										alpha_load = 0.0001,
										aperture = 10):

		stateColl = np.zeros((self.N, t_learn*len(patterns)))
		inputColl = np.zeros((self.inputDim, t_learn*len(patterns)))
		self.conceptors = []

		for i_pat, pattern in enumerate(patterns):

			x = np.zeros((self.N, 1))
			
			offset = i_pat*t_learn

			for t in range(t_washout + t_learn):

				u = pattern(t)
				if not isinstance(u, np.ndarray):
					u = np.array([[u]])

				x = np.tanh(self.W_res @ x + self.W_in @ u + self.W_bias)

				if t >= t_washout:
					stateColl[:, offset + t - t_washout] = x.T
					inputColl[:, offset + t - t_washout] = u.T

			thisPatternStateColl = stateColl[:, offset:offset + t_learn]
			R = (thisPatternStateColl @ thisPatternStateColl.T) / t_learn
			U,S,V = np.linalg.svd(R, full_matrices=True)
			S = np.diag(S)
			I = np.eye(self.N)
			S_new = S @ np.linalg.inv(S + (aperture**-2)*I)
			C = U @ S_new @ V
			self.conceptors.append(C)

		# output training
		self.W_out = functions.ridgeRegression(stateColl.T, inputColl.T, alpha_wout).T

		# nrmse for output training
		computedOutput = self.W_out @ stateColl
		self.nrmse_out = functions.NRMSE(computedOutput, inputColl)
		#print("NRMSE for output training: {}".format(self.nrmse_out))


		# loading
		oldStateColl = np.zeros((self.N, t_learn*len(patterns)))
		oldStateColl[:, 1:] = stateColl[:, 0:-1]
		bias_rep = np.tile(self.W_bias, (1, t_learn*len(patterns)))
		loadingTarget = np.arctanh(stateColl) - bias_rep
		self.W_res = functions.ridgeRegression(oldStateColl.T, loadingTarget.T, alpha_load).T

		# nrmse for loading
		computedTarget = self.W_res @ oldStateColl
		self.nrmse_load = np.mean(functions.NRMSE(computedTarget, loadingTarget))
		#print("average neuron NRMSE for loading process: {}".format(self.nrmse_load))




	def batchLoadingWithConceptorsGradient(	self,
											patterns,
											t_washout = 500,
											t_learn = 1000,
											alpha_wout = 0.01,
											alpha_load = 0.0001,
											aperture = 10,
											learning_rate = 0.01,
											gradient_cut = 2.0):

		stateColl = np.zeros((self.N, t_learn*len(patterns)))
		inputColl = np.zeros((self.inputDim, t_learn*len(patterns)))

		self.conceptors = []

		for i_pat, pattern in enumerate(patterns):

			x = np.zeros((self.N, 1))
			C = np.zeros((self.N, self.N))

			for t in range(t_washout + t_learn):

				u = pattern(t)
				if not isinstance(u, np.ndarray):
					u = np.array([[u]])

				x = np.tanh(self.W_res @ x + self.W_in @ u + self.W_bias)

				

				if t >= t_washout:
					offset = i_pat*t_learn
					stateColl[:, offset + t - t_washout] = x.T
					inputColl[:, offset + t - t_washout] = u.T

					grad = ((x - C @ x) @ x.T) - (aperture**-2) * C

					# perform gradient cutting, otherwise it gets too big
					norm = np.linalg.norm(grad)
					if norm > gradient_cut:
						grad = grad * gradient_cut/norm
						
					C = C + learning_rate * grad

			# conceptors
			self.conceptors.append(C)

		# output training
		self.W_out = functions.ridgeRegression(stateColl.T, inputColl.T, alpha_wout).T

		# nrmse for output training
		computedOutput = self.W_out @ stateColl
		self.nrmse_out = functions.NRMSE(computedOutput, inputColl)
		#print("NRMSE for output training: {}".format(self.nrmse_out))


		# loading
		oldStateColl = np.zeros((self.N, t_learn*len(patterns)))
		oldStateColl[:, 1:] = stateColl[:, 0:-1]
		bias_rep = np.tile(self.W_bias, (1, t_learn*len(patterns)))
		loadingTarget = np.arctanh(stateColl) - bias_rep
		self.W_res = functions.ridgeRegression(oldStateColl.T, loadingTarget.T, alpha_load).T

		# nrmse for loading
		computedTarget = self.W_res @ oldStateColl
		self.nrmse_load = np.mean(functions.NRMSE(computedTarget, loadingTarget))
		#print("average neuron NRMSE for loading process: {}".format(self.nrmse_load))



	def batchRecall(self,
					patternNumber,
					t_washout = 100,
					t_recall = 100):


		C = self.conceptors[patternNumber]

		x = np.random.randn(self.N, 1)

		output = np.zeros((self.outputDim, 1))

		for t in range(t_washout + t_recall):
			x = C @ np.tanh(self.W_res @ x + self.W_bias)

			if t >= t_washout:
				y = self.W_out @ x
				output = np.append(output, y)

		return output



	def singleRecallInit(self):
		self.x = np.random.randn(self.N, 1)

	def singleRecall(self, C):
		self.x = C @ np.tanh(self.W_res @ self.x + self.W_bias)
		output = self.W_out @ self.x
		return output




