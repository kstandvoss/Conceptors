import numpy as np
import numpy.polynomial.polynomial as poly
import scipy as sp
import scipy.interpolate
import math
#from matplotlib import pyplot as plt

import pickle
import sys

sys.path.append('Classes')
from reservoir import Reservoir
speakerN = 9
channelsN = 12
samplesN = 30
trialsN = 1
N = 10

with open('prepdata.pickle', 'rb') as f:
	data = pickle.load(f)

trainData = data['train']
testData = data['test']

# speakerN x (Nodes x (Timesteps*samplesN))
stateData = []
# speakerN x (Channels x (Timesteps*noSamples))
inputData = []
for trial in range(trialsN):
	r = Reservoir(N = N, inputScaling=0.2, biasScaling=1, inputDim=12)
	for speakerSamples in trainData:
		state, inp = r.collectStates(speakerSamples, t_learn=4, t_washout=0, mode = 'batch')
		stateData.append(state)
		inputData.append(inp)

	#Ridge Regression with different regularizers (Tikhonov Alpha)
#	alphas = 2.0**np.arange(-10,0)
#	for candidateAlpha in alphas:
#		mse = 0		
		#n-fold crossvalidation
#		n = 5
#		foldSize = samplesN/n  
#		for fold in range(n):
#			trainInds = [list(range(1,(fold - 1) * foldSize + 1)),list(range(fold * foldSize + 1, samplesN + 1))]
#			trainExamples = [stateData[(fold - 1) * (foldSize + 1):fold * foldSize + 1)]]

	samplesPerSpeaker = []
	for speakerSamples in trainData:
		speakerInput = np.array(speakerSamples)
		speakerInput = speakerInput.reshape((120,12))
		samplesPerSpeaker.append(speakerInput)

	r.loadingAndConceptors(patterns=samplesPerSpeaker, t_learn=120, t_washout=0, mode = 'batch')
	print(r.conceptors[1].shape)
		


