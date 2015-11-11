import numpy as np
import numpy.polynomial.polynomial as poly
import scipy as sp
import scipy.interpolate
import math
from matplotlib import pyplot as plt

import pickle
import sys

sys.path.append('Classes')
from reservoir import Reservoir


speakerN = 9
channelsN = 12
trialsN = 1
N = 10

with open('prepdata.pickle', 'rb') as f:
	data = pickle.load(f)

trainData = data['train']
testData = data['test']

stateData = []
inputData = []
for trial in range(trialsN):
	r = Reservoir(N = N, inputScaling=0.2, biasScaling=1, inputDim=12)
	for speakerSamples in trainData:
		state, inp = r.collectStates(speakerSamples, t_learn=4, t_washout=0, mode = 'batch')
		stateData.append(state)
		inputData.append(inp)


		


