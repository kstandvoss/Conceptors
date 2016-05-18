# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:15:55 2016

@author: asus
"""

import pickle
import os.path
file = os.path.abspath('C:/users/asus/Dropbox/Conceptors/Task1_Recognition/syllable_prepdata.pickle')
import numpy as np
import sys
sys.path.append('C:/users/asus/Documents/GitHub/Conceptors/Classes')
import reservoir_c as c
import functions as fct
import preprocessing as prep

#%%

""" import data """

path = os.path.abspath('D:/Data/Projects/StudyProject/syll')
data = pickle.load(open(file, 'rb'))
training_data = np.squeeze(data.get('train'))
test_data = np.squeeze(data.get('test'))

pickle.dump(data,open('arturaid.pkl','wb'))

#%%

""" Parameters """

sample_length = 30
N = 10
SR = 1.2
bias_scale = 1
gamma_pos = 25
gamma_neg = 27
conn = 1

syllableN = 9
trainSamplesNs = [30] * syllableN
testSamplesNs = [41, 25, 38, 34, 29, 24, 30, 40, 19]
channelsN = 12
smoothLength = 5

#%%

class syllableClassifier:
    
    def __init__(self, fname):   
        """ Class that performs supervised learning on syllable data in order to perform classification.
        
        :param fname: Complete path to folder which includes folders for each syllable which include folders for each sample which include wave data
        """
    
        self.folder = fname
    
    def prepData(self, n_syllables, n_train, n_test, mel_channels = 12, smoothLength = 5):
        """ Function that performs the following preprocessing steps on data in file:
        1. loading
        2. Zero Padding
        3. Extraction of Mel Frequency Cepstral Coefficients
        4. Extraction of shift and scale of training data
        5. Data normalization with shift and scale of training data
        6. Data smoothing
        
        :param file: complete path name for file to be loaded (string)
        :param n_syllables: number of syllables to include in preprocessing (scalar)
        :param n_train: number of training samples (scalar)
        :param n_test: number of test samples for each syllable (vector of length n_syllables)
        :param mel_channels: number of channels to include from the Mel freq spectrum
        :param smoothLength: Number of sampling points to reduce mel transformed data to
    
        :returns trainDataSmoothend: list of preprocessed training data
        :returns testDataSmoothend: list of preprocessed test data
        """
        
        """ Load Data """
        
        syllables = [files for files in os.listdir(file)]
        
        self.trainDataRaw = []
        self.testDataRaw = []
        self.skipped_syllables = []
        
        stepsize = len(syllables) // n_syllables
        ind = np.arange(0, stepsize * n_syllables, stepsize)
        
        for i in range(n_syllables):
            success = False
            while not success:
                try:
                    self.trainDataRaw.append(prep.load_data(file + '/' + syllables[ind[i]], n_train, 0))
                    self.testDataRaw.append(prep.load_data(file + '/' + syllables[ind[i]], n_test[i], trainSamplesNs[i]))
                    success = True
                except:
                    self.skipped_syllables.append(syllables[ind[i]])
                    if ind[i] < ind[i+1] and ind[i] < len(syllables):
                        ind[i] += 1 
                    else:
                        break
                    pass
        
        """ Zero Padding """
        
        trainDataZP = prep.zeroPad(self.trainDataRaw)
        testDataZP = prep.zeroPad(self.testDataRaw)
        
        """ MFCC extraction """
        
        self.trainDataMel = prep.getMEL(trainDataZP)
        self.testDataMel = prep.getMEL(testDataZP)
        
        """ shift and scale both datasets according to properties of training data """
        
        shifts, scales = prep.getShiftsAndScales(self.trainDataMel)
        
        trainDataNormalized = prep.normalizeData(self.trainDataMel, shifts, scales)
        testDataNormalized = prep.normalizeData(self.testDataMel, shifts, scales)
        
        """ Interpolate datapoints so that each sample has only (smoothLength) timesteps """
        
        self.trainDataSmoothend = prep.smoothenData(trainDataNormalized, smoothLength)
        self.testDataSmoothend = prep.smoothenData(testDataNormalized, smoothLength)
    
    def cLearning(self, N, SR, bias_scale, inp_scale, conn, gamma_pos, gamma_neg):
        """ Function that learns positive and negative conceptors on data with the following steps:
        1. create Reservoir
        2. Feed each sample of each syllable in reservoir and collect its states
        3. Use states to compute positive conceptor
        4. Use Conceptor logic to compute negative conceptor
        
        :param data: list of syllables with sample data
        :param N: size of the reservoir
        :param SR: spectral radius of the reservoir
        :param bias_scale: scaling of the bias while running reservoir
        :param inp_scale: scaling of the input when fed into the reservoir
        :param conn: scaling of the amount of connectivity within the reservoir
        :param gamma_pos: aperture of the positive conceptors
        :param gamma_neg: aperture of the negative conceptors
        
        :returns C_pos: List of positive conceptors
        :returns C_neg: List of negative conceptors
        """
        
        self.res = c.Reservoir(N = N, NetSR = SR, bias_scale = bias_scale, inp_scale = 0.2, conn = conn)
        self.C_pos = []
        
        for syllable in self.trainDataSmoothend:
            
            R_syll = np.zeros((syllable.shape[1] * (N + syllable.shape[2]), syllable.shape[0]))
            
            for i, sample in enumerate(syllable):
                
                self.res.run([sample], t_learn = len(sample), t_wash = 0, load = False)
                states = np.concatenate((self.res.TrainArgs.T, sample), axis = 1)
                R_syll[:,i] = np.reshape(states, states.shape[0] * states.shape[1])
            
            R = np.dot(R_syll, R_syll.T) / sample_length
            C_tmp = np.dot(R, np.linalg.inv(R + np.eye(len(R))))
            self.C_pos.append(C_tmp)
            
        self.C_neg = []
        
        for i in range(len(self.C_pos)):
            C = np.zeros_like(self.C_pos[0])
            for j in list(range(0,i))+list(range(i+1,len(self.C_pos))):
                C = fct.OR(C,self.C_pos[j])
            self.C_neg.append(C)
        
        for i in range(len(self.C_pos)):
            self.C_pos[i] = fct.phi(self.C_pos[i], gamma_pos)
            self.C_neg[i] = fct.phi(self.C_neg[i], gamma_neg)

    def cTest(self):
        """ Function that uses trained conceptors to recognize syllables in data by going through the following steps:
        1. Feed each sample of each syllable into reservoir and collect its states
        2. Analyize similarity of collected states and trained conceptors
        3. Choose syllable, for which similarity is highest
        
        :param data: list of syllables with sample data (different from training data)
        :param C_pos: list of trained positive Conceptors
        :param C_neg: list of trained negative Conceptors
        
        :returns evidences: list of arrays of evidences with rows = trials and columns = syllables
                            for positive, negative and combined conceptors
        :returns class_perf: Mean classification performance on test data set for 
                             positive, negative and combined conceptors
        """
        
        h_pos = []
        h_neg = []
        h_comb = []
        class_pos = []
        class_neg = []
        class_comb = []
        
        for syll_i, syllable in enumerate(self.testDataSmoothend):
            
            for sample in syllable:
                
                self.res.run([sample], t_learn = sample.shape[0], t_wash = 0, load = False)
                states = np.concatenate((self.res.TrainArgs.T,sample), axis = 1)
                z = np.reshape(states, states.shape[0] * states.shape[1])
                    
                h_pos_tmp = np.zeros(len(self.C_pos))
                h_neg_tmp = np.zeros(len(self.C_pos))
                h_comb_tmp = np.zeros(len(self.C_pos))
                
                for k in range(len(self.C_pos)):
                
                    h_pos_tmp[k] = np.dot(np.dot(z.T, self.C_pos[k]), z)
                    h_neg_tmp[k] = np.dot(np.dot(z.T, self.C_neg[k]), z)
                
                h_pos_tmp = h_pos_tmp - np.min(h_pos_tmp)
                h_pos_tmp = h_pos_tmp/np.max(h_pos_tmp)
                h_neg_tmp = h_neg_tmp - np.min(h_neg_tmp)
                h_neg_tmp = 1 - h_neg_tmp/np.max(h_neg_tmp)
                h_comb_tmp = (h_pos_tmp + h_neg_tmp) / 2
                h_pos.append(h_pos_tmp)
                h_neg.append(h_neg_tmp)
                h_comb.append(h_comb_tmp)
                
                classification_pos_tmp = 1 if np.where(h_pos_tmp == 1) == syll_i else 0
                classification_neg_tmp = 1 if np.where(h_neg_tmp == 1) == syll_i else 0
                classification_comb_tmp = 1 if np.where(h_comb_tmp == 1) == syll_i else 0
                
                class_pos.append(classification_pos_tmp)
                class_neg.append(classification_neg_tmp)
                class_comb.append(classification_comb_tmp)
                
        h_pos = np.array(h_pos)
        h_neg = np.array(h_neg)
        h_comb = np.array(h_comb)
        class_pos = np.array(class_pos)
        class_neg = np.array(class_neg)
        class_comb = np.array(class_comb)
        
        self.evidences = [h_pos, h_neg, h_comb]
        self.class_perf = [np.mean(class_pos), np.mean(class_neg), np.mean(class_comb)]

#%%
#
#""" Plotting """
#
#figure()
#subplot(1,3,1)
#imshow(h_pos, origin = 'lower', extent = [0,h_pos.shape[0],0,h_pos.shape[1]], aspect = 'auto', interpolation = 'none', cmap = 'Greys')
#title('positive evidence')
#subplot(1,3,2)
#imshow(h_neg, origin = 'lower', extent = [0,h_pos.shape[0],0,h_pos.shape[1]], aspect = 'auto', interpolation = 'none', cmap = 'Greys')
#title('negative evidence')
#subplot(1,3,3)
#imshow(h_comb, origin = 'lower', extent = [0,h_pos.shape[0],0,h_pos.shape[1]], aspect = 'auto', interpolation = 'none', cmap = 'Greys')
#title('combined evidence')