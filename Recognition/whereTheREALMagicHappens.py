# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:10:28 2016

@author: artur
"""

import os.path
import sys
sys.path.append(os.path.abspath('../Classes'))

import reservoir_c
import functions

import numpy as np
import matplotlib as plt
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import pickle

with open('arturaid.pkl','rb') as f:
    japsen_data = pickle.load(f)
    f.close()
 
test = japsen_data['test']; train = japsen_data['train']

N = 10
SR = 1.2
bias_scale = 1
gamma_pos = 25
gamma_neg = 27

if N <= 20:
    connectivity = 1;
else:
    connectivity = 10/N;

Z = np.zeros((88,30))
C_pos = []
C_neg = []

W = reservoir_c.Reservoir(N = N, NetSR = SR , conn = connectivity, bias_scale = bias_scale, inp_scale = 0.2)

for s in range(9):
    for i in range(30):
        
        data = [train[s][i]]
        W.run(data, t_learn = 4, t_wash = 0, load = False)
        state = W.TrainArgs
        z = np.reshape(np.concatenate((state.T,data[0]),axis = 1),(88))   
        Z[:,i] = z
    R = np.dot(Z,Z.T)/30   
    C_pos.append(np.dot(R,np.linalg.inv(R+np.eye(88))))
    
for i in range(9):
    C = np.zeros_like(C_pos[0])
    for j in list(range(0,i))+list(range(i+1,9)):
        C = functions.OR(C,C_pos[j])
    C_neg.append(C)

''' Pack das hier noch in deine functions
def phi(C, gamma):
    return np.dot(C, np.linalg.inv((C+gamma**(-2)*(np.eye(len(C))-C))))
'''

for i in range(9):
    C_pos[i] = functions.phi(C_pos[i], gamma_pos)
    C_neg[i] = functions.phi(C_neg[i], gamma_neg)  
    
h_pos = np.zeros((370,9))
h_neg = np.zeros((370,9))
h_comb = np.zeros((370,9))

count = 0
for s in range(9):
    for i in range(len(test[s])):
        data = [test[s][i]]
        W.run(data, t_learn = 4, t_wash = 0, load = False)
        state = W.TrainArgs
        z = np.reshape(np.concatenate((state.T,data[0]),axis = 1),(88)) 
        for c in range(9):
            h_pos[count,c] = np.dot(np.dot(z.T,C_pos[c]),z)
            h_neg[count,c] = np.dot(np.dot(z.T,C_neg[c]),z)
        h_pos[count] = h_pos[count]-np.min(h_pos[count])
        h_pos[count] = h_pos[count]/np.max(h_pos[count])
        h_neg[count] = h_neg[count]-np.min(h_neg[count])
        h_neg[count] = 1-(h_neg[count]/np.max(h_neg[count]))
        h_comb[count] = (h_pos[count] + h_neg[count])/2
        count += 1
        
plt.figure(figsize= (30,10))
gs1 = gridspec.GridSpec(1,3)
plt.subplot(gs1[0])
plt.imshow(h_pos, origin='lower', extent=[0,370,0,9], aspect='auto', interpolation = 'none', cmap = 'Greys')
plt.subplot(gs1[1])
plt.imshow(h_neg, origin='lower', extent=[0,370,0,9], aspect='auto', interpolation = 'none', cmap = 'Greys')      
plt.subplot(gs1[2])  
plt.imshow(h_comb, origin='lower', extent=[0,370,0,9], aspect='auto', interpolation = 'none', cmap = 'Greys')