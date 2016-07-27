# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:19:20 2016

@author: asus
"""

from matplotlib.pyplot import *
import numpy as np
from songClassifier import *
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

#%matplotlib inline

#%%
# create list of syllables and initialize SongClassifier with it
syllables = ['aa','ao','ba','bm','ca','ck','da','dl','ea','ej']
SC = SongClassifier(syllables)

# define parameters of songClassifier
RFCParams = {'N': 400,
             'K': 2000,
             'NetSR': 1.5,
             'bias_scale': 1.2,
             'inp_scale': 1.5}
loadingParams = {'gradient_c': True}
dataPrepParams = {}
cLearningParams = {}
HFCParams = {'sigma': 0.9,
             'drift': 0.05,
             'gammaRate': 0.002,
             'dcsv': 6,
             'SigToNoise': float('inf')}

#%%
# create random songs and load them into a RFC
s1_length = 3
s2_length = 5
SC.addSong(s1_length)
SC.addSong(s2_length)
SC.loadSongs(RFCParams = RFCParams, loadingParams = loadingParams)

# plot RFC recall
plotrange = 50
figure()
for s in range(len(SC.Songs)):
    recall = np.argmax(SC.R.Y_recalls[s][0:plotrange,:], axis = 1)
    target = np.argmax(SC.patterns[s][0:plotrange,:], axis = 1)
    subplot(len(SC.Songs),1,s+1)
    plot(recall, 'r')
    plot(target, 'b')
    ylim([0,SC.nSylls - 1])
    ylabel(' Syllable #')
    xlabel('t')

#%%
# run song classification and plot gammas
SC.run(pattRepRange = (10,20), nLayers = 2, useSyllRecog = True, SyllPath = '/Users/pfaion/Desktop/syll',
       dataPrepParams = dataPrepParams, cLearningParams = cLearningParams, HFCParams = HFCParams)
SC.H.plot_gamma()


