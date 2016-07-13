# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:13:15 2016

@author: asus
"""

""" Libraries """

from matplotlib.pyplot import *
import syllableClassifier as sC
import os
import crossSylidation as cS

#%%

""" Parameters """

#path = os.path.abspath('D:/Data/Projects/StudyProject/syll')
##path = os.path.abspath('/Users/apple1/Desktop/syll')
#
#syllableN = 9
#cval_runs = 1
#gamma_pos = [25]
#gamma_neg = [27]
#trainSamplesN = 30
#
#SR = 20000
#downsampleType = 'IIR'
#n_mfcc = 17
#invCoeffOrder = True
#winsize = 20
#n_melFrames = 64
#smoothLength = 4
#polyOrder = 3
#
#nRes = 10
#SR = 1.2
#biasScale = 0.2
#inpScale = 1.0
#conn = 1.0

def runSyllClass(path, syllN, trainN = 30, cvalRuns = 1, sampRate = 20000, interpolType = 'IIF', mfccN = 17,
                 invCoeffOrder = True, winsize = 20, melFramesN = 64, smoothL = 4, polyOrder = 3, resN = 10, 
                 specRad = 1.2, biasScale = 0.2, inpScale = 1., conn = 1., gammaPos = 25, gammaNeg = 27, plot = True):
    """ Function that runs syllable classification in a supervised manner using positive, negative and combined conceptors.
    
    :param path: Full path to folder that includes subfolders for syllables which include samples of datatype wave (string)
    :param syllN: Number of syllables to include in train/test data (scalar)
    :param trainN: Number of training samples to use (scalar)
    :param cvalRuns: Number of runs with different training/test data splits (scalar)
    :param sampRate: Desired sampling rate of wave data (scalar)
    :param interpolType: Type of interpolation used for downsampling - 'mean' or 'IIR' (Chebichev filter)
    :param mfccN: Number of mel frequency cesptral coefficients to extract for each time point
    :param invCoeffOrder: False - extract first n mfccs; True: extract last n mfccs
    :param winsize: size of the time window to be used for mfcc extraction in ms (scalar)
    :param melFramesN: Number of timesteps to extract mfccs for (scalar)
    :param smoothL: Number of timesteps to downsample mfcc data to (scalar)
    :param polyOrder: Order the polynomial to be used for smoothing the mfcc data
    :param resN: Size of the reservoir to be used for classification (scalar)
    :param specRad: Desired spectral radius of the connectivity matrix of the reservoir (scalar)
    :param biasScale: Scaling of the bias term to affect each reservoir unit (scalar)
    :param inpScale: Scaling of the input to be entered into the reservoir (scalar)
    :param conn: Downscaling of the weights within the reservoir (scalar)
    :param gammaPos: Aperture to be used for positive conceptors
    :param gammaNeg: Aperture to be used for negative conceptors
    :param plot: boolean, True: Plot raw & smoothed mfcc data as well as (pos, neg, comb) evidences for last run
    
    :returns: cvalResults: Mean classification performance on test data over all runs for positive, negative and combined conceptors (list)
    """
    
    path = os.path.abspath(path)
    
    prepParams = {
    'SR': sampRate,
    'dsType': interpolType,
    'mel_channels': mfccN,
    'invCoeffOrder': invCoeffOrder,
    'winsize': winsize,
    'frames': melFramesN,
    'smoothLength': smoothL,
    'polyOrder': polyOrder}
    
    clearnParams = {
    'N': nRes,
    'SR': specRad,
    'bias_scale': biasScale,
    'inp_scale': inpScale,
    'conn': conn}
    
    classParameters = {
    'prepParams': prepParams,
    'clearnParams': clearnParams}

    syllClass = sC.syllableClassifier(path)
    cval_results = cS.crossVal(cvalRuns, trainN, syllN, syllClass, gammaPos, gammaNeg, **prepParams, **clearnParams)
    #cval_results = cS.crossValAperture(cvaRuns, trainN, syllN, syllClass, gammaPos, gammaNeg, **classParamters)
    
    if plot:
        
        figure(figsize=(15,5))
        syllables = [2, 5, 8]
        for syllable_i, syllable in enumerate(syllables):
            subplot(2, len(syllables), syllable_i + 1)
            xlim([0, 60])
            utteranceDataRaw = syllClass.trainDataMel[syllable - 1][0]
            for channel in range(n_mfcc):
                plot(utteranceDataRaw[:, channel])
            
            subplot(2, len(syllables), syllable_i + 1 + len(syllables))
            utteranceData = syllClass.trainDataSmoothend[syllable - 1][0]
            for channel in range(n_mfcc):
                plot(utteranceData[:, channel])
        
        h_pos = syllClass.evidences[0]
        h_neg = syllClass.evidences[1]
        h_comb = syllClass.evidences[2]
        
        figure()
        subplot(1,3,1)
        imshow(h_pos, origin = 'lower', extent = [0,h_pos.shape[0],0,h_pos.shape[1]], aspect = 'auto', interpolation = 'none', cmap = 'Greys')
        title('positive evidence')
        subplot(1,3,2)
        imshow(h_neg, origin = 'lower', extent = [0,h_pos.shape[0],0,h_pos.shape[1]], aspect = 'auto', interpolation = 'none', cmap = 'Greys')
        title('negative evidence')
        subplot(1,3,3)
        imshow(h_comb, origin = 'lower', extent = [0,h_pos.shape[0],0,h_pos.shape[1]], aspect = 'auto', interpolation = 'none', cmap = 'Greys')
        title('combined evidence')
        
        show()
        
    return cval_results
