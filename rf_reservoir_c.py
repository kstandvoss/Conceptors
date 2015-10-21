# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:48:49 2015

@author: aspeiser
"""

import functions
import numpy as np
import scipy as sp


class RF_Reservoir:
    
    def __init__(self, N = 200, K = 1000, alpha = 3, NetSR = 1.4, bias_scale = 0.2, inp_scale = 1.2):
                 
        self.N = N; self.K = K; self.alpha = alpha; self.NetSR = NetSR; 
        self.bias_scale = bias_scale; self.inp_scale = inp_scale
        
        self.F = np.random.randn(self.K, self.N)
        self.G = np.random.randn(self.N, self.K)
        
        sr = np.max(np.abs(sp.linalg.eigvals(np.dot(self.F,self.G))))
        
        self.F *= np.sqrt(self.NetSR)/np.sqrt(sr)   
        self.G *= np.sqrt(self.NetSR)/np.sqrt(sr)
        
        self.W_bias = self.bias_scale*np.random.randn(self.N)

        
    def load(self, patterns, t_learn = 400, t_cadapt = 2000, t_wash = 200, TyA_wout = 1, TyA_wload = 0.01, 
             gradient_load = False, gradient_c = False, gradient_window = 1, c_adapt_rate = 0.5, gradient_cut = 2.0):
        
        self.patterns = patterns; self.t_learn = t_learn; self.t_cadapt = t_cadapt; self.t_wash = t_wash; self.TyA_wout = TyA_wout; self.TyA_wload = TyA_wload
        self.gradient_load = gradient_load; self.gradient_c = gradient_c; self.gradien_cut = gradient_cut; self.c_adapt_rate = c_adapt_rate        
        self.n_patts = len(self.patterns)
        
        if not self.gradient_load: self.c_adapt = 0         
        
        if type(self.patterns[0](0)) == np.float64:
            self.n_ip_dim = 1
        else:
           self.n_ip_dim = len(self.patterns[0](0))
        
        self.W_in = self.inp_scale*np.random.randn(self.N,self.n_ip_dim)
        
        self.C = []            
        
        TrainRs = np.zeros([self.N,self.n_patts*self.t_learn])
        TrainZOld = np.zeros([self.K,self.n_patts*self.t_learn])
        TrainOuts = np.zeros([self.n_ip_dim,self.n_patts*self.t_learn]) 
        I = np.eye(self.N)
      
        for i,p in zip(xrange(self.n_patts), self.patterns):

            z =         np.zeros([self.K]) 
            rColl =     np.zeros([self.N,self.t_learn])  
            zOldColl =  np.zeros([self.K,self.t_learn])  
            uColl =     np.zeros([self.n_ip_dim,self.t_learn]) 
            c =         np.ones([self.K])
            
            for t in xrange(self.t_learn + self.t_wash + self.t_cadapt):
                
                u = np.reshape(p(t), self.n_ip_dim)            
                r = np.tanh(np.dot(self.G,z) + np.dot(self.W_in,u) + self.W_bias)               
                zOld = z
                z = c*np.dot(self.F,r)           
               
                if gradient_c and t > self.t_wash and t < self.t_cadapt:           
            
                    c = c + self.c_adapt_rate*(z*z - c*z*z - (self.alpha**-2)*c)
                
                if (t > self.t_wash + self.t_cadapt): 
                
                    zOldColl[:, t - (self.t_wash + self.t_cadapt)]  = zOld
                    rColl[:, t - (self.t_wash + self.t_cadapt)]     = r
                    uColl[:, t - (self.t_wash + self.t_cadapt)]     = u   
                                                    
            if not gradient_c:
            
                c = np.mean(zOldColl**2, axis = 1)*(np.mean(zOldColl**2, axis = 1)+np.ones(self.K)*3**-2)**-1
                
            self.C.append(c) 
            
            TrainRs[:,i*self.t_learn:(i+1)*self.t_learn] = rColl
            TrainZOld[:,i*self.t_learn:(i+1)*self.t_learn] = zOldColl
            TrainOuts[:,i*self.t_learn:(i+1)*self.t_learn] = uColl 
        
        """ Output Training """    
            
        self.W_out = functions.RidgeWout(TrainRs, TrainOuts, self.TyA_wout)
        self.NRMSE_readout = functions.NRMSE(np.dot(self.W_out,TrainRs), TrainOuts);
        print self.NRMSE_readout
        self.G = functions.RidgeWload(TrainZOld,np.dot(self.G,TrainZOld),TyA_wload)
        
        """ Loading """
        
        self.D = functions.RidgeWload(TrainZOld,TrainOuts,self.TyA_wload)
        self.NRMSE_load =  functions.NRMSE(np.dot(self.D,TrainZOld),TrainOuts)
        print np.mean(self.NRMSE_load)        
        
    def recall(self, t_recall = 200, cueing = False, t_cue = 30, c_adapt_rate = 0.01, t_cue_washout = 100, t_adapt = 500, gradien_cut = 2.0):
        
        self.Y_recalls = []; self.t_recall = t_recall; self.cueing = cueing; self.gradien_cut = gradien_cut
        self.t_cue = t_cue; self.c_adapt_rate = c_adapt_rate; self.t_cue_washout = t_cue_washout; self.t_adapt = t_adapt
                
        for i,p in zip(xrange(self.n_patts), self.patterns):

            z = np.zeros([self.K])
            y_recall = np.zeros([self.t_recall, self.n_ip_dim])

            if not self.cueing: 
            
                Cc = self.C[i]
            
                for t in range(self.t_recall + self.t_wash):
                    
                    r = np.tanh(np.dot(self.G,z) + np.dot(self.W_in,np.dot(self.D,z)) + self.W_bias)
                    z = Cc*np.dot(self.F,r)
                    if t > self.t_wash: y_recall[t-self.t_wash]= np.dot(self.W_out,r)
                    
                self.Y_recalls.append(y_recall)   
                

            
            