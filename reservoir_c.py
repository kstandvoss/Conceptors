# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 16:18:45 2015

@author: User
"""
import functions
import numpy as np

class Reservoir:
    
    def __init__(self, N = 100, alpha = 10, NetSR = 1.5, bias_scale = 0.2, inp_scale = 1.5):
                 
        self.N = N; self.alpha = alpha; self.NetSR = NetSR; 
        self.bias_scale = bias_scale; self.inp_scale = inp_scale
        conn = 10./self.N
        
        self.W_raw = self.NetSR * functions.IntWeights(self.N,self.N,conn)
        self.W_bias = self.bias_scale*np.random.randn(self.N)

        
    def load(self, patterns, t_learn = 1000, t_wash = 100, TyA_wout = 0.01, TyA_wload = 0.0001, 
             gradient_load = False, gradient_c = False, gradient_window = 1, c_adapt_rate = 0.01, gradient_cut = 2.0):
        
        self.patterns = patterns; self.t_learn = t_learn; self.t_wash = t_wash; self.TyA_wout = TyA_wout; self.TyA_wload = TyA_wload
        self.gradient_load = gradient_load; self.gradient_c = gradient_c; self.gradien_cut = gradient_cut; self.c_adapt_rate = c_adapt_rate        
        self.n_patts = len(self.patterns)
        
        if type(self.patterns[0](0)) == np.float64:
            self.n_ip_dim = 1
        else:
           self.n_ip_dim = len(self.patterns[0](0))
        
        self.W_in = self.inp_scale*np.random.randn(self.N,self.n_ip_dim)
        
        self.C = []            
        
        TrainArgs = np.zeros([self.N,self.n_patts*self.t_learn])
        TrainOldArgs = np.zeros([self.N,self.n_patts*self.t_learn])
        TrainOuts = np.zeros([self.n_ip_dim,self.n_patts*self.t_learn]) 
        I = np.eye(self.N)
      
        for i,p in zip(xrange(self.n_patts), self.patterns):

            x =         np.zeros([self.N]) 
            xOld =      np.zeros([self.N]) 
            xColl =     np.zeros([self.N,self.t_learn])  
            xOldColl =  np.zeros([self.N,self.t_learn])  
            uColl =     np.zeros([self.n_ip_dim,self.t_learn])    
            Cc =        np.zeros([self.N,self.N])            

            for t in xrange(self.t_learn+self.t_wash):
                
                u = np.reshape(p(t), self.n_ip_dim)            
                xOld = x          
                
                x = np.tanh(np.dot(self.W_raw,x)+np.dot(self.W_in,u)+self.W_bias)           
               
                if gradient_c:           
            
                    grad = x-np.dot(Cc,x)
                    norm = np.linalg.norm(grad)     
                    if (norm > self.gradien_cut):
                        grad = self.gradien_cut/norm * grad 
                    Cc = Cc + self.c_adapt_rate*(np.outer(grad,x.T)-(self.alpha**-2)*Cc)
                
                if (t > self.t_wash): 
                
                    xColl[:,t-self.t_wash] = x
                    xOldColl[:,t-self.t_wash] = xOld
                    uColl[:,t-self.t_wash] = u                    
            
            if not gradient_c:
            
                R = np.dot(xColl,np.transpose(xColl))/self.t_learn
                U,S,V = np.linalg.svd(R, full_matrices=True)  
                S = np.diag(S)      
                S = (np.dot(S,np.linalg.inv(S + (self.alpha**-2)*I)))
                self.C.append(np.dot(U,np.dot(S,U.T)))        
                
            else:
                
                self.C.append(Cc) 

            TrainArgs[:,i*self.t_learn:(i+1)*self.t_learn] = xColl
            TrainOldArgs[:,i*self.t_learn:(i+1)*self.t_learn] = xOldColl
            TrainOuts[:,i*self.t_learn:(i+1)*self.t_learn] = uColl        
        
        """ Output Training """    
            
        self.W_out = functions.RidgeWout(TrainArgs, TrainOuts, self.TyA_wout)
        self.NRMSE_readout = functions.NRMSE(np.dot(self.W_out,TrainArgs), TrainOuts);
        print self.NRMSE_readout
        
        """ Loading """
        
        W_bias_rep = np.tile(self.W_bias,(self.n_patts*self.t_learn,1)).T
        W_targets = (np.arctanh(TrainArgs) - W_bias_rep)
        self.W = functions.RidgeWload(TrainOldArgs, W_targets, self.TyA_wload )
        self.NRMSE_load =  functions.NRMSE(np.dot(self.W,TrainOldArgs),W_targets)
        print np.mean(self.NRMSE_load)        
        
    def recall(self, t_recall = 200, cueing = False, t_cue = 30, c_adapt_rate = 0.01, t_cue_washout = 100, t_adapt = 500, gradien_cut = 2.0):
        
        self.Y_recalls = []; self.t_recall = t_recall; self.cueing = cueing; self.gradien_cut = gradien_cut
        self.t_cue = t_cue; self.c_adapt_rate = c_adapt_rate; self.t_cue_washout = t_cue_washout; self.t_adapt = t_adapt
                
        for i,p in zip(xrange(self.n_patts), self.patterns):

            x = 0.5*np.random.randn(self.N)
            y_recall = np.zeros([self.t_recall, self.n_ip_dim])

            if not self.cueing: 
            
                Cc = self.C[i]
            
                for t in range(self.t_recall):
                    
                    x = np.dot(Cc,np.tanh(np.dot(self.W,x)+self.W_bias))
                    y_recall[t]= np.dot(self.W_out,x)
                    
                self.Y_recalls.append(y_recall)   
                
            else: 
                
                Cc = np.zeros([self.N, self.N])                
                
                for t in xrange(self.t_cue_washout + self.t_cue):
                    
                    u = np.reshape(p(t), self.n_ip_dim) 
                    x = np.tanh(np.dot(self.W_raw,x)+np.dot(self.W_in,u)+self.W_bias)
                    
                    if t > t_cue_washout:
  
                        grad = x-np.dot(Cc,x)
                        norm = np.linalg.norm(grad)     
                        if (norm > self.gradien_cut):
                            grad = self.gradien_cut/norm * grad                      
                        Cc = Cc + self.c_adapt_rate*(np.outer(grad,x.T)-(self.alpha**-2)*Cc)
                        
                for t in xrange(self.t_adapt):
                    
                    x = np.dot(Cc,np.tanh(np.dot(self.W,x)+self.W_bias))

                    grad = x-np.dot(Cc,x)
                    norm = np.linalg.norm(grad)     
                    if (norm > self.gradien_cut):
                        grad = self.gradien_cut/norm * grad                      
                    Cc = Cc + self.c_adapt_rate*(np.outer(grad,x.T)-(self.alpha**-2)*Cc)    
                    
                for t in xrange(self.t_recall):                    
                
                    x = np.dot(Cc,np.tanh(np.dot(self.W,x)+self.W_bias))
                    y_recall[t]= np.dot(self.W_out,x)
                    
                self.Y_recalls.append(y_recall)                    
                    
            
            