# The cvBMS Unit
# _
# This module collects methods for calculating the
# cross-validated log model evidence (cvLME).


# import packages
#-----------------------------------------------------------------------------#
import numpy as np
import scipy.special as sp_special


###############################################################################
# class: univariate general linear model                                      #
###############################################################################
class GLM:
    """univariate general linear model"""
    
    # initialize GLM
    #-------------------------------------------------------------------------#
    def __init__(self, Y, X, V=None):
        self.Y = Y                          # data matrix
        self.X = X                          # design matrix
        if V is None: V = np.eye(Y.shape[0])# covariance matrix
        self.V = V                          # covariance matrix
        self.P = np.linalg.inv(V)           # precision matrix
        self.n = Y.shape[0]                 # number of observations
        self.v = Y.shape[1]                 # number of instances
        self.p = X.shape[1]                 # number of regressors
        
    # function: ordinary least squares
    #-------------------------------------------------------------------------#
    def OLS(self):
        B_est = np.linalg.inv(self.X.T @ self.X) @ (self.X.T @ self.Y)
        return B_est
    
    # function: weighted least squares
    #-------------------------------------------------------------------------#
    def WLS(self):
        B_cov = np.linalg.inv(self.X.T @ self.P @ self.X)
        B_est = B_cov @ (self.X.T @ self.P @ self.Y)
        return B_est
    
    # function: maximum likelihood estimation
    #-------------------------------------------------------------------------#
    def MLE(self):
        B_est  = self.WLS()
        E_est  = self.Y - (self.X @ B_est)
        s2_est = np.zeros(self.v)
        for j in range(self.v):
            s2_est[j] = 1/self.n * (E_est[:,j].T @ self.P @ E_est[:,j])
        return B_est, s2_est
    
    # function: Bayesian estimation
    #-------------------------------------------------------------------------#
    def Bayes(self, m0, L0, a0, b0):
        
        # enlarge priors if required
        if m0.shape[1] == 1:
            m0 = np.tile(m0, (1, self.v))
        if np.isscalar(b0):
            b0 = b0 * np.ones(self.v)
        
        # estimate posterior parameters
        Ln = self.X.T @ self.P @ self.X + L0
        mn = np.linalg.inv(Ln) @ ( self.X.T @ self.P @ self.Y + L0 @ m0 )
        an = a0 + self.n/2
        bn = np.zeros(self.v)
        for j in range(self.v):
            bn[j] = b0[j] + 1/2 * ( self.Y[:,j].T @ self.P @ self.Y[:,j] + m0[:,j].T @ L0 @ m0[:,j] - mn[:,j].T @ Ln @ mn[:,j] )
        
        # return posterior parameters
        return mn, Ln, an, bn
    
    # function: log model evidence
    #-------------------------------------------------------------------------#
    def LME(self, L0, a0, b0, Ln, an, bn):
        
        # calculate log model evidence
        LME = 1/2 * np.log(np.linalg.det(self.P)) - self.n/2 * np.log(2*np.pi)      \
            + 1/2 * np.log(np.linalg.det(L0))     - 1/2 * np.log(np.linalg.det(Ln)) \
            + sp_special.gammaln(an)              - sp_special.gammaln(a0)          \
            + a0 * np.log(b0)                     - an * np.log(bn)
        
        # return log model evidence
        return LME
    
    # function: cross-validated log model evidence
    #-------------------------------------------------------------------------#
    def cvLME(self, S=2):
        
        # determine data partition
        npS  = np.int(self.n/S);# number of data points per subset, truncated
        inds = range(S*npS)     # indices for all data, without remainders
        
        # set non-informative priors
        m0_ni = np.zeros((self.p,1))        # flat Gaussian
        L0_ni = np.zeros((self.p,self.p))
        a0_ni = 0;                          # Jeffrey's prior
        b0_ni = 0;
        
        # calculate out-of-sample log model evidences
        #---------------------------------------------------------------------#
        oosLME = np.zeros((S,self.v))
        for j in range(S):
            
            # set indices
            i2 = range(j*npS, (j+1)*npS)                # test indices
            i1 = [i for i in inds if i not in i2]       # training indices
            
            # partition data
            Y1 = self.Y[i1,:]                           # training data
            X1 = self.X[i1,:]
            P1 = self.P[i1,:][:,i1]
            S1 = GLM(Y1, X1, P1)
            Y2 = self.Y[i2,:]                           # test data
            X2 = self.X[i2,:]
            P2 = self.P[i2,:][:,i2]
            S2 = GLM(Y2, X2, P2)
            
            # calculate oosLME
            m01 = m0_ni; L01 = L0_ni; a01 = a0_ni; b01 = b0_ni;
            mn1, Ln1, an1, bn1 = S1.Bayes(m01, L01, a01, b01)
            m02 = mn1; L02 = Ln1; a02 = an1; b02 = bn1;
            mn2, Ln2, an2, bn2 = S2.Bayes(m02, L02, a02, b02)
            oosLME[j,:] = S2.LME(L02, a02, b02, Ln2, an2, bn2)
            
        # return cross-validated log model evidence
        cvLME = np.sum(oosLME,0)
        return cvLME
    
    
###############################################################################
# class: Poisson distribution                                                 #
###############################################################################
class Poiss:
    """Poisson distribution"""
    
    # initialize Poisson
    #-------------------------------------------------------------------------#
    def __init__(self, Y, x=None):
        self.Y = Y                          # data matrix
        if x is None:
            x = np.ones(Y.shape[0])         # design vector
        self.x = x
        self.n = Y.shape[0]                 # number of observations
        self.v = Y.shape[1]                 # number of instances
        
    # function: maximum likelihood estimation
    #-------------------------------------------------------------------------#
    def MLE(self):
        l_est = 1/self.n * np.sum(self.Y,0)
        return l_est
    
    # function: Bayesian estimation
    #-------------------------------------------------------------------------#
    def Bayes(self, a0, b0):
        
        # enlarge priors if required
        if np.isscalar(a0):
            a0 = a0 * np.ones(self.v)
        
        # estimate posterior parameters
        an = a0 + self.n * np.sum(self.Y,0)
        bn = b0 + self.n * np.sum(self.x)
        
        # return posterior parameters
        return an, bn
    
    # function: log model evidence
    #-------------------------------------------------------------------------#
    def LME(self, a0, b0, an, bn):
        
        # calculate log model evidence
        x   = np.reshape(self.x, (self.n, 1))
        X   = np.tile(x, (1, self.v))
        LME = np.sum(self.Y * np.log(X), 0) - np.sum(sp_special.gammaln(self.Y+1), 0) \
            + sp_special.gammaln(an)        - sp_special.gammaln(a0)                  \
            + a0 * np.log(b0)               - an * np.log(bn)
        
        # return log model evidence
        return LME
    
    # function: cross-validated log model evidence
    #-------------------------------------------------------------------------#
    def cvLME(self, S=2):
        
        # determine data partition
        npS  = np.int(self.n/S);# number of data points per subset, truncated
        inds = range(S*npS)     # indices for all data, without remainders
        
        # set non-informative priors
        a0_ni = 0;
        b0_ni = 0;
        
        # calculate out-of-sample log model evidences
        #---------------------------------------------------------------------#
        oosLME = np.zeros((S,self.v))
        for j in range(S):
            
            # set indices
            i2 = range(j*npS, (j+1)*npS)                # test indices
            i1 = [i for i in inds if i not in i2]       # training indices
            
            # partition data
            Y1 = self.Y[i1,:]                           # training data
            x1 = self.x[i1]
            S1 = Poiss(Y1, x1)
            Y2 = self.Y[i2,:]                           # test data
            x2 = self.x[i2]
            S2 = Poiss(Y2, x2)
            
            # calculate oosLME
            a01 = a0_ni; b01 = b0_ni;
            an1, bn1 = S1.Bayes(a01, b01)
            a02 = an1; b02 = bn1;
            an2, bn2 = S2.Bayes(a02, b02)
            oosLME[j,:] = S2.LME(a02, b02, an2, bn2)
            
        # return cross-validated log model evidence
        cvLME = np.sum(oosLME,0)
        return cvLME
    
    
###############################################################################
# class: model space                                                          #
###############################################################################
class MS:
    """model space"""
    
    # initialize MS
    #-------------------------------------------------------------------------#
    def __init__(self, LME):
        self.LME = LME          # log model evidences
        self.M   = LME.shape[0] # number of models
        self.N   = LME.shape[1] # number of instances
        
    # function: log Bayes factor
    #-------------------------------------------------------------------------#
    def LBF(self, m1=1, m2=2):
        LBF12 = self.LME[m1-1,:] - self.LME[m2-1,:]
        return LBF12
    
    # fucntion: Bayes factor
    #-------------------------------------------------------------------------#
    def BF(self, m1=1, m2=2):
        BF12 = np.exp(self.LBF(m1, m2))
        return BF12
        
    # function: posterior model probabilities
    #-------------------------------------------------------------------------#
    def PP(self, prior=None):
        
        # set uniform prior
        if prior is None:
            prior = 1/self.M * np.ones((self.M,1))
        if prior.shape[1] == 1:
            prior = np.tile(prior, (1, self.N))
            
        # subtract average LMEs
        LME = self.LME
        LME = LME - np.tile(np.mean(LME,0), (self.M, 1))
        
        # calculate PPs
        post = np.exp(LME) * prior
        post = post / np.tile(np.sum(post,0), (self.M, 1))
        
        # return PPs
        return post
    
    # function: log family evidences
    #-------------------------------------------------------------------------#
    def LFE(self, m2f):
        
        # get number of model families
        F = np.int(m2f.max())
        
        # calculate log family evidences
        #---------------------------------------------------------------------#
        LFE = np.zeros((F,self.N))
        for f in range(F):
            
            # get models from family
            mf = [i for i, m in enumerate(m2f) if m == (f+1)]
            Mf = len(mf)
            
            # set uniform prior
            prior = 1/Mf * np.ones((Mf,1))
            prior = np.tile(prior, (1, self.N))
            
            # calculate LFEs
            LME_fam  = self.LME[mf,:]
            LME_fam  = LME_fam + np.log(prior) + np.log(Mf)
            LME_max  = LME_fam.max(0)
            LME_diff = LME_fam - np.tile(LME_max, (Mf, 1))
            LFE[f,:] = LME_max + np.log(np.mean(np.exp(LME_diff),0))
            
        # return log family evidence
        return LFE
    