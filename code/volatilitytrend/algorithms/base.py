import numpy as np
from os.path import join
from cPickle import load

from .. import config
from .linearized_admm import linearizedADMM_fit

__metaclass__=type
class BaseAlgorithmClass():
    '''
    All alogrithm classes are sub-class of this class.
    '''
    
    def __init__(self,dataDir=None,destDir=None):
        if dataDir is None:
            self.dataDir=config.DATA_DIR
        else:
            self.dataDir=dataDir
        
        self.metadata={'n_rows':[],'n_cols':[],'T':[],'dates':[],
                       'lats':[],'lons':[]}
        self.dataMat=[]
        
    def loadData(self,data_fn,metadata_fn):
        
        #===load metadata===
        with open(join(self.dataDir,metadata_fn)) as f:
            self.metadata=load(f)
            
        n_rows,n_cols,T=(self.metadata['n_rows'],self.metadata['n_cols'],
                         self.metadata['T'])
        #===load metadata===

        #===load data===
        self.dataMat=np.fromfile(join(self.dataDir,data_fn),dtype='float32').\
                            reshape((n_rows*n_cols,T))
        #===load data===
    
 
class LinearizedADMM(BaseAlgorithmClass):

    def fit(self,destDir,lam_t_vec,lam_s_vec,
            mu=.01,maxIter=40000,freq=100,
            ifWarmStart=True,lh_trend=True,
            earlyStopping=True,patience=2,tol=.1):
        
        
        linearizedADMM_fit(self.dataMat,destDir,self.metadata,
                           lam_t_vec,lam_s_vec,mu=mu,
                           maxIter=maxIter,freq=freq,
                           ifWarmStart=ifWarmStart,lh_trend=lh_trend,
                           earlyStopping=earlyStopping,
                           patience=patience,tol=tol)
        
    
    
    
    
    
    
    
    
    
    
    
    
    