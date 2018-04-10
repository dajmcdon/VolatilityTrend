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

        if destDir is None:
            self.destDir=config.DATA_DIR
        else:
            self.destDir=destDir
        
        self.metadata={'nrows':[],'ncols':[],'T':[],'dates':[],
                       'lats':[],'lons':[]}
        self.dataMat=[]
        
    def loadData(self,data_fn,metadata_fn):
        
        #===load metadata===
        with open(join(self.dataDir,metadata_fn)) as f:
            self.metadata=load(f)
            
        nrows,ncols,T=(self.metadata['nrows'],self.metadata['ncols'],
               self.metadata['T'])
        #===load metadata===

        #===load data===
        self.dataMat=np.fromfile(join(self.dataDir,data_fn)).\
                            reshape((nrows*ncols,T))
        #===load data===
    
 
class LinearizedADMM(BaseAlgorithmClass):

    def fit(self,lam_t_vec,lam_s_vec,mu=.01,maxIter=40000,freq=100,
            ifWarmStart=False,lh_trend=True,
            earlyStopping=True,patience=2,tol=.1):
        

        
        
        
        linearizedADMM_fit()
        
    
    
    
    
    
    
    
    
    
    
    
    
    