#from .. import config
from os.path import join
import numpy as np
from scipy import stats
from cPickle import dump
import pandas as pd

def computeCovarianceMatrix(n_rows,n_cols,
                            var_sources_centers,var_sources_weight,
                            var_sources_sigma):
    '''
    This function computes the covariance matrix at each time.
    The covariance matrix is computed as a weighted sum of several
    Guassian functions at different locations on the grid and with 
    different variances. We call each Gaussian a "variance source".
    
    Parameters
    ---------    
    n_rows: float
        Number of rows in the grid.
        
    n_cols: float
        Number of columns in the grid.

    var_sources_centers: numpy.array
        This is a n_sources x 2 array where each row is the center 
        of a source (the mean of the Gaussian). The first column is 
        the row number and the second column is the column number
        of the centers.
    
    var_sources_weight: array_like
        Each element is the weight of a Gaussian source.
    
    var_sources_sigma: array_like
        Each element is the SD of a Gaussian source.
    
        
    Returns
    -------
    covarMat: numpu.array
        The covariance matrix at a time. This will have the same size as
        the grid.
        
    '''
    
    n_sources=var_sources_centers.shape[0]
    covarMat=np.zeros((n_rows,n_cols))
    
    Y,X=np.meshgrid(np.arange(n_cols),np.arange(n_rows))

    for s in range(n_sources):
        center=var_sources_centers[s,];        
        weight=var_sources_weight[s]
        sigma=var_sources_sigma[s]
        
        dist = (X-center[0])**2 + (Y-center[1])**2
        covarMat=covarMat + weight*sigma*np.sqrt(2*np.pi)*\
                            stats.norm.pdf(dist,0,sigma)   
                
    return covarMat


def simulateSpatioTemporalData(dstDir,n_rows,n_cols,T,var_sources_centers,
                               var_sources_weight_mat,var_sources_sigma):
    
    times=np.arange(0,T)
    simulated_data=np.zeros((n_rows*n_cols,T))
    covarMat=np.zeros((n_rows*n_cols,T))
    for t in times:
        covarMat[:,[t]]=computeCovarianceMatrix(n_rows,n_cols,
                                                var_sources_centers,
                                                var_sources_weight_mat[:,t],
                                                var_sources_sigma).\
                                                     reshape((-1,1),order='F')
        simulated_data[:,[t]]=np.random.normal(scale=covarMat[:,[t]])


    #===metadata===
    dates=pd.date_range('2010-01-01',periods=T,freq='1W')
    metadata={'n_rows':n_rows,'n_cols':n_cols,'T':T,'dates':dates}
    #===metadata===
    
    #===saving data===
    covarMat.tofile(join(dstDir,'covMat'))
    np.array(simulated_data,dtype='float32').\
        tofile(join(dstDir,'simulated_data'))
    with open(join(dstDir,'metadata'),'wb') as f:
        dump(metadata,f)
    #===saving data===


    

