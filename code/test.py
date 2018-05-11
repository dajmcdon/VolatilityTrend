import numpy as np
from os.path import join
from volatilitytrend.algorithms.base import ConsensusADMM
import matplotlib.pyplot as plt

bl=[False,True]
ifSimulateData=bl[0]
ifFitModel=bl[1]
ifVisualize=bl[1]
ifModelSelection=bl[0]

dstDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/Simulated data'


lam_t_vec=[5.];
lam_s_vec=[0.1];
rho=.015

ca = ConsensusADMM()#construct linearizedADMM object
data_fn=join(dstDir,'simulated_data')
metadata_fn=join(dstDir,'metadata')
ca.loadData(data_fn,metadata_fn)#load data

if ifFitModel: 
    ca.fit(dstDir,lam_t_vec,lam_s_vec,rho=rho,maxIter=100,freq=10,
           n_r_b=3,n_c_b=3,lh_trend=True,wrapAround=False,
           earlyStopping=False,patience=2,tol=.1)
    
if ifVisualize:
    lam_t=5.;lam_s=.1;
    
    rhp,lam_t,lam_s=(float(rho),float(lam_t),float(lam_s))
    fn='rho_{}_lam_t_{}_lam_s_{}'.format(rho,lam_t,lam_s)
    
    #===load fitted values===
    n_rows=ca.metadata['n_rows'];n_cols=ca.metadata['n_cols']
    gridsize=n_rows*n_cols
    
    covMat = np.fromfile(join(dstDir,'covMat')).reshape((gridsize,-1))
    filepath=join(dstDir,'Z_'+fn)
    ca.analyseFittedValues(filepath)
    plt.figure()
    i=0;plt.plot(np.exp(ca.fittedVar[i,]/2));plt.plot(covMat[i,:])
    #===load fitted values===