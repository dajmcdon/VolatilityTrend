'''
In this script we compare the performance of four models for
in estimating the variance. The models are:
    1. linearizedADMM with optimal penalty parameters
    2. linearizedADMM with lam_t=0
    3. linearizedADMM with lam_s=0
    4. GARCH(1,1)

'''

import numpy as np
from volatilitytrend.data_utils.simulate_data import simulateSpatioTemporalData
from os.path import join
from volatilitytrend.algorithms.base import LinearizedADMM
from arch import arch_model
import pandas as pd
from time import ctime
import matplotlib.pyplot as plt

n_sim=20;#number of simulations

#===models parameters===
#---linearizedADMM optimal---
lam_t_opt=5.;lam_s_opt=0.1;mu=.01

#---linearizedADMM temporal---
lam_t_temp=5.;lam_s_temp=0.;mu=.01

#---linearizedADMM spatial---
lam_t_sp=0.;lam_s_sp=.1;mu=.01
#===models parameters===

#===Specify parameters of data simulator===
n_rows=5;n_cols=7;
n_yeras=15
T=n_yeras*52#time-series length
gridsize=n_rows*n_cols

#---specify sources properties---
var_sources_centers=np.array([[0,0],[0,5],[3,0],[3,5]])
var_sources_sigma=[7.]*4
#---specify sources properties---

#---compute the weight of each source at wach time---
times=np.arange(0,T)
trend1=.5*times/float(T)
trend2=.1*times/float(T)
trend3=-.5*times/float(T)
trend4=-.1*times/float(T)

var_sources_weight1=np.exp(np.sin(n_yeras*2*np.pi*times/float(T)))+trend1
var_sources_weight2=np.exp(np.sin(n_yeras*2*np.pi*times/float(T)))+trend2
var_sources_weight3=np.exp(np.cos(n_yeras*2*np.pi*times/float(T))+trend3)
var_sources_weight4=np.exp(np.cos(n_yeras*2*np.pi*times/float(T))+trend4)
var_sources_weight_mat=np.concatenate((var_sources_weight1,
                                       var_sources_weight2,
                                       var_sources_weight3,
                                       var_sources_weight4)).\
                                       reshape((4,-1))
#---compute the weight of each source at wach time---

#---data directories---                                       
dstDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/Simulated data'
savedResultsDir='/home/arash/datasets/ecmwf/Fits_to_simulated_data/'+\
'LinADMM_withTrend'
saveFigDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Reports/051218/Figures'

data_fn=join(dstDir,'simulated_data')
metadata_fn=join(dstDir,'metadata')
#---data directories---
#===Specify parameters of data simulator===
                                       
def computeMAE_ADMM(data_fn,metadata_fn,savedResultsDir,lam_t,lam_s,covMat):
    la = LinearizedADMM()#construct linearizedADMM object    
    la.loadData(data_fn,metadata_fn)#load data
    
    #---fit linearizedADMM---
    la.fit(dstDir,[float(lam_t)],[float(lam_s)],mu=mu,
           maxIter=1000,freq=500,
           lh_trend=True,wrapAround=False,patience=2,
           ifWarmStart=False,earlyStopping=False)
    fn='mu_{}_lam_t_{}_lam_s_{}'.format(mu,lam_t,lam_s)
    filepath=join(dstDir,'X_'+fn)
    X=np.fromfile(filepath).reshape((gridsize,T))
    return np.mean(np.abs(np.exp(X/2)-covMat))
#    return np.mean(np.abs((X/2.)-np.log(covMat)))

def computeMAE_GARCH(data_fn,metadata_fn,covMat):
    la = LinearizedADMM()#construct linearizedADMM object    
    la.loadData(data_fn,metadata_fn)#load data
    dataMat=la.dataMat
    
    #---fit GARCH---
    err=[]   
    for r in range(dataMat.shape[0]):
        y=la.dataMat[r,:]
        am=arch_model(y)
        res=am.fit()
        err.append(np.mean(np.abs(res.conditional_volatility-covMat[r,:])))
#        err.append(np.mean(np.abs(np.log(res.conditional_volatility)-\
#                                  np.log(covMat[r,:]))))
    return np.mean(err)
    

#===simulation and model fitting=== 
MAE_opt=[];MAE_temp=[];MAE_sp=[];MAE_ga=[]
for i in range(n_sim):
    print(ctime()+'...simulation number {}'.format(i))
    
    #simulate data
    var_sources_sigma=[4+3*np.random.rand()]*4
    simulateSpatioTemporalData(dstDir,n_rows,n_cols,T,var_sources_centers,
                               var_sources_weight_mat,var_sources_sigma)
    covMat = np.fromfile(join(savedResultsDir,'covMat')).\
                reshape((gridsize,-1))
    
    
    #---fit optimal linearizedADMM---
    MAE_opt.append(computeMAE_ADMM(data_fn,metadata_fn,savedResultsDir,
                                   lam_t_opt,lam_s_opt,covMat))

    #---fit temporal linearizedADMM---
    MAE_temp.append(computeMAE_ADMM(data_fn,metadata_fn,savedResultsDir,
                                    lam_t_temp,lam_s_temp,covMat))
        
    #---fit spatial linearizedADMM---
    MAE_sp.append(computeMAE_ADMM(data_fn,metadata_fn,savedResultsDir,
                                  lam_t_sp,lam_s_sp,covMat))

    #---fit GARCH---
    MAE_ga.append(computeMAE_GARCH(data_fn,metadata_fn,covMat))    

errors=pd.DataFrame({'admm_opt':MAE_opt,'admm_temp':MAE_temp,'admm_sp':MAE_sp,
           'garch':MAE_ga})
    
errors.to_csv(join(dstDir,'modelComp_MAE.csv'),index=False)
fig=plt.figure()
errors.boxplot()
plt.ylabel('MAE')
fig.savefig(join(saveFigDir,'modelComp_MAE.pdf'))
#===simulation and model fitting===
    
    
    
    