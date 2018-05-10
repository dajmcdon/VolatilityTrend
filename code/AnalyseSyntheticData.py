import numpy as np
from volatilitytrend.data_utils.simulate_data import simulateSpatioTemporalData
from os.path import join
from volatilitytrend.algorithms.base import LinearizedADMM
import matplotlib.pyplot as plt

bl=[False,True]
ifSimulateData=bl[0]
ifFitModel=bl[1]
ifVisualize=bl[0]
ifModelSelection=bl[0]

dstDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/Simulated data'

n_rows=5;n_cols=7;
T=1000#time-series length
gridsize=n_rows*n_cols

###################
###simulate data###
###################
if ifSimulateData:
    
    #===specify sources properties===
    var_sources_centers=np.array([[0,0],[0,5],[3,0],[3,5]])
    var_sources_sigma=[5.]*4
    #===specify sources properties===
    
    #===compute the weight of each source at wach time===
    times=np.arange(0,T)
    var_sources_weight1=np.exp(np.sin(10*np.pi*times/float(T)))
    var_sources_weight2=np.exp(np.sin(10*np.pi*times/float(T)))
    var_sources_weight3=np.exp(np.cos(10*np.pi*times/float(T)))
    var_sources_weight4=np.exp(np.cos(10*np.pi*times/float(T)))
    var_sources_weight_mat=np.concatenate((var_sources_weight1,
                                           var_sources_weight2,
                                           var_sources_weight3,
                                           var_sources_weight4)).\
                                           reshape((4,-1))
    #===compute the weight of each source at wach time===
    
    simulateSpatioTemporalData(dstDir,n_rows,n_cols,T,var_sources_centers,
                                   var_sources_weight_mat,var_sources_sigma)



###################
###Fit model###
###################
lam_t_vec=[0,1,5,10,50,100];lam_s_vec=[0.0,.05,.1,.2,.3];mu=.01
#lam_t_vec=[10];lam_s_vec=[.05];mu=.01

la = LinearizedADMM()#construct linearizedADMM object
data_fn=join(dstDir,'simulated_data')
metadata_fn=join(dstDir,'metadata')
la.loadData(data_fn,metadata_fn)#load data

if ifFitModel:   
    la.fit(dstDir,lam_t_vec,lam_s_vec,maxIter=10000,freq=500,
           lh_trend=False,wrapAround=False,
           ifWarmStart=False,earlyStopping=True,mu=mu,
           ifAdaptMu=False,mu_adapt_rate=.999,mu_adapt_freq=1)


###################
###visualization###
###################

if ifVisualize:
    lam_t=lam_t_vec[0];lam_s=lam_s_vec[0]
    mu,lam_t,lam_s=(float(mu),float(lam_t),float(lam_s))
    fn='mu_{}_lam_t_{}_lam_s_{}'.format(mu,lam_t,lam_s)

    #===load fitted values===
    n_rows=la.metadata['n_rows'];n_cols=la.metadata['n_cols']
    gridsize=n_rows*n_cols
    
    covMat = np.fromfile(join(dstDir,'covMat')).reshape((gridsize,-1))
    filepath=join(dstDir,'X_'+fn)    
    la.analyseFittedValues(filepath)
    i=3;plt.plot(np.exp(la.fittedVar[i,]/2));plt.plot(covMat[i,:])
    #===load fitted values===
    
    MAE=np.mean(np.abs(np.exp(la.fittedVar/2)-covMat))
    print('MAE is :{:.4f}'.format(MAE))


#####################
###model selection###
#####################

if ifModelSelection: 
    lam_t_vec=[10.];lam_s_vec=[0.0,.05,.1,.2,.3];mu=.01
    
    MAE_mat=np.zeros((len(lam_t_vec),len(lam_s_vec)))
    covMat = np.fromfile(join(dstDir,'covMat')).reshape((gridsize,-1))        
    for i,lam_t in enumerate(lam_t_vec):
        for j,lam_s in enumerate(lam_s_vec):
            fn='mu_{}_lam_t_{}_lam_s_{}'.format(mu,lam_t,lam_s)
            filepath=join(dstDir,'X_'+fn)
            X=np.fromfile(filepath).reshape((gridsize,-1))
            MAE_mat[i,j]=np.mean(np.abs(np.exp(X/2)-covMat))
            
        
    
    
    
