import numpy as np
from volatilitytrend.data_utils.simulate_data import simulateSpatioTemporalData
from os.path import join
from volatilitytrend.algorithms.base import LinearizedADMM
import matplotlib.pyplot as plt

bl=[False,True]
ifSimulateData=bl[0]
ifFitModel=bl[0]
ifVisualize=bl[1]
ifModelSelection=bl[0]

dstDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/Simulated data'

n_rows=5;n_cols=7;
n_yeras=15
T=n_yeras*52#time-series length
gridsize=n_rows*n_cols

def computeChangeInVariance(X,n_rows,n_cols,T):
    '''
    this is also implemented in 'analyseFittedValues' but there 
    it is computed for log(var) instead of variance itself.
    '''    
    n_years=T/52;gridsize=n_rows*n_cols 
    X=X[:,0:n_years*52].reshape((n_years*gridsize,52))
    X=np.diff(np.mean(X,axis=1).reshape((gridsize,n_years)),axis=1)
    changeInVar=np.sum(X,axis=1).reshape((n_rows,n_cols),order='F')
    return changeInVar

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
    #===compute the weight of each source at wach time===
    
    simulateSpatioTemporalData(dstDir,n_rows,n_cols,T,var_sources_centers,
                                   var_sources_weight_mat,var_sources_sigma)



###################
###Fit model###
###################
lam_t_vec=[100.];
lam_s_vec=[0.0,.05,.1,.2,.3];
#lam_s_vec=[0.1];
mu=.01
#lam_t_vec=[10];lam_s_vec=[.05];mu=.01

la = LinearizedADMM()#construct linearizedADMM object
data_fn=join(dstDir,'simulated_data')
metadata_fn=join(dstDir,'metadata')
la.loadData(data_fn,metadata_fn)#load data

if ifFitModel:   
    la.fit(dstDir,lam_t_vec,lam_s_vec,maxIter=15000,freq=500,
           lh_trend=True,wrapAround=False,patience=2,
           ifWarmStart=False,earlyStopping=True,mu=mu,
           ifAdaptMu=False,mu_adapt_rate=.999,mu_adapt_freq=1)


###################
###visualization###
###################

if ifVisualize:
    lam_t=lam_t_vec[-1];lam_s=lam_s_vec[2]
    lam_t=5.;lam_s=.2;
    
    mu,lam_t,lam_s=(float(mu),float(lam_t),float(lam_s))
    fn='mu_{}_lam_t_{}_lam_s_{}'.format(mu,lam_t,lam_s)

    #===load fitted values===
    n_rows=la.metadata['n_rows'];n_cols=la.metadata['n_cols']
    gridsize=n_rows*n_cols
    
    covMat = np.fromfile(join(dstDir,'covMat')).reshape((gridsize,-1))
    filepath=join(dstDir,'X_'+fn)    
    la.analyseFittedValues(filepath)
    plt.figure()
    i=0;plt.plot(np.exp(la.fittedVar[i,]/2));plt.plot(covMat[i,:])
    #===load fitted values===
    
    #===compute mean absolute error between true and estimated variance===
    MAE=np.mean(np.abs(np.exp(la.fittedVar/2)-covMat))
    print('MAE is :{:.4f}'.format(MAE))
    #===compute mean absolute error between true and estimated variance===
    
    #===compute change in variance===
    true_changeInVar = computeChangeInVariance(covMat,
                                               n_rows,n_cols,T)
    est_changeInVar = computeChangeInVariance(np.exp(la.fittedVar/2),
                                              n_rows,n_cols,T)
    
    err_in_changeInVar=100.*np.abs(true_changeInVar-est_changeInVar)/\
                                    np.abs(true_changeInVar)
    print('mean error in estimated variance trend is:{:.2f} %'.\
          format(np.median(err_in_changeInVar))) 
    f=plt.figure(figsize=(8,4))
    f.add_subplot(1,2,1)
    plt.pcolor(true_changeInVar)
    f.add_subplot(1,2,2)
    plt.pcolor(est_changeInVar)
    #===compute change in variance===

#####################
###model selection###
#####################

if ifModelSelection: 
    lam_t_vec=[0.0,1.,5.,10.,50.,100.];
    lam_s_vec=[0.0,.05,.1,.2,.3];
    mu_vec=[.01]*3+[.001]*2+[.0001]
    
    MAE_mat=np.zeros((len(lam_t_vec),len(lam_s_vec)))
    covMat = np.fromfile(join(dstDir,'covMat')).reshape((gridsize,-1))        
    for i,lam_t in enumerate(lam_t_vec):
        for j,lam_s in enumerate(lam_s_vec):
            mu=mu_vec[i]
            fn='mu_{}_lam_t_{}_lam_s_{}'.format(mu,lam_t,lam_s)
            filepath=join(dstDir,'X_'+fn)
            X=np.fromfile(filepath).reshape((gridsize,-1))
            MAE_mat[i,j]=np.mean(np.abs(np.exp(X/2)-covMat))


    y,x=np.meshgrid(np.log1p(lam_s_vec),
                    np.log1p(lam_t_vec))
    plt.figure()
    plt.pcolor(MAE_mat)
    
    i,j=np.unravel_index(np.argmin(MAE_mat),MAE_mat.shape)
    print('best parameters: lam_t={} , lam_s={}'.format(lam_t_vec[i],
                                                        lam_s_vec[j]))
        
    
    
    
