from scipy.special import lambertw
from scipy.sparse.linalg import norm

from ..utils import compute_D_and_lam,computeLoss

import numpy as np
import gc
from time import ctime
from os.path import join
import os
import pandas as pd
           
                
def prox_f(v,o2,rho):
    u=v-rho
    return np.real(lambertw( rho * o2 * np.exp(-u) )) + u

def prox_g(v,lam_vec):    
    v1=np.abs(v)-lam_vec
    v1[v1<0]=0
    return v1*np.sign(v)

def warmStart(i_t,i_s,lam_t_vec,lam_s_vec,mu,c_D,r_D,destDir): 
    fn=None
    if i_s!=0:
        fn='mu_'+str(mu)+'_lam_t_'+str(lam_t_vec[i_t])+\
           '_lam_s_'+str(lam_s_vec[i_s-1])
    elif (i_s==0) and (i_t!=0):
        fn='mu_'+str(mu)+'_lam_t_'+str(lam_t_vec[i_t-1])+\
           '_lam_s_'+str(lam_s_vec[i_s])
    elif (i_s==0) and (i_t==0):
        #check destDir to see if there is any results saved
        #there which can be used to warm start.
                
        files=os.listdir(destDir)#get all files in destDir
        if len(files)==0:
            #no fitted values for warmstart
            return [],[],[],0#X,Z,U,ifWarmStart            
        else:
            fitted_lam_t=np.array([float(s[(s.find('lam_t')+6):\
                                         s.find('_lam_s')]) for s in files \
                                         if s.find('lam_t')!=-1])
            fitted_lam_s=np.array([float(s[(s.find('lam_s')+6):]) \
                                   for s in files if s.find('lam_t')!=-1])
            candidate_idx=(fitted_lam_t<=lam_t_vec[i_t]) & \
                          (fitted_lam_s<=lam_s_vec[i_s])
            if candidate_idx.sum()==0:
                #no appropriate fitted values for warmstart
                return [],[],[],0#X,Z,U,ifWarmStart  
            else:
                #choose the one with the closest lam_t and then lam_s
                df=pd.DataFrame({'lam_t':fitted_lam_t[candidate_idx],
                                 'lam_s':fitted_lam_s[candidate_idx],
                                 'dist_t':np.abs(lam_t_vec[i_t]-\
                                                 fitted_lam_t[candidate_idx]),
                                 'dist_s':np.abs(lam_s_vec[i_s]-\
                                                 fitted_lam_s[candidate_idx])})
                df=df.sort_values(by=['dist_t','dist_s']).reset_index()

                fn='mu_'+str(mu)+'_lam_t_'+str(df.lam_t[0])+\
                   '_lam_s_'+str(df.lam_s[0])
    
    #===load fitted values===       
    print('warm starting from %s'%fn)
    
    X=np.fromfile(join(destDir,'X_'+fn),'float64').reshape((c_D,1))
    Z=np.fromfile(join(destDir,'Z_'+fn),'float64').reshape((r_D,1))
    U=np.fromfile(join(destDir,'U_'+fn),'float64').reshape((r_D,1)) 
    return X,Z,U,1
    #===load fitted values===

def initializeWithMeanVar(o2_mat):
    n_r,n_c=o2_mat.shape
    X=np.log(np.mean(o2_mat,1)).reshape((n_r,1)).dot(np.ones((1,n_c)))
    return X.reshape((o2_mat.size,1))


def linearizedADMM_fit(dataMat,destDir,metadata,
                       lam_t_vec,lam_s_vec,mu=.01,
                       maxIter=40000,freq=100,
                       ifWarmStart=True,lh_trend=True,
                       earlyStopping=True,patience=2,tol=.1):
    
    #===reshape data===
    o=dataMat.reshape((dataMat.size,1))
    o2=o**2
    del dataMat;gc.collect()
    #===reshape data===
    
    #===metadata===
    n_rows,n_cols,T=(metadata['n_rows'],metadata['n_cols'],metadata['T'])    
    #===metadata===
    
    #===make sure the parameters are float===
    lam_t_vec=[float(lam_t) for lam_t in lam_t_vec]
    lam_s_vec=[float(lam_s) for lam_s in lam_s_vec]
    mu=float(mu)
    #===make sure the parameters are float===   
    
    #===compute D and lam_vec===
    D,lam_vec=compute_D_and_lam(n_rows,n_cols,T,lam_t,lam_s,lh_trend=lh_trend)
    r_D,c_D=D.shape
    D_norm=norm(D)    
    lam=1.*D_norm*mu
    scaled_lam=lam*lam_vec    
    #===compute D and lam_vec===

    #===compute solution for all lam_t and lam_s=== 
    for i_t,lam_t in enumerate(lam_t_vec):
        for i_s,lam_s in enumerate(lam_s_vec):
            print('\014')
            print('\n'+ctime()+'...fitting model with lam_t=%.2f, lam_s=%.2f'%\
                  (lam_t,lam_s))
            
            #results filename 
            result_fn = 'mu_'+str(mu)+'_lam_t_'+str(lam_t)+\
                    '_lam_s_'+str(lam_s)
                                
            #-initialization-
            ifWarmStartFlag=0
            if ifWarmStart:
                X,Z,U,ifWarmStartFlag=warmStart(i_t,i_s,lam_t_vec,lam_s_vec,
                                                mu,c_D,r_D,destDir)                
            if (ifWarmStart==0) or (ifWarmStartFlag==0):
                X=initializeWithMeanVar(o2.reshape((n_cols*n_rows,T)))
                Z=np.zeros((r_D,1));U=np.zeros((r_D,1));    
        
            Dx=D*X        
            totalLoss=[]
            totalLoss.append(computeLoss(X,o2,D,lam_vec))
            
            print('time                           iteration       loss')     
            print(ctime()+'          %5.0f          %.1f'\
                  %(0,totalLoss[-1]))    
            #-initialization-
            
            #---ADMM loop---
            for it in range(maxIter):
                X_= X-(mu/lam)*D.transpose()*(Dx-Z+U)
                
                X = prox_f(X_,o2,mu)
                
                Dx=D*X
                DxU=Dx+U
                Z = prox_g(DxU,scaled_lam)
                U = DxU-Z
                                
                if  (it+1)%freq==0:
                    totalLoss.append(computeLoss(X,o2,D,lam_vec))
                    print(ctime()+'          %5.0f          %.1f'\
                          %(it+1,totalLoss[-1]))
                    if earlyStopping and (len(totalLoss)>patience):
                        chng=100*np.abs(totalLoss[-patience]-\
                                       totalLoss[-patience:])/\
                                        totalLoss[-patience]
                        #if change in totalLoss is less than tol%
                        if np.max(chng)<tol:
                            break
            #---ADMM loop---             
            
            #---save results---
            print(ctime()+'...saving results...')
            
            X.tofile(join(destDir,'X_'+result_fn))
            Z.tofile(join(destDir,'Z_'+result_fn))
            U.tofile(join(destDir,'U_'+result_fn))
            totalLoss=np.vstack((freq*np.arange(len(totalLoss)),
                                 np.array(totalLoss)))
            totalLoss.tofile(join(destDir,'loss_'+result_fn))               
            #---save results---    
    #===compute solution for all lam_t and lam_s===
    
    
    
    
    
    
    