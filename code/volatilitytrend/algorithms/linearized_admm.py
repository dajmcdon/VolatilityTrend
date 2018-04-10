from scipy.special import lambertw
from scipy.sparse import spdiags,csr_matrix
from scipy.sparse.linalg import norm

import numpy as np
import gc
from time import ctime
from os.path import join
import os
import pandas as pd

def compute_D_and_lam(n_rows,n_cols,T,lam_t,lam_s,lh_trend=True,
                        ifCompute_Gh=False,wrapAround=False):
    '''
    This computes the matrix D and vector lam_vec.
    '''
    
    grid_size=n_rows*n_cols
    if grid_size>200:
        print(ctime()+'...computing optimization matrices...')

        
    r_s=T*(2*n_rows*n_cols-n_rows-n_cols);#no. of spatial constraints
    r_t=n_rows*n_cols*(T-2);#no. of temporal constraints
    
    #===form matrix D===
    #---spatial penalty---
    I_s=[];J_s=[];x_s=[];

    for c in range(n_cols):
        for r in range(n_rows):
            
            #---determine the neighbors of the current point---
            if ((r<(n_rows-1)) & (c<(n_cols-1))):
                r_n=[r+1,r];c_n=[c,c+1]
            elif ((r==(n_rows-1)) & (c<(n_cols-1))):
                r_n=[r];c_n=[c+1]
            elif (c==(n_cols-1)):
                if wrapAround:
                    r_n=[r+1,r];c_n=[c,0]
                else:
                    r_n=[r+1];c_n=[c]
                if (r==(n_rows-1)):
                    continue
                    
            idx_n=np.ravel_multi_index((r_n,c_n),
                                       dims=(n_rows,n_cols),order='F')

            idx=np.ravel_multi_index((r,c),
                                       dims=(n_rows,n_cols),order='F')               
            #---determine the neighbors of the current point---
            
            #---add indices corresponding to current point and its neighbors---
            for i in idx_n:
                I_s.append(len(I_s)*T+np.tile(np.arange(T),(1,2)))
                J_s.append(np.hstack([np.arange(idx*T,(idx+1)*T),
                                        np.arange(i*T,(i+1)*T)]))
                x_s.append(np.hstack([np.ones(T),-1*np.ones(T)]))
            #---add indices corresponding to current point and its neighbors---           
            
    I_s=np.hstack(I_s).flatten();J_s=np.hstack(J_s).flatten();
    x_s=np.hstack(x_s).flatten();
    #---spatial penalty---

    #---temporal penalty---   
    m=T-2;p=grid_size
    ind=np.arange(m)
    I0=np.tile(ind,(1,3))
    J0=np.concatenate((ind,ind+1,ind+2)).reshape((3*ind.size))
    x_t=np.tile(np.concatenate((np.ones((1,m)),
                      -2*np.ones((1,m)),
                      np.ones((1,m))),axis=1),p).flatten()
    
    #-long-horizon penalty-
    if lh_trend:
        n_year=T/52
        I_lh0=[];J_lh0=[];x_lh0=[]
        for i in range(n_year-2):
            I_lh0.append([i]*3*52)
            J_lh0.append(np.arange(i*52,i*52+3*52))
            x_lh0.append([1.0]*52+[-2.0]*52+[1.0]*52)
        I_lh0=np.concatenate(I_lh0);J_lh0=np.concatenate(J_lh0);
        x_lh=np.tile(np.concatenate(x_lh0),(p,))
        I_lh=[I_lh0];J_lh=[J_lh0];
    #-long-horizon penalty-
    
    I_t=[I0];J_t=[J0];    
    for pp in range(p-1):
        I_t.append(I0+(pp+1)*m)
        J_t.append(J0+(pp+1)*T)
        if lh_trend:
            I_lh.append(I_lh0+(n_year-2)*(pp+1))
            J_lh.append(J_lh0+T*(pp+1))
    I_t=np.hstack(I_t).flatten();J_t=np.hstack(J_t).flatten();
    #---temporal penalty---
    
    if lh_trend:
        r_lh=(n_year-2)*p#this is used in computing h below
        I_lh=np.hstack(I_lh);J_lh=np.hstack(J_lh);
        I=np.hstack([I_t,I_s+r_t,I_lh+r_s+r_t]);
        J=np.hstack([J_t,J_s,J_lh]);
        x=np.hstack([x_t,x_s,x_lh]);        
        D = csr_matrix((x.flatten(),(I.flatten(),J.flatten())),
                       shape=(r_s+r_t+r_lh,T*grid_size))        
    else:                
        I=np.hstack([I_t,I_s+r_t]);J=np.hstack([J_t,J_s]);
        x=np.hstack([x_t,x_s]);
        D = csr_matrix((x.flatten(),(I.flatten(),J.flatten())),
                       shape=(r_s+r_t,T*grid_size))
        r_lh=0
    #===form matrix D===
    
    lam_vec=np.vstack( (lam_t*np.ones((r_t,1)),lam_s*np.ones((r_s,1)),
                         lam_t*np.ones((r_lh,1))) )
    

    return D,lam_vec
           
                
def prox_f(v,o2,rho):
    u=v-rho
    return np.real(lambertw( rho * o2 * np.exp(-u) )) + u

def prox_g(v,lam_vec):    
    v1=np.abs(v)-lam_vec
    v1[v1<0]=0
    return v1*np.sign(v)

def computeLoss(X,o2,D,lam_vec):
    l= np.sum(X + o2*np.exp(-X)) + np.sum(lam_vec*np.abs(D.dot(X)))
    return l

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
            files=filter(lambda x: 'X' in x,files)
            fitted_lam_t=np.array([float(s[(s.find('lam_t')+6):\
                                         s.find('_lam_s')]) for s in files])
            fitted_lam_s=np.array([float(s[(s.find('lam_s')+6):]) \
                                   for s in files])
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
                   
    print(fn)                       
    X=np.fromfile(join(destDir,'X_'+fn),'float64').reshape((c_D,1))
    Z=np.fromfile(join(destDir,'Z_'+fn),'float64').reshape((r_D,1))
    U=np.fromfile(join(destDir,'U_'+fn),'float64').reshape((r_D,1)) 
    return X,Z,U,1


def initializeWithMeanVar(o2_mat):
    n_r,n_c=o2_mat.shape
    X=np.log(np.mean(o2_mat,1)).reshape((n_r,1)).dot(np.ones((1,n_c)))
    return X.reshape((o2_mat.size,1))



def linearizedADMM_fit(dataMat,destDir,
                       metadata,lam_t_vec,lam_s_vec,mu=.01,
                       maxIter=40000,freq=100,
                       ifWarmStart=False,lh_trend=True,
                       earlyStopping=True,patience=2,tol=.1):
    
    #===reshape data===
    o=dataMat.reshape((dataMat.size,1))
    o2=o**2
    del dataMat;gc.collect()
    #===reshape data===
    
    #===metadata===
    nrows,ncols,T=(metadata['nrows'],metadata['ncols'],metadata['T'])    
    #===metadata===
    
    #===make sure the parameters are float===
    lam_t_vec=[float(lam_t) for lam_t in lam_t_vec]
    lam_s_vec=[float(lam_s) for lam_s in lam_s_vec]
    mu=float(mu)
    #===make sure the parameters are float===   
    
    #===compute D and lam_vec===
    D,lam_vec=compute_D_and_lam(nrows,ncols,T,lam_t,lam_s,lh_trend=lh_trend)
    r_D,c_D=D.shape
    D_norm=norm(D)    
    lam=1.*D_norm*mu
    scaled_lam=lam*lam_vec    
    #===compute D and lam_vec===

    #===compute solution for all lam_t and lam_s=== 
    for i_t,lam_t in enumerate(lam_t_vec):
        for i_s,lam_s in enumerate(lam_s_vec):
            
            #results filename 
            result_fn = 'mu_'+str(mu)+'_lam_t_'+str(lam_t)+\
                    '_lam_s_'+str(lam_s)
                                
            #-initialization-
            if ifWarmStart:
                X,Z,U,ifWarmStart=warmStart(i_t,i_s,lam_t_vec,lam_s_vec,mu,
                                     c_D,r_D,destDir)
            if ifWarmStart==0:
                X=initializeWithMeanVar(o2.reshape((ncols*nrows,T)))
                Z=np.zeros((r_D,1));U=np.zeros((r_D,1));    
        
            Dx=D*X        
            totalLoss=[]
            totalLoss.append(computeLoss(X,o2,D,lam_vec))
            print('\014')
            print('time                           iteration       loss')     
            print(ctime()+'          %5.0f          %.1f'\
                  %(0,totalLoss[-1]))    
            #-initialization-
            
            #---ADMM loop---
            
            #---ADMM loop---
    
    #===compute solution for all lam_t and lam_s===
    
    
    
    
    
    
    