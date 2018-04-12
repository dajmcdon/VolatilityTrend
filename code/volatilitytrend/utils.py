from scipy.sparse import csr_matrix
from time import ctime
import numpy as np

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

def computeLoss(X,o2,D,lam_vec):
    l= np.sum(X + o2*np.exp(-X)) + np.sum(lam_vec*np.abs(D.dot(X)))
    return l

def latlon_to_rowcol(lat,lon,lats,lons,n_rows,n_cols):
    
    lats=lats.reshape((n_rows,n_cols),order='F')
    lons=lons.reshape((n_rows,n_cols),order='F')
    if lon<0:
        lon=lon+360

    row=np.argmin(np.abs(lats[:,0]-lat))
    col=np.argmin(np.abs(lons[0,:]-lon))
    idx=np.ravel_multi_index((row,col),lats.shape,order='F')
    return row,col,idx