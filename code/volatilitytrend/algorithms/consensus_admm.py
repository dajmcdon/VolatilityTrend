from scipy.special import lambertw
from cvxopt import solvers,matrix,spmatrix,sparse,spdiag,div,mul,exp

import numpy as np
import gc
from time import ctime
from os.path import join


def compute_cp_matrices(n_rows,n_cols,T,lam_t,lam_s,lh_trend=True,
                        ifCompute_Gh=False,wrapAround=True):
    '''
    This computes the matrices used in cp optimization.
    '''
    
    grid_size=n_rows*n_cols
    if grid_size>200:
        print(ctime()+'...computing optimization matrices...')

    if wrapAround:#no. of spatial constraints
        r_s=T*(2*n_rows*n_cols-n_rows);
    else:
        r_s=T*(2*n_rows*n_cols-n_rows-n_cols);
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
        D=spmatrix(x,I,J,size=(r_s+r_t+r_lh,T*grid_size))        
    else:                
        I=np.hstack([I_t,I_s+r_t]);J=np.hstack([J_t,J_s]);
        x=np.hstack([x_t,x_s]);
        D=spmatrix(x,I,J,size=(r_s+r_t,T*grid_size))
        r_lh=0
    #===form matrix D===
    
    #===form matrix G,h===
    if ifCompute_Gh:
        r_D=D.size[0];c_D=D.size[1]
        if grid_size>200:
            print('\t'+ctime()+'...computing G...')    
        G=sparse([-D.T,spdiag([1.0]*r_D),spdiag([-1.0]*r_D)])
        h=np.atleast_2d(np.hstack(([1.0]*c_D,
                                   [lam_t]*r_t,[lam_s]*r_s,[lam_t]*r_lh,
                                   [lam_t]*r_t,[lam_s]*r_s,[lam_t]*r_lh))).\
                                   transpose()
        h=matrix(h)
        return D,G,h
    else:
        return D
    #===form matrix G,h===

def computeLoss(Z,Y,D,lam_t,lam_s,T,n_rows,n_cols,lh_trend):
    '''
    This function evaluates the objective function at a given solution
    Z.
    '''
        
    n_t=n_rows*n_cols*(T-2);#no. of temporal constraints
    n_lh=0
    if lh_trend:
        n_year=T/52
        n_lh=(n_year-2)*n_rows*n_cols
    n_s=D.size[0]-n_lh-n_t#no. of spatial constraints
    lam_vec=np.array(n_t*[lam_t]+n_s*[lam_s]+n_lh*[lam_t]).reshape((1,-1))
    
    f1=np.sum(np.array(Z+mul(Y**2,exp(-Z))))
    f2=float(lam_vec.dot( np.abs(np.array(D*Z)) ))
    
    return f1+f2

def ComputeLocToGlobVarTransform(n_rows,n_cols,T,n_r,n_c):
    '''
    This function computes a matrix :math:`A` which specifies the relationship
    between the local variables :math:`x_i` and the global variable :math:`z` 
    for a problem in which the data on a grid are devided into several 
    sub-grids. Specifically we have: :math:`z=Ax`, where 
    :math:`x=(x_1,...,x_K)^T` and each :math:`x_i` is the collection of the
    local variables corresponding to the sub-grid :math:`i`.
    
    Parameters
    ---------
    n_rows : integer
        number of rows in the oroginal grid.
    n_cols : integer
        number of columns in the oroginal grid.
    n_r : integer
        number of rows in each sub-grid. 
    n_c : integer
        number of columns in each sub-grid.
        
    Returns
    -------
    A : sparse
        The transformation matrix.
    '''
    
    boundary_rows=np.arange(0,n_rows,n_r-1)[1:-1]
    boundary_cols=np.arange(0,n_cols,n_c-1)[1:-1]
    n_boundary_rows=boundary_rows.size;n_boundary_cols=boundary_cols.size
    n_x_blk=n_r*n_c*T#no. of x variables in each block
    I_vec=[];J_vec=[]

    #===determine block===
    def determineBlock(r,c):
        '''
        This function determines the block :math:`x_i` to which
        a given entity of z belongs. It also returns what row and col
        in that block, the entity is.
        '''
        
        n1=np.sum((r-boundary_rows)>0)
        n2=np.sum((c-boundary_cols)>0)
        blk=n2*(n_boundary_rows+1)+n1#block number
        r_b=r-n1*(n_r-1);c_b=c-n2*(n_c-1)
        return (blk,r_b,c_b)
    #===determine block===    
    
    #===corner points===
    r1=np.tile(boundary_rows,(n_boundary_cols,1)).flatten('F')
    c1=np.tile(boundary_cols,(n_boundary_rows,1)).flatten()
    idx1=np.ravel_multi_index((r1,c1),(n_rows,n_cols),order='F')
    
    #these are the conrner points. So they are always the
    #(n_r,n_c) entery of one block, and the (0,n_c) entry of the block below,
    #(n_r,0) entery of block ont the right,
    #and the (0,0) entry of the block diagonal to the first block.
    for i,idx in enumerate(idx1):
        I=np.tile(np.arange(idx*T,(idx+1)*T),4)
        
        blk1,r_b1,c_b1=determineBlock(r1[i],c1[i])
        idx_in_blk1=np.ravel_multi_index((r_b1,c_b1),(n_r,n_c),order='F')        
        J1=np.arange(blk1*n_x_blk+idx_in_blk1*T,
                     blk1*n_x_blk+(idx_in_blk1+1)*T)
        
        blk2,r_b2,c_b2=(blk1+1,0,n_c-1)
        idx_in_blk2=np.ravel_multi_index((r_b2,c_b2),(n_r,n_c),order='F')
        J2=np.arange(blk2*n_x_blk+idx_in_blk2*T,
                     blk2*n_x_blk+(idx_in_blk2+1)*T)

        blk3,r_b3,c_b3=(blk1+n_boundary_rows+1,n_r-1,0)
        idx_in_blk3=np.ravel_multi_index((r_b3,c_b3),(n_r,n_c),order='F')
        J3=np.arange(blk3*n_x_blk+idx_in_blk3*T,
                     blk3*n_x_blk+(idx_in_blk3+1)*T)

        blk4,r_b4,c_b4=(blk1+n_boundary_rows+2,0,0)
        idx_in_blk4=np.ravel_multi_index((r_b4,c_b4),(n_r,n_c),order='F')
        J4=np.arange(blk4*n_x_blk+idx_in_blk4*T,
                     blk4*n_x_blk+(idx_in_blk4+1)*T)
        
        I_vec.append(I);J_vec.append(np.hstack((J1,J2,J3,J4)))
    #===corner points===
    
    #===row boundary points===
    r2=np.tile(boundary_rows,(n_cols-n_boundary_cols,1)).flatten('F')
    c2=np.tile(np.delete(np.arange(n_cols),boundary_cols),
               (n_boundary_rows,1)).flatten()
    idx2=np.ravel_multi_index((r2,c2),(n_rows,n_cols),order='F')
    
    #these are the points on the row boundaries. So they are always the
    #(n_r,c) entery of one block and the (0,c) entry of the block below.
    for i,idx in enumerate(idx2):
        blk1,r_b1,c_b1=determineBlock(r2[i],c2[i])
        idx_in_blk1=np.ravel_multi_index((r_b1,c_b1),(n_r,n_c),order='F')
        I=np.tile(np.arange(idx*T,(idx+1)*T),2)
        J1=np.arange(blk1*n_x_blk+idx_in_blk1*T,
                     blk1*n_x_blk+(idx_in_blk1+1)*T)
        blk2,r_b2,c_b2=(blk1+1,0,c_b1)
        idx_in_blk2=np.ravel_multi_index((r_b2,c_b2),(n_r,n_c),order='F')
        J2=np.arange(blk2*n_x_blk+idx_in_blk2*T,
                     blk2*n_x_blk+(idx_in_blk2+1)*T)
        
        I_vec.append(I);J_vec.append(np.hstack((J1,J2)))        
    #===row boundary points===
    
    #===column boundary points===    
    r3=np.tile(np.delete(np.arange(n_rows),boundary_rows),
               (n_boundary_cols,1)).flatten()
    c3=np.tile(boundary_cols,(n_rows-n_boundary_rows,1)).flatten('F')
    idx3=np.ravel_multi_index((r3,c3),(n_rows,n_cols),order='F')
    
    #these are the points on the col boundaries. So they are always the
    #(r,n_c) entery of one block and the (r,0) entry of the block to the right.
    for i,idx in enumerate(idx3):
        blk1,r_b1,c_b1=determineBlock(r3[i],c3[i])
        idx_in_blk1=np.ravel_multi_index((r_b1,c_b1),(n_r,n_c),order='F')
        I=np.tile(np.arange(idx*T,(idx+1)*T),2)
        J1=np.arange(blk1*n_x_blk+idx_in_blk1*T,
                     blk1*n_x_blk+(idx_in_blk1+1)*T)
        blk2,r_b2,c_b2=(blk1+n_boundary_rows+1,r_b1,0)
        idx_in_blk2=np.ravel_multi_index((r_b2,c_b2),(n_r,n_c),order='F')
        J2=np.arange(blk2*n_x_blk+idx_in_blk2*T,
                     blk2*n_x_blk+(idx_in_blk2+1)*T)   
        
        I_vec.append(I);J_vec.append(np.hstack((J1,J2))) 
    #===column boundary points===
    
    #===non-boundary points===
    idx4=np.delete(np.arange(n_cols*n_rows),np.hstack((idx1,idx2,idx3)))
    r4,c4=np.unravel_index(idx4,(n_rows,n_cols),'F')
    for i,idx in enumerate(idx4):
        blk,r_b,c_b=determineBlock(r4[i],c4[i])
        idx_in_blk=np.ravel_multi_index((r_b,c_b),(n_r,n_c),order='F')
        I=np.arange(idx*T,(idx+1)*T)
        J=np.arange(blk*n_x_blk+idx_in_blk*T,blk*n_x_blk+(idx_in_blk+1)*T)
        
        I_vec.append(I);J_vec.append(J) 
    
    I_vec=np.hstack(I_vec);J_vec=np.hstack(J_vec)
    A=spmatrix(np.ones(I_vec.size),I_vec,J_vec,
               size=(I_vec.max()+1,J_vec.max()+1))        
    #===non-boundary points===
    
    return A



def compute_primal(Y,v,D,Kinv,alpha,rho):
    u = -D.T*v #compute u 
    Y2=mul(Kinv,Y**2)
    z1=u-Kinv+rho*alpha       
    W=matrix(np.real( lambertw(mul((Y2/rho),exp(-z1/rho))) ))
    beta= W +(z1/rho)

    return beta
    
def x_update(Y,D,G,h,Kinv,alpha,rho):
    '''
    This function performs the x-update step of the ADMM algorithm.
    See Boyd et al, 2010, page 55.
    '''    

    m,n=D.size
    ki=matrix(Kinv)
    
    def F(v=None,z=None):           
        if v is None: return 0,matrix(0.0,(m,1))
        
        u = -D.T*v #compute u
        
        #===define some auxilary variables===
        Y2=mul(ki,Y**2)
        z1=u-ki+rho*alpha    
        
        W=matrix(np.real( lambertw(mul((Y2/rho),exp(-z1/rho))) ))
        h_opt= W +(z1/rho)
        dh_to_du=(1/rho)*(div(1,1+W))
        z2=mul(Y2,exp(-h_opt))
        z3=z2+z1-rho*h_opt
        #===define some auxilary variables===
        
        #====compute f===
        f=sum( mul(u,h_opt)-mul(ki,h_opt)-z2-(rho/2)*(h_opt-alpha)**2 )
        #====compute f===
        
        #===compute Jacobian===     
        df_to_du=h_opt+mul(dh_to_du,z3)
        Df = -df_to_du.T*D.T
        if z is None: return f, Df                 
        #===compute Jacobian===    
        
        #===compute Hessian===
        d2h_to_du2=(1/rho**2)*div(W,(1+W)**3)
        d2f_to_du2=mul(d2h_to_du2,z3)+mul(dh_to_du,2-mul(z2+rho,dh_to_du))
        H=D*spdiag(mul(z[0],d2f_to_du2))*D.T
        #===compute Hessian===
        
        return f, Df, H 

    solvers.options['maxiters']=500;solvers.options['show_progress']=False
    sol=solvers.cp(F=F,G=G,h=h)
    v=sol['x'];#dual solution
    x=compute_primal(Y,v,D,matrix(Kinv),alpha,rho)#primal solution
    
    return x

def scale_h(h,K,lh_trend,T,nr_blk,nc_blk,r_D,c_D):
    '''
    Let D (the matrix in generalized lasso) have size (r_Dxc_D).
    We have h^T=[1_(c_D)|L_t|L_s|L_lh|L_t|L_s|L_lh] where:
        1_(c_D) is a (c_Dx1) vector of ones. c_D is the no. of columns of D.
        L_t=lam_t*1_(nt) where nt=(T-2)*nr_blk*nc_blk is the number of 
        temporal constraints.
        L_s=lam_s*1_(ns) where ns=r_D-nt is the number of spatial constraints.
        L_lh=lam_t*1_(nlh) where nlh=n_years*nr_blk*nc_blk is the number
        of long horizon constraints. If lh_trend is False `h` does not 
        include this part.
    This function repalces each element of L_t by lam_t/k_(i,j) where 
    k_(i,j) is the number of 
    local variables corresponding to each global variable at position (i,j)
    on the grid. Note that there are T-2 temporal constraints corresponding 
    to each position (i,j) on the grid and k_(i,j) is the same for all 
    of them. 
    '''

    n_t=nr_blk*nc_blk*(T-2);
    K=K[0::T]
    K1=matrix(np.tile(K,(T-2,1)).reshape((n_t,1),order='F'))
    h[c_D:(c_D+n_t)]=mul(h[c_D:(c_D+n_t)],K1)
    
    if lh_trend:
        n_year=T/52
        n_lh=(n_year-2)*nr_blk*nc_blk;n_s=r_D-n_lh-n_t
        K2=matrix(np.tile(K,(n_year-2,1)).reshape((n_lh,1),order='F'))
        h[(c_D+n_t+n_s):(c_D+n_t+n_s+n_lh)]=\
            mul(h[(c_D+n_t+n_s):(c_D+n_t+n_s+n_lh)],K2)
        h[(c_D+r_D):(c_D+r_D+n_t)]=mul(h[(c_D+r_D):(c_D+r_D+n_t)],K1)
        h[(h.size[0]-n_lh):]=mul(h[(h.size[0]-n_lh):],K2)
    else:
        h[(c_D+r_D):(c_D+r_D+n_t)]=mul(h[(c_D+r_D):(c_D+r_D+n_t)],K1)
    
    return h


def consensusADMM_fit(dataMat,destDir,metadata,
                      lam_t_vec,lam_s_vec,rho=.1,
                      n_r_b=2,n_c_b=2,
                      maxIter=1000,freq=100,
                      lh_trend=True,wrapAround=True,
                      earlyStopping=True,patience=2,tol=.1):

    #===metadata===
    n_rows,n_cols,T=(metadata['n_rows'],metadata['n_cols'],metadata['T'])    
    #===metadata===
    
    #===check the compatability of grid size with sub-blocks size===
    if (((n_rows-1)%(n_r_b-1)!=0) or ((n_cols-1)%(n_c_b-1)!=0)):
        msg='In current implementation, n_rows-1 and n_cols-1 should be'+\
        'divisive to (n_r_b-1) and (n_c_b)-1, respectively.'
        raise ValueError(msg)
    #===check the compatability of grid size with sub-blocks size===
    
    #===convert to cvxopt matrix===
    Y=matrix(np.asarray(dataMat,dtype='float64').flatten())
    y_flat=dataMat.flatten()
    del dataMat;gc.collect()
    #===convert to cvxopt matrix===
    
    #===partition data===
    n_x_b=n_r_b*n_c_b*T#no. of local variables in each block
    A=ComputeLocToGlobVarTransform(n_rows,n_cols,T,n_r_b,n_c_b)
    n_z,n_x=A.size
    
    I=np.array(A.I).flatten();J=np.array(A.J).flatten()
    y_partitioned=np.zeros(n_x)
    y_partitioned[J]=y_flat[I]
    del y_flat;gc.collect()
    n_blocks=y_partitioned.size/n_x_b
    y_partitioned=y_partitioned.reshape((n_x_b,n_blocks),order='F')
    #each column of y_partitioned contains the data of a block                                       
    #===partition data===
    
    #===compute K and Kinv===
    #K is no. of local variables corresponding each global variable and 
    #Kinv is 1/K
    K=A*matrix(np.ones(A.size[1]))
    ki=np.array(div(1.,K)).flatten()
    Kinv=np.ones(n_x)
    Kinv[J]=ki[I]
    Kinv=Kinv.reshape((n_x_b,n_blocks),order='F')
    Kinv=np.ones(Kinv.shape)#???delete this
    del ki;gc.collect()
    #===compute no. of local variables corresponding each global variable===
    
    #===compute D===
    D=compute_cp_matrices(n_rows,n_cols,T,[],[],
                          wrapAround=wrapAround,lh_trend=lh_trend,
                          ifCompute_Gh=False)
    #===compute D===
    
    #===make sure the parameters are float===
    lam_t_vec=[float(lam_t) for lam_t in lam_t_vec]
    lam_s_vec=[float(lam_s) for lam_s in lam_s_vec]
    rho=float(rho)
    #===make sure the parameters are float=== 

    #===compute solution for all lam_t and lam_s=== 
    for i_t,lam_t in enumerate(lam_t_vec):
        for i_s,lam_s in enumerate(lam_s_vec):            
            print('\014')
            print('\n'+ctime()+'...fitting model with lam_t=%.2f, lam_s=%.2f'%\
                  (lam_t,lam_s))
            
            #===initialize X,Z,W===        
            Z_hat=np.zeros((n_x_b,n_blocks))
            W=np.zeros((n_x_b,n_blocks));X=np.zeros((n_x_b,n_blocks))
            totalLoss=[]
            #===initialize X,Z,W===
            
            #===compute D,G,h for each block===
            #'b' means that these are computed for each block
            Db,Gb,hb=compute_cp_matrices(n_r_b,n_c_b,T,lam_t,lam_s,
                                         ifCompute_Gh=True,wrapAround=False,
                                         lh_trend=lh_trend)
            m_b,n_b=Db.size
            #===compute D,G,h for each block===
            
            #results filename 
            result_fn = 'rho_'+str(rho)+'_lam_t_'+str(lam_t)+\
                    '_lam_s_'+str(lam_s)
            
                        
            print('time                           iteration       loss')      

            #---ADMM loop---
            for it in range(maxIter):                  
                #---x-update---
                alpha=Z_hat-W
                for blk in range(n_blocks):
                    hb=scale_h(hb,Kinv[:,blk],True,T,n_r_b,n_c_b,m_b,n_b)
                    alpha_b=matrix(alpha[:,blk])
                    Y_b=matrix(y_partitioned[:,blk])
                    X[:,[blk]]=np.array(x_update(Y_b,Db,Gb,hb,Kinv[:,blk],
                                                 alpha_b,rho))
                
                X_vec=matrix(X.flatten(order='F'))
                #---x-update---
                
                #---z-update---
                Z=div(A*X_vec,K)
                #---z-update---
                
                #---w-update---
                Z_hat=np.zeros(n_x)
                Z_hat[J]=np.array(Z).flatten()[I]
                Z_hat=Z_hat.reshape((n_x_b,n_blocks),order='F') 
                
                W=W+X-Z_hat
                #---w-update---
                
                if ((it==0) or ((it+1)%freq==0)):
                    totalLoss.append(computeLoss(Z,Y,D,lam_t,lam_s,
                                                 T,n_rows,n_cols,lh_trend))
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
            W.tofile(join(destDir,'W_'+result_fn))
            np.array(Z).tofile(join(destDir,'Z_'+result_fn))
            Z_hat.tofile(join(destDir,'Zhat_'+result_fn))
            totalLoss=np.vstack((freq*np.arange(len(totalLoss)),
                                 np.array(totalLoss)))
            totalLoss.tofile(join(destDir,'loss_'+result_fn))               
            #---save results--- 
    
    
