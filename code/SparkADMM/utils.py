import numpy as np
from cvxopt import matrix,spmatrix,sparse,spdiag
from os.path import join
import os
from cPickle import load
from time import ctime
import shutil

def downloadS3Dir(s3,bucketname,srcDir,desDir):
    '''
    This function downloads all files in a directory on S3 into a folder on
    local computer.
    '''
    
    srcDir=srcDir.rstrip('/');srcDir=srcDir.lstrip('/');
    bucket=s3.Bucket(bucketname)
    objs = bucket.objects.filter(Prefix=srcDir)
    l=len(srcDir)
    if os.path.exists(desDir):
       shutil.rmtree(desDir)       
    os.makedirs(desDir)
   
    for obj in objs:        
        if not '$folder$' in obj.key:
            filepath=obj.key[l+1:]
            if len(filepath.split('/'))>1:
                curr_dir=join(desDir,'/'.join(filepath.split('/')[:-1]))
                if not os.path.exists(curr_dir):
                    os.makedirs(curr_dir)
            bucket.download_file(obj.key,join(desDir,filepath))
       

def loadData(dataDir,weekly_avg=True):
    '''
    This function loads the data from a file into memory.
    Data should be saved as binary files. 
    
    Parameters
    ---------- 
      
    dataDir : str
        Directory where the data is saved.
        
    weekly_avg: boolean
        if True (default) use the weekly averaged data. In this case
        the folder ``dataDir`` should contain 5 binary files: 
        ``detrended_avg_data``, ``dates_avg``, ``detrended_avg_n_dim``,
        ``detrended_lats``, ``detrended_lons``.\n
        If False use the daily data. In this case
        the folder ``dataDir`` should contain the same 5 binary files
        but without '_avg' in their name.

    Returns
    -------

    data: dict
        It will contain the following fields: ``dataMat``,``n_rows``,
        ``n_cols``,``dates``,``lats``,``lons``.
    '''
    data={}
    if weekly_avg:            
        I=np.fromfile(join(dataDir,'detrended_avg_data'),dtype='float32')
        J=np.fromfile(join(dataDir,'detrended_avg_n_dim'),dtype='float32')
        data['dates']=load(open(join(dataDir,'dates_avg'))) 
    else:#load detrended data            
        I=np.fromfile(join(dataDir,'detrended_data'),dtype='float32')
        J=np.fromfile(join(dataDir,'raw_n_dim'),dtype='float32')
        data['dates']=load(open(join(dataDir,'dates'))) 
        
    data['n_rows']=int(J[0]);data['n_cols']=int(J[1]);data['T']=int(J[2]);
    lats=np.fromfile(join(dataDir,'detrended_lats'),dtype='float32')
    lons=np.fromfile(join(dataDir,'detrended_lons'),dtype='float32')
    grid_size=lons.size
    data['dataMat']=I.reshape((grid_size,
                            I.shape[0]/grid_size))
    data['lats']=lats.reshape((grid_size,1))
    data['lons']=lons.reshape((grid_size,1))  
                  
    return data



def resize_data(n_rows,n_cols,nr_blk,nc_blk):
    '''
    This function computes which rows and columns of the original data
    should be deleted so the number of rows and columns is divisible to 
    n_row_blk,n_col_blk. For example if n_rows=n_cols=6 and nr_blk,nc_blk=3
    then the last row and column should be dropped. It assumes that the data
    is in a (n_rows x n_cols) x T matrix and the first n_rows row contain the 
    time-series of the points in the first column of the grid and so on.
    It returns the index of the rows which should be kept.
    '''
    
    last_r=np.arange(0,n_rows,nr_blk-1)[-1]
    last_c=np.arange(0,n_cols,nc_blk-1)[-1]    
    x,y=np.meshgrid(range(last_r+1),range(last_c+1))
    idx=np.ravel_multi_index((x.flatten(),y.flatten()),
                             dims=(n_rows,n_cols),order='F')
    n_rows=last_r+1;n_cols=last_c+1
    return idx,n_rows,n_cols




def ComputeLocToGlobVarTransform(n_rows,n_cols,T,nr_blk,nc_blk):
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
    nr_blk : integer
        number of rows in each sub-grid. 
    nc_blk : integer
        number of columns in each sub-grid.
        
    Returns
    -------
    A : sparse
        The transformation matrix.
    '''
    
    boundary_rows=np.arange(0,n_rows,nr_blk-1)[1:-1]
    boundary_cols=np.arange(0,n_cols,nc_blk-1)[1:-1]
    n_boundary_rows=boundary_rows.size;n_boundary_cols=boundary_cols.size
    n_x_blk=nr_blk*nc_blk*T#no. of x variables in each block
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
        r_b=r-n1*(nr_blk-1);c_b=c-n2*(nc_blk-1)
        return (blk,r_b,c_b)
    #===determine block===    
    
    #===corner points===
    r1=np.tile(boundary_rows,(n_boundary_cols,1)).flatten('F')
    c1=np.tile(boundary_cols,(n_boundary_rows,1)).flatten()
    idx1=np.ravel_multi_index((r1,c1),(n_rows,n_cols),order='F')
    
    #these are the conrner points. So they are always the
    #(nr_blk,nc_blk) entery of one block, and the (0,nc_blk) entry of the block below,
    #(nr_blk,0) entery of block ont the right,
    #and the (0,0) entry of the block diagonal to the first block.
    for i,idx in enumerate(idx1):
        I=np.tile(np.arange(idx*T,(idx+1)*T),4)
        
        blk1,r_b1,c_b1=determineBlock(r1[i],c1[i])
        idx_in_blk1=np.ravel_multi_index((r_b1,c_b1),(nr_blk,nc_blk),order='F')        
        J1=np.arange(blk1*n_x_blk+idx_in_blk1*T,
                     blk1*n_x_blk+(idx_in_blk1+1)*T)
        
        blk2,r_b2,c_b2=(blk1+1,0,nc_blk-1)
        idx_in_blk2=np.ravel_multi_index((r_b2,c_b2),(nr_blk,nc_blk),order='F')
        J2=np.arange(blk2*n_x_blk+idx_in_blk2*T,
                     blk2*n_x_blk+(idx_in_blk2+1)*T)

        blk3,r_b3,c_b3=(blk1+n_boundary_rows+1,nr_blk-1,0)
        idx_in_blk3=np.ravel_multi_index((r_b3,c_b3),(nr_blk,nc_blk),order='F')
        J3=np.arange(blk3*n_x_blk+idx_in_blk3*T,
                     blk3*n_x_blk+(idx_in_blk3+1)*T)

        blk4,r_b4,c_b4=(blk1+n_boundary_rows+2,0,0)
        idx_in_blk4=np.ravel_multi_index((r_b4,c_b4),(nr_blk,nc_blk),order='F')
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
    #(nr_blk,c) entery of one block and the (0,c) entry of the block below.
    for i,idx in enumerate(idx2):
        blk1,r_b1,c_b1=determineBlock(r2[i],c2[i])
        idx_in_blk1=np.ravel_multi_index((r_b1,c_b1),(nr_blk,nc_blk),order='F')
        I=np.tile(np.arange(idx*T,(idx+1)*T),2)
        J1=np.arange(blk1*n_x_blk+idx_in_blk1*T,
                     blk1*n_x_blk+(idx_in_blk1+1)*T)
        blk2,r_b2,c_b2=(blk1+1,0,c_b1)
        idx_in_blk2=np.ravel_multi_index((r_b2,c_b2),(nr_blk,nc_blk),order='F')
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
    #(r,nc_blk) entery of one block and the (r,0) entry of the block to the right.
    for i,idx in enumerate(idx3):
        blk1,r_b1,c_b1=determineBlock(r3[i],c3[i])
        idx_in_blk1=np.ravel_multi_index((r_b1,c_b1),(nr_blk,nc_blk),order='F')
        I=np.tile(np.arange(idx*T,(idx+1)*T),2)
        J1=np.arange(blk1*n_x_blk+idx_in_blk1*T,
                     blk1*n_x_blk+(idx_in_blk1+1)*T)
        blk2,r_b2,c_b2=(blk1+n_boundary_rows+1,r_b1,0)
        idx_in_blk2=np.ravel_multi_index((r_b2,c_b2),(nr_blk,nc_blk),order='F')
        J2=np.arange(blk2*n_x_blk+idx_in_blk2*T,
                     blk2*n_x_blk+(idx_in_blk2+1)*T)   
        
        I_vec.append(I);J_vec.append(np.hstack((J1,J2))) 
    #===column boundary points===
    
    #===non-boundary points===
    idx4=np.delete(np.arange(n_cols*n_rows),np.hstack((idx1,idx2,idx3)))
    r4,c4=np.unravel_index(idx4,(n_rows,n_cols),'F')
    for i,idx in enumerate(idx4):
        blk,r_b,c_b=determineBlock(r4[i],c4[i])
        idx_in_blk=np.ravel_multi_index((r_b,c_b),(nr_blk,nc_blk),order='F')
        I=np.arange(idx*T,(idx+1)*T)
        J=np.arange(blk*n_x_blk+idx_in_blk*T,blk*n_x_blk+(idx_in_blk+1)*T)
        
        I_vec.append(I);J_vec.append(J) 
    
    I_vec=np.hstack(I_vec);J_vec=np.hstack(J_vec)
    A=spmatrix(np.ones(I_vec.size),I_vec,J_vec,
               size=(I_vec.max()+1,J_vec.max()+1))        
    #===non-boundary points===
    
    return A

def copy_dir_to_s3(s3,src,dst,bucketname,verbose=1):
    '''
    this function copies all files in a directory on the local computer
    into a bucket in s3.
    
    Parameters
    ---------
    
    s3: boto3.resources.factory.s3.ServiceResource
        An s3 object created by boto3.
    
    src: str
        The directory on the local computer which will be copied.
        
    dst: str
        The destination directory in s3 where the source directory will
        be copied into.
        
    bucketname: str
        Bucket name on s3.
        
    verbose: int
        If 1, print which file is being copied now.
        
    Returns
    -------
    filenames: list
        a list of the name of the files which were copied.
    '''
    
    src=src.rstrip('/')
    src_len=len(src)+1
        
#    filenames=[]
    for root,dirs,files in os.walk(src):
        for fn in files:
            filename = join(root,fn)
            f=open(filename,r'rb')
            key=join(dst,root[src_len:],fn)
            if verbose:
                print(ctime()+'...copying file %s' %filename)
            s3.Bucket(bucketname).put_object(Key=key,Body=f)
            f.close()
#            filenames.append(filename)
            
#    return filenames
    
    
def Compute_Z_idx_for_blk(blk,n_rows,n_cols,T,nr_blk,nc_blk):
    '''
    Each local variable x in each block corresponds to a global
    variable Z. This function computes the indices of the global variables
    corresponding to the local variables in a given block number `blk`. It 
    return a numpy.array of size (n_x_b,1) where `n_x_b` is the number 
    of local variables in each block. The i'th entry of this array is the 
    index of the global variable corresponding the i'th local variable in 
    block `blk`.
    
    This function is similar to 'ComputeLocToGlobVarTransform'. The difference
    is that 'ComputeLocToGlobVarTransform' computes these indices for all
    variables in all blocks but this function computes the indices only 
    for the local variables in the given block. So this function can be 
    used in SparkADMM where each element of the RDD is the data of one block.

    Parameters
    ---------
    blk : integer
        Numberof the block for which the indices are computed.
        
    n_rows : integer
        number of rows in the oroginal grid.
        
    n_cols : integer
        number of columns in the oroginal grid.
        
    nr_blk : integer
        number of rows in each sub-grid. 
        
    nc_blk : integer
        number of columns in each sub-grid.
        
    Returns
    -------
    
    z_idx: numpy.array
       The i'th entry of this array is the 
       index of the global variable corresponding the i'th local variable in 
       block `blk`.    
    '''
    
    #no. of rows and column partitions
    n_r_part=(n_rows-1)/(nr_blk-1);n_c_part=(n_cols-1)/(nc_blk-1);
    rb,cb=np.unravel_index(blk,(n_r_part,n_c_part),'F')
    grid_r,grid_c = (rb*(nr_blk-1),cb*(nc_blk-1))#row and col of the upper left
                                       #corner of block on original grid
    x,y=np.meshgrid(range(0,nr_blk),range(0,nc_blk))
    spatial_idx = np.ravel_multi_index((grid_r+x.flatten(), 
                                  grid_c+y.flatten()),
                                (n_rows,n_cols),order='F')
    #spatial_idx are the indices on the original grid corrsponding to 
    #block=blk. Each point on grid has T data points. So the final 
    #indices are obtained as follows:
    
    z_idx = np.hstack([range(z*T,(z+1)*T) for z in spatial_idx])
    return z_idx             
    
    
def compute_cp_matrices(n_rows,n_cols,T,lam_t,lam_s,lh_trend=True,
                        ifCompute_Gh=False,wrapAround=False):
    '''
    This computes the matrices used in cp optimization.
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
        D=spmatrix(x,I,J,size=(r_s+r_t+r_lh,T*grid_size))        
    else: 
        r_lh=0               
        I=np.hstack([I_t,I_s+r_t]);J=np.hstack([J_t,J_s]);
        x=np.hstack([x_t,x_s]);
        D=spmatrix(x,I,J,size=(r_s+r_t,T*grid_size))
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
    
    
    