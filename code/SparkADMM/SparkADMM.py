import numpy as np
from os.path import join
from pyspark import SparkContext
import utils,boto3,gc,shutil,os
from time import ctime
from cvxopt import matrix,mul,exp,div,spdiag,solvers
from scipy.special import lambertw
from pyspark.broadcast import Broadcast#to check if a variable is broadcast
print('\014')
#import matplotlib.pyplot as plt



#=====map functions===
#these functions are used in RDD transformations perormed during different 
#steps of spark ADMM.

def mapFunc1(inputTuple):
    '''
    This function takes 'inputTuple' as input and returns a list where
    the input and output have the following form:
    inputTuple: (blk,inputDict) where blk is the block number and 
    inputDict is a dictionary. This dictionary is explained in `x_update`.
    Each field of this dictionary is a vector. 
    mapFunc1 then returns a list where each element is a tuple of the form:
    (Z_idx[i],(X[i],W[i],blk,i)) where for example Z_idx[i] is the i'th element
    of the vector Z_idx in inputDict.
    '''
    blk=inputTuple[0]
    l=[(idx,(blk,x,w,i))\
       for i,(x,w,idx) in enumerate(zip(list(inputTuple[1]['X']),
                                        list(inputTuple[1]['W']),
                                        list(inputTuple[1]['Z_idx'])))]
    return l

def mapFunc2(inputTuple):
    '''
    This function computes the updated values of Z and W.
    inputTuple is a tuple of the form:
    (Z_idx,[(blk,X[0],W[0],0),...,(blk,X[m],W[m],m)]) where X[0],...,X[m]
    are m local variables corresponding to the global variable Z[Z_idx].
    The updated value of Z[Z_idx] is Z_updated=mean(X[0],...,X[m]).
    The updated value of W[j] is W[j]=W[j]+X[j]-Z_updated
    This function returs a list of the form:
    [(blk,(Z_updated,W[0],0)),...,(blk,(Z_idx,Z_updated,W[m],m))]
    '''
    
    Z_updated=np.mean([x[1] for x in inputTuple[1]])
    l=[(x[0],(Z_updated,x[2]+x[1]-Z_updated,x[3])) \
       for x in inputTuple[1]]
    return l

def mapFunc3(inputTuple,parameters):
    '''
    This function receives a tuple of the form:
    ( inputDict , pyspark.resultiterable)
    inputDict is a dictionary explained in `x_update` and containing the data
    for a block.
    pyspark.resultiterable contains the updated values of Zl,X and W for that 
    block.Each element of it is a tuple of form:
    (Zl[i],W[i],i)).
    This function replaces the updated values into inputDict and returns 
    a dictionary with the exact form as inputDict. It also computes the 
    lasso cost for each block.
    '''
        
    inputDict = inputTuple[0]
    z_updated = np.array([x[0] for x in inputTuple[1]])
    w_updated = np.array([x[1] for x in inputTuple[1]])
    idx = [x[2] for x in inputTuple[1]]
    inputDict['Zl'][idx]=z_updated
    inputDict['W'][idx]=w_updated
    inputDict['blkLoss']=computeGLassoCost(inputDict['X'],inputDict['data'],
             inputDict['K'],parameters)
    return inputDict

def mapFunc4(blk,parameters):
    '''
    This function is used in computing K, the number of local variables 
    corresponding to each global variable.
    
    The input is blk, the block number. It returns a list of tuples of the
    form [(Z_idx[i],(blk,i,1))]
    '''
    
    if isinstance(parameters,Broadcast):
        parameters = parameters.value 
    metadata=parameters['metadata']
    n_rows,n_cols,T,nr_blk,nc_blk=\
    (metadata['n_rows'],metadata['n_cols'],metadata['T'],
     metadata['nr_blk'],metadata['nc_blk'])     
    
    Z_idx = list(Compute_Z_idx_for_blk(blk,n_rows,n_cols,T,nr_blk,nc_blk))
    l=[(idx,(blk,i)) for i,idx in enumerate(Z_idx)] 
    return l

def mapFunc5(inputTuple):
    '''
    This function is used in computing K, the number of local variables 
    corresponding to each global variable.
    
    inputTuple has the following form:
    (Z_idx,pyspark.resultiterable) where Z_idx is the index of a global
    variable and each element of pyspark.resultiterable is a tuple of the 
    form: (blk,i). The function returns a list of tuples. Each element of 
    this list has the following form:
    (blk,(Z_idx,i,k)) where k = len(pyspark.resultiterable).
    '''
    
    Z_idx=inputTuple[0]
    k=len(inputTuple[1])
    l = [(x[0],(Z_idx,x[1],k)) for x in inputTuple[1]]
    return l

def mapFunc6(inputTuple,parameters):
    '''
    This function receives a tuple of the form:
    ( data , pyspark.resultiterable)
    data is an array containing the blocks data.
    
    Each element of pyspark.resultiterable is a tuple of form:
    (Z_idx,i,k). These values are explained in `mapFunc3` and `mapFunc4`.
    This function returns a dictionary with the following fields: 
        'data','X','W','Zl','z_idx','K','blkLoss'
    These are explained in the help of `x_update` function.        
    '''

    if isinstance(parameters,Broadcast):
        parameters = parameters.value 
    metadata=parameters['metadata']
    n_x_b=metadata['n_x_b']
    
    Z_idx = np.zeros((n_x_b),dtype='int64')
    K = np.zeros((n_x_b))
    idx = [x[1] for x in inputTuple[1]]
    z = np.array([x[0] for x in inputTuple[1]],dtype='int64')
    k = np.array([1.0/x[2] for x in inputTuple[1]])    
    Z_idx[idx]=z;K[idx]=k;
    
    D = {'data':inputTuple[0],
         'X':matrix(0.0,(n_x_b,1)),
         'W':matrix(0.0,(n_x_b,1)),
         'Zl':matrix(0.0,(n_x_b,1)),
         'Z_idx':Z_idx,
         'K':K,
         'blkLoss':None}
    return D
#=====map functions===
    
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

def computeGLassoCost(X,Y,K,parameters):
    
    #===get parameters===
    if isinstance(parameters,Broadcast):
        parameters = parameters.value         
    optimParam=parameters['optimParam']           
    metadata=parameters['metadata']
    T,nr_blk,nc_blk=(metadata['T'],metadata['nr_blk'],metadata['nc_blk'])     
    D,lam_t,lam_s = (optimParam['D'],
                     optimParam['lam_t'],optimParam['lam_s'])
    lh_trend=parameters['lh_trend']
    r_D,c_D=D.size
    #===get parameters===
    
    #===compute penalty vector===
    n_t=nr_blk*nc_blk*(T-2);
    K0=K[0::T]
    K=matrix(K)
    K1=matrix(np.tile(K0,(T-2,1)).reshape((n_t,1),order='F'))
    if lh_trend:
        n_year=T/52
        n_lh=(n_year-2)*nr_blk*nc_blk;n_s=r_D-n_lh-n_t
        K2=matrix(np.tile(K0,(n_year-2,1)).reshape((n_lh,1),order='F'))
        lam_mat=spdiag(matrix([lam_t*K1, matrix(lam_s,(n_s,1)), lam_t*K2]))
    else:
        n_s=D.size[0]-n_t
        lam_mat=spdiag(matrix([lam_t*K1, matrix(lam_s,(n_s,1))]))
    #===compute penalty vector===
    
    f1=np.sum(np.array( mul(K,(X+mul(Y**2,exp(-X)))) )) 
    f2=np.sum(np.abs( np.array(lam_mat*D*X) ))
    
    return f1+f2  
    
def lossFunc(v=None,z=None,param=None): 
    '''
    This function computes the augmented Lagrangian.
    '''
    Y,D,K,alpha,m,rho=(param['Y'],param['D'],
                       matrix(param['K']),param['alpha'],
                       param['m'],param['rho']);
    if v is None: return 0,matrix(0.0,(m,1))
    
    u = -D.T*v #compute primal
    
    K=matrix(1.0,K.size)#???delete this
    
    #===define some auxilary variables===
    Y2=mul(K,Y**2)
    z1=u-K+rho*alpha    
    
    W=matrix(np.real( lambertw( mul((Y2/rho),exp(-z1/rho)))  ))
    h_opt= W +(z1/rho)
    dh_to_du=(1/rho)*(div(1,1+W))
    z2=mul(Y2,exp(-h_opt))
    z3=z2+z1-rho*h_opt
    #===define some auxilary variables===
    
    #====compute f===
    f=sum( mul(u,h_opt)-mul(K,h_opt)-z2-(rho/2)*(h_opt-alpha)**2 )
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

def compute_primal(Y,v,D,K,alpha,rho):
    u = -D.T*v #compute u  
    Y2=mul(K,Y**2)
    z1=u-K+rho*alpha       
    W=matrix(np.real( lambertw(mul((Y2/rho),exp(-z1/rho))) ))
    beta= W +(z1/rho)

    return beta

def x_update(inputDict,parameters):
    '''
    This function performs the x-update step of the ADMM algorithm.
    It updates the local variables x for each block of data.
    See Boyd et al, 2010, page 55.
    
    Parameters
    --------
    
    inputDict: dict
        This should be in the form (blk,D) where blk is the block number and 
        For blk=i, D has the following fields:
            'data': observed data for block i.
            'X': the local primal variables for block i.
            'W': the local dual variables for block i. 
            'Zl': the global variables corresponding to block i. X, W and Zl
            are np.array of size n_blk_var (no. of variables in a block).
            'z_idx': each local variable X corrsponds to one and only one 
            global variable Z. This is an np.array of size n_blk_var. The i'th
            entry of this vector is the index of the global variable Z 
            corresponding to X[i]. This will be used for updating Z where 
            Z[j] is the average of the local variables X corresponding Z[j].
            'K': this is an np.array of size n_blk_var. Element i of this 
            array is 1/n where n is number of local variables corresponding
            the global variable Z[Z_idx[i]].
            'blkLoss': loss function evaluated for the block.
    
    parameters: dict
        This function only uses the following field of this dictionary:
    optimParam: pyspark.broadcast.Broadcast
        This is a broadcast variable which contains a dictionary with the 
        following fields:
            'D','G','h': matrices used in optimization. These should be 
            created by utils.compute_cp_matrices.
            'rho': ADMM penalty term.
    '''  
    #===get parameters===
    if isinstance(parameters,Broadcast):
        parameters = parameters.value 
    optimParam=parameters['optimParam']
    metadata=parameters['metadata']
    T,nr_blk,nc_blk=(metadata['T'],metadata['nr_blk'],metadata['nc_blk'])     
        
    Y = inputDict['data'];K = inputDict['K'];
    alpha = inputDict['Zl']-inputDict['W'];#alpha=Z-W    
    D,G,h,rho = (optimParam['D'],optimParam['G'],optimParam['h'],
                 optimParam['rho'])
    m,n=D.size
    #===get parameters===
    
    #===scale h===
    K=np.ones(K.shape)#???delete this
    h = scale_h(h,K,parameters['lh_trend'],T,nr_blk,nc_blk,m,n)
    #===scale h===
    
    #===compute local solution using cvxopt.cp===
    param={'Y':Y,'D':D,'K':K,'alpha':alpha,'m':m,'rho':rho}
    solvers.options['maxiters']=250;#???change this to 250
    solvers.options['show_progress']=False        
    sol=solvers.cp(F=lambda u=None,v=None:lossFunc(u,v,param=param),G=G,h=h)
    #===compute local solution using cvxopt.cp===
    
    #===update X and blkLoss===
    v=sol['x'];#dual solution
    x=compute_primal(Y,v,D,matrix(K),alpha,rho)#primal solution
    inputDict['X'] = x
    return inputDict
    #===update X and blkLoss===
    

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

__metaclass__ = type
class SparkADMM:    
    def __init__(self,sc,loc_srcDir=None,loc_desDir=None,
                 s3_srcDir=None,s3_desDir=None,bucketname=None,
                 nr_blk=2,nc_blk=2,where='local'):
        
        '''
        Constructor of SparkADMM class.
        
        Parameters
        ----------
        sc: pyspark.context.SparkContext
            spark context object. 
            
        loc_srcDir: str
            The directory on the local computer where the data will be read 
            from.
            
        loc_desDir: str
            The directory on the local computer where data or results will
            be written into.        

        s3_srcDir: str
            The directory on S3 where the data will be read from. This
            directory is relative to ``bucketname``.
            
        s3_desDir: str
            The directory on S3 where data or results will be written into.
             This directory is relative to ``bucketname``.
         
        bucketname: str
            Bucket name on S3.
        
        n_row_blk : integer
            number of rows in each sub-grid. 
            
        nc_blk : integer
            number of columns in each sub-grid.   

        where: str
            either 'local' (default) or 's3'. If it is set to 'local' then the
            data of the blocks will be saved into a local directory specified
            by ``destDir``. If it is 's3' the data will be
            saved into a bucket on aws 's3'.
                        
        All this parameters can also be set by the following methods:
        ``writeBlocksData``, ``runADMM``.             
        '''
        
        self.sc=sc
        self.loc_srcDir=loc_srcDir;self.loc_desDir=loc_desDir;
        self.s3_srcDir=s3_srcDir;self.s3_desDir=s3_desDir;
        self.bucketname=bucketname;
        self.nr_blk=nr_blk;self.nc_blk=nc_blk
        self.where=where;
        
    def writeBlocksData(self,loc_srcDir=None,loc_desDir=None,
                 s3_srcDir=None,s3_desDir=None,bucketname=None,
                 nr_blk=None,nc_blk=None,where=None,
                 verbose=1,weekly_avg=True):
        """
        This function first reads the data from a binary file on the local
        computer, then partitions the data into several blocks and finally
        saves the data of each block into a seperate file. The data can be
        written into either a directory on the local computer or into a 
        bucket on amazon S3. This is determined by the argument 
        ``where``.
        
        This data can then be read into an RDD where each element will be
        a dictionary {'block','data'} where 'data' is the data of a block.
        This data will be saved (as spark pickle files) in
        `loc_desDir\blocks` (or `s3_desDir\blocks`).
        
        In addition to the 
        blocks data, a dictionary of metadata are saved as spark pickle file
        in the folder `loc_desDir\metadata` (or `s3_desDir\metadata`).
        The metadata has the following fields: n_rows,n_cols,T, nr_blk,nc_blk,
        lats,lons, dates,n_blocks,
        n_x_b (number of local variables in each block)
        
        Parameters
        ----------
        
        weekly_avg: boolean
            if True (default) use the weekly averaged data. In this case
            the folder ``loc_srcDir`` should contain 5 binary files: 
            ``detrended_avg_data``, ``dates_avg``, ``detrended_avg_n_dim``,
            ``detrended_lats``, ``detrended_lons``.\n
            If False use the daily data. In this case
            the folder ``loc_srcDir`` should contain the same 5 binary files
            but without '_avg' in their name.                            
            
        verbose: int
            If 1 (default) when ``where='s3'``, it shows the name
            of the files being copied.
           
        For other parameters see ``__init()__``.
        
        Example
        -------
        
        >>> sa=SparkADMM()
        >>> sa.writeBlocksData(loc_srcDir='~/Data',loc_desDir='~/test',
        nr_blk=3,nc_blk=3,where='local')#write on local machine
        
        >>> sa.writeBlocksData(loc_srcDir='~/Data',loc_desDir='~/test',
        s3_desDir='test',nr_blk=3,nc_blk=3,
        where='s3',bucketname='arash-test-data')
        """
        
        if loc_srcDir is not None: self.loc_srcDir=loc_srcDir
        if loc_desDir is not None: self.loc_desDir=loc_desDir
        if s3_srcDir is not None: self.s3_srcDir=s3_srcDir
        if s3_desDir is not None: self.s3_desDir=s3_desDir
        if bucketname is not None: self.bucketname=bucketname
        if nr_blk is not None: self.nr_blk=nr_blk
        if nc_blk is not None: self.nc_blk=nc_blk
        if where is not None: self.where=where
        sc = self.sc
        
        
        nr_blk=self.nr_blk;nc_blk=self.nc_blk;
        
        #===load data from binary files===
        data = utils.loadData(self.loc_srcDir,weekly_avg=weekly_avg)
        n_rows,n_cols,T=(data['n_rows'],data['n_cols'],data['T'])        
        idx,n_rows,n_cols=utils.resize_data(n_rows,n_cols,nr_blk,nc_blk)
        dataMat=data['dataMat'][idx,:]
        lats=data['lats'][idx,:]
        lons=data['lons'][idx,:]
        #===load data from binary files===
        
        #===partition data into blocks===        
        n_x_b=nr_blk*nc_blk*T#no. of local variables in each block                
        y_flat=np.asarray(dataMat.flatten(),dtype='float64') 
        del data;gc.collect()
        
        A=utils.ComputeLocToGlobVarTransform(n_rows,n_cols,T,nr_blk,nc_blk)
        n_z,n_x=A.size
        I=np.array(A.I).flatten();J=np.array(A.J).flatten()
        
        data_blk=np.zeros(n_x)
        data_blk[J]=y_flat[I]
        n_blocks=data_blk.size/n_x_b
        data_blk=data_blk.reshape((n_x_b,n_blocks),
                                            order='F')
        #Now, each column of data_blk contains the data of a block
        #===partition data into blocks===        
        
        #===create a RDD with each element being data of a block===        
        kv_pairs=[{'block':i,'data':data_blk[:,i]} \
              for i in range(data_blk.shape[1])]
        rdd=sc.parallelize(kv_pairs,numSlices=data_blk.shape[1])
        #===create a RDD with each element being data of a block===

        #===create metadata dictionary===
        metadata = {'n_rows':n_rows,'n_cols':n_cols,'T':T,
                    'lats':lats,'lons':lons,'n_blocks':n_blocks,
                    'n_x_b':n_x_b,'nr_blk':nr_blk,'nc_blk':nc_blk,
                    'weekly_avg':weekly_avg}
        rdd_meta=sc.parallelize([metadata],numSlices=1)
        #===create metadata dictionary===
        
        #===save data in local computer===
        #note that even if where='s3', we first need to save
        #data in local computer.
        if os.path.exists(join(self.loc_desDir,'blocks')):
           shutil.rmtree(join(self.loc_desDir,'blocks'))

        if os.path.exists(join(self.loc_desDir,'metadata')):
           shutil.rmtree(join(self.loc_desDir,'metadata'))
           
        rdd.saveAsPickleFile(join(self.loc_desDir,'blocks'))
        rdd_meta.saveAsPickleFile(join(self.loc_desDir,'metadata'))        
            
        self.loc_srcDir=self.loc_desDir
        #===save data in local computer===
        
        #===save data in S3 bucket===
        if self.where=='s3':            
            print(ctime()+'...connecting to aws...')
            s3 = boto3.resource('s3')
            print(ctime()+'...writing data to s3...')
            utils.copy_dir_to_s3(s3,self.loc_desDir,self.s3_desDir,
                                 self.bucketname,verbose=verbose)
            self.s3_srcDir=self.s3_desDir
        #===save data in S3 bucket===
        
        
    def runADMM(self,maxIter,lam_t,lam_s,rho,lh_trend=True,logfreq=10):
        '''
        This function computes the solution using ADMM.
        TODO: more help
        
        Somewhere it creates an RDD named `rdd`.
        Each element of this rdd has
        the following form: (blk,D) where blk is the block number and 
        D is a dictionary. For blk=i D has the following fields:
            'data': observed data for block i.
            'X': the local primal variables for block i.
            'W': the local dual variables for block i. 
            'Zl': the global variables corresponding to block i. X, W and Zl
            are np.array of size n_blk_var (no. of variables in a block).
            'z_idx': each local variable X corrsponds to one and only one 
            global variable Z. This is an np.array of size n_blk_var. The i'th
            entry of this vector is the index of the global variable Z 
            corresponding to X[i]. This will be used for updating Z where 
            Z[j] is the average of the local variables X corresponding Z[j]. 
            'blkLoss': loss function evaluated for the block.
            
        Parameters
        --------
        
        lam_t: float
            temporal penalty.
        lam_s: float
            spatial penalty.
        rho: flaot
            augmented Lagrangian penalty.
        lh_trend: boolean
            If Ture (default) applies a penalty on the yearly average of 
            variance. More precisely, it adds the penalty 
            |v(i,j,y)-2v(i,j,y+1)+v(i,j,y+2)| where v(i,j,y) is the average 
            of estimated variance for spatial point (i,j) in year y.
        logfreq: int
            The results will be saved every `logfreq` iterations.
        '''        
        
        #===determine necessary directories===
#        conf = SparkConf().setMaster("local").setAppName("My App")
        sc = self.sc#(conf = conf)
        param_str=self.where+'_lamt_'+str(lam_t)+\
                  '_lams_'+str(lam_s)+'_rho_'+str(rho)
        if self.where=='local':
            dataDir = self.loc_srcDir            
            resultsDir = join(self.loc_desDir,param_str)
            if os.path.exists(resultsDir):
                shutil.rmtree(resultsDir)

        elif self.where=='s3':            
            #note that you cannot use this on local computer
            #because of missing dependencies of your spark to connect to s3.
            
            #--delete results folder on s3 if it exists---
#            s3 = boto3.resource('s3')
#            bucket=s3.Bucket(self.bucketname)
#            objs = bucket.objects.filter(Prefix=self.s3_desDir)
#            for obj in objs:
#                obj.delete()
#            del s3
            #--delete results folder on s3 if it exists---
            
            #---read aws credentials---
            f = open(join(os.path.expanduser('~'),'.aws/credentials'),'rb')
            for line in f:
                if 'AWS_ACCESS_KEY_ID' in line.split('='):
                    AWS_ACCESS_KEY_ID=line.split('=')[-1].strip('\n')
                if 'AWS_SECRET_ACCESS_KEY' in line.split('='):
                    AWS_SECRET_ACCESS_KEY=line.split('=')[-1].strip('\n')
            f.close()
            #---read aws credentials---
            
            dataDir = 's3n://'+AWS_ACCESS_KEY_ID+':'+AWS_SECRET_ACCESS_KEY+\
            '@'+join(self.bucketname,self.s3_srcDir)
            resultsDir = 's3n://'+AWS_ACCESS_KEY_ID+':'+AWS_SECRET_ACCESS_KEY+\
            '@'+join(self.bucketname,self.s3_desDir,param_str)
        #===determine necessary directories===
        
        #===load metadata===
        metadata=sc.pickleFile(join(dataDir,'metadata')).collect()[0]
        T,nr_blk,nc_blk,n_blocks=\
        (metadata['T'],metadata['nr_blk'],
         metadata['nc_blk'],metadata['n_blocks'])
        #===load metadata===
        
        #===create optimization matrices===
        #these matrices are used in cp optimization in x_update step.
        D,G,h=utils.compute_cp_matrices(nr_blk,nc_blk,T,lam_t,lam_s,lh_trend,
                                     ifCompute_Gh=True)
        optimParam = {'D':D,'G':G,'h':h,
                                   'lam_t':lam_t,'lam_s':lam_s,'rho':rho} 
        #===create optimization matrices===
        
        #===define  parameters as broadcast variable===
        parameters=sc.broadcast({'metadata':metadata,
                                 'optimParam':optimParam,
                                 'lh_trend':lh_trend})
#        parameters={'metadata':metadata,'optimParam':optimParam,
#                    'lh_trend':lh_trend}
        #===define  parameters as broadcast variable===
        
        #===compute K===
        #Here we compute the number of local variables corresponding to
        #each global variable.
        rdd_k=sc.parallelize(range(n_blocks)).\
                    flatMap(lambda x:mapFunc4(x,parameters)).groupByKey().\
                    flatMap(mapFunc5).groupByKey().partitionBy(n_blocks)        
        #===compute K===
        
        #===load or create main rdd===
        #TODO:???
        #if loadModel:
        rdd = sc.pickleFile(join(dataDir,'blocks')).\
                map(lambda x:(x['block'],matrix(x['data']))).\
                partitionBy(n_blocks)

        rdd = rdd.join(rdd_k).\
              mapValues(lambda x:mapFunc6(x,parameters)).cache()
        #else:
#        rdd=sc.pickleFile(join(resultsDir,'iteration47'))
        #===load or create main rdd===
        
        #===ADMM loop===
        totalLoss=[]
        for it in range(maxIter): 
            if self.where=='local':
                #---update X---
                rdd = rdd.collect()
                rdd = sc.parallelize(map(lambda x: (x[0],
                                                     x_update(x[1],parameters)),
                rdd)).partitionBy(n_blocks)
                #---update X---
                
                #---update Z and W and compute loss---
                updated_Z_and_W = rdd.\
                                     flatMap(lambda x:mapFunc1(x)).\
                                     groupByKey().\
                                     flatMap(lambda x:mapFunc2(x)).\
                                     groupByKey().partitionBy(n_blocks)
                 
                rdd = rdd.join(updated_Z_and_W).collect()
                rdd=sc.parallelize(map(lambda x:(x[0],
                                                 mapFunc3(x[1],parameters)),
                                        rdd)).partitionBy(n_blocks)                     
                #---update Z and W and compute loss---                
            else:    
                #---update X---
                rdd = rdd.mapValues(lambda inputDict:\
                                    x_update(inputDict,parameters))         
                #---update X---
            
                #---update Z and W and compute loss---
                updated_Z_and_W = rdd.\
                                     flatMap(lambda x:mapFunc1(x)).\
                                     groupByKey().\
                                     flatMap(lambda x:mapFunc2(x)).\
                                     groupByKey().partitionBy(n_blocks)
             
                rdd = rdd.join(updated_Z_and_W,numPartitions=n_blocks).\
                          mapValues(lambda x:mapFunc3(x,parameters)).cache()
                updated_Z_and_W.unpersist()  
                #---update Z and W and compute loss---  
            
            #---compute total loss---
            totalLoss.append(
                    rdd.map(lambda x:x[1]['blkLoss']).reduce(lambda x,y:x+y))
            #---compute total loss---
           
            #---save results---
            if (it==maxIter) | (it%logfreq==0) | (it>=50):#???
                print(ctime()+'...iteration %d...Loss=%f'%(it,totalLoss[-1]))
                rdd.saveAsPickleFile(join(resultsDir,'iteration'+str(it)))
                
                #---save loss function---
                if self.where=='local':
                    np.array(totalLoss).tofile(join(resultsDir,'totalLoss'))
                elif self.where=='s3':
                    s3 = boto3.resource('s3')
                    obj = s3.Object(self.bucketname,
                                    join(self.s3_desDir,'totalLoss'))
                    obj.put(Body=np.array(totalLoss).tobytes())
                #---save loss function---
            #---save results---
            
#        return rdd.collect()#???
        #===ADMM loop===        
        
        
       
if __name__=='__main__':
    sc = SparkContext.getOrCreate()
    where='s3'
    lam_t=100.0;lam_s=.1;rho=.1;maxIter=11
    
    if where=='s3':
        sa=SparkADMM(sc,s3_srcDir='test',s3_desDir='results',where=where,
                     bucketname='arash-test-data',nr_blk=3,nc_blk=3)
    else:    
        #colorado data
        loc_srcDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/'+\
        'colorado_19920831to20020831/test'
        loc_desDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/'+\
        'colorado_19920831to20020831/results/local'
        
#        #US data
#        loc_srcDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/'+\
#        '19920831to20020831/dataForSparkADMM'
#        loc_desDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/'+\
#        '19920831to20020831/results/local'  
        
        sa=SparkADMM(sc,loc_srcDir=loc_srcDir,loc_desDir=loc_desDir,
                     where=where,nr_blk=3,nc_blk=3)
    
#    sa.writeBlocksData()
    r=sa.runADMM(maxIter,lam_t,lam_s,rho,lh_trend=True,logfreq=10)
    sc.stop()

#s3 = boto3.resource('s3')
#utils.downloadS3Dir(s3,'arash-test-data','results','/home/arash/')                
#sc = SparkContext.getOrCreate()
#rdd=sc.pickleFile('/home/arash/results/')        
#r=rdd.collect()
#out=r[0][1]
#err=np.abs(np.array(out['Zl']).\
#           flatten()-np.array(Z_hat[:,3]).flatten());err.max()        
        