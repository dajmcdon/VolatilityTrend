import boto3,utils
import numpy as np
from os.path import join
from pyspark import SparkContext
import matplotlib.pyplot as plt


#===configuration===
iteration=4
where='local'
lam_t=100.0;lam_s=10.;rho=50.
dataDir ='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/'+\
         'colorado_19920831to20020831/test'
loc_resultsDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/'+\
            'colorado_19920831to20020831/results/'+where
param_str=param_str=where+'_lamt_'+str(lam_t)+\
                    '_lams_'+str(lam_s)+'_rho_'+str(rho)
ifLoadModelFromS3=False
s3_resultsDir='results/'+param_str                    
loc_resultsDir=join(loc_resultsDir,param_str)
#===configuration===

#===load fitted model from S3 into local computer===
if ifLoadModelFromS3:
    s3 = boto3.resource('s3')
    utils.downloadS3Dir(s3,'arash-test-data',s3_resultsDir,loc_resultsDir)
#===load fitted model from S3 into local computer===

#===load metadata===
sc = SparkContext.getOrCreate()
metadata=sc.pickleFile(join(dataDir,'metadata')).collect()[0]
n_rows,n_cols,T,nr_blk,nc_blk,n_blocks,n_x_b,lats,lons,weekly_avg=\
(metadata['n_rows'],metadata['n_cols'],metadata['T'],
metadata['nr_blk'],metadata['nc_blk'],metadata['n_blocks'],metadata['n_x_b'],
metadata['lats'],metadata['lons'],
metadata['weekly_avg'])
#===load metadata===

#===load fittted values===
rdd=sc.pickleFile(join(loc_resultsDir,'iteration'+str(iteration)))
fittedModel=rdd.collect()
Z_idx=np.concatenate([x[1]['Z_idx'] for x in fittedModel])
Zl=np.concatenate([np.array(x[1]['Zl']) for x in fittedModel]).flatten()
Yl=np.concatenate([np.array(x[1]['data']) for x in fittedModel]).flatten()

h=np.zeros(T*n_rows*n_cols);Y=np.zeros(T*n_rows*n_cols)
h[Z_idx]=Zl;Y[Z_idx]=Yl;
h=h.reshape((n_rows*n_cols,T));Y=Y.reshape((n_rows*n_cols,T));

i=10
plt.figure()
plt.plot(Y[i,:]);
plt.plot(np.exp(h[i,:]/2))
sc.stop()
#===load fittted values===




