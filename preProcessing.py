'''
This script does the following pre-processing steps:
    + data is read from a grib file. The data of a sub-region of earth
    is selected and saved as an numpy.ndarray. 
    
    + Data is subsampled spatially.
    
    + The weekly average of the data at each location is computed.
    
    + The results are saved in a csv file. This data should be detrended
    using genlasso in R.        
'''

from volatilitytrend.dataUtils.ecmwf_dataUtils import gribToArray,\
computeAveragedData,spatialSubSample
from time import ctime
print('\014')

fn='1960-01-01_to_2010-12-31'
dataDir='/home/arash/datasets/ecmwf'
print(ctime()+'...converting grib to array...')
gribToArray(gribFilename=fn,outputFilename=fn,
            destDir=dataDir,
            region={'min_lat':0.1,'min_lon':.1,
                    'max_lat':80,'max_lon':-.1})

print(ctime()+'...computing weekly average...')
computeAveragedData(dataDir,data_fn=fn+'_data',metadata_fn=fn+'_metadata')

print(ctime()+'...spatial sub-sampling...')
spatialSubSample(dataDir,data_fn=fn+'_data_avg',
                 metadata_fn=fn+'_metadata_avg',ss_factor=(2,4))


#import pandas as pd
#from cPickle import load
#import numpy as np
#from os.path import join
#
#metadata=load(open(join(dataDir,'1960-01-01_to_2010-12-31_metadata_avg_sub')))
#
#dataMat=np.fromfile(join(dataDir,'1960-01-01_to_2010-12-31_data_avg_sub'),
#                    'float32')
#dataMat=dataMat.reshape((metadata['n_cols']*metadata['n_rows'],
#                         len(metadata['dates'])))
#df=pd.DataFrame(dataMat)
#df.to_csv(join(dataDir,'1960-01-01_to_2010-12-31_data_avg_sub.csv'),
#          header=False,index=False)
