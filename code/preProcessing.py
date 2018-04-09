'''
This script does the following pre-processing steps:
    + data is read from a grib file. The data of a sub-region of earth
    is selected and saved as an numpy.ndarray. 
    
    + Data is subsampled spatially.
    
    + The weekly average of the data at each location is computed.
    
    + The results are saved in a csv file. This data should be detrended
    using genlasso in R.        
'''

from volatilitytrend.data_utils.ecmwf_dataUtils import gribToArray,\
computeAveragedData,spatialSubSample
from volatilitytrend import config
from time import ctime
import pandas as pd
from cPickle import load
import numpy as np
from os.path import join
print('\014')

fn='1960-01-01_to_2010-12-31'
dataDir=config.GRIB_FILES_DIR

#===convert grib to numpy.array===
#read data from grib files and save the data of the desired region
#as numpy.array in a binary file. Also save the metadata.

print(ctime()+'...converting grib to array...')
gribToArray(gribFilename=fn,outputFilename=fn,
            destDir=dataDir,
            region={'min_lat':0.1,'min_lon':.1,
                    'max_lat':80,'max_lon':-.1})
#===convert grib to numpy.array===

#===compute weekly average of data at each location===
print(ctime()+'...computing weekly average...')
computeAveragedData(dataDir,data_fn=fn+'_data',metadata_fn=fn+'_metadata')
#===compute weekly average of data at each location===

#===perform spatial sub-sampling====
print(ctime()+'...spatial sub-sampling...')
spatialSubSample(dataDir,data_fn=fn+'_data_avg',
                 metadata_fn=fn+'_metadata_avg',ss_factor=(2,4))
#===perform spatial sub-sampling====

#===save data into csv file===
#prepare data for genlasso R package. genlasso is used to detrend the data.
metadata=load(open(join(dataDir,
                        '1960-01-01_to_2010-12-31_metadata_avg_sub')))
dataMat=np.fromfile(join(dataDir,
                         '1960-01-01_to_2010-12-31_data_avg_sub'),
                    'float32')
dataMat=dataMat.reshape((metadata['n_cols']*metadata['n_rows'],
                         len(metadata['dates'])))
df=pd.DataFrame(dataMat)
df.to_csv(join(dataDir,'1960-01-01_to_2010-12-31_data_avg_sub.csv'),
          header=False,index=False)
#===save data into csv file===

#===save detrended data as numpy.array inbinary file===
#use gen lasso package in R to detrend this data. Use the script 
#`detrending.R` in dataUtils package. This will save data in a csv file.
#Then come back and run the folllowing lines.

#fn='1960-01-01_to_2010-12-31_data_avg_sub_north_detrended'
#dataMat=pd.read_csv(join(dataDir,fn+'.csv')).values
#dataMat.tofile(join(dataDir,fn))
#===save detrended data as numpy.array inbinary file===