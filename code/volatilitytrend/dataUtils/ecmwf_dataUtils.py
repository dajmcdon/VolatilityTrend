'''
This module contains functions for loading and analyzing
datasets from ecmwf. See:
https://software.ecmwf.int/wiki/display/WEBAPI/Access+ECMWF+Public+Datasets        
'''

from .. import config
from os.path import join
import pygrib
from time import ctime
import numpy as np
from cPickle import dump,load
import gc

def loadGribFilesFromServer(dataset,start_date,end_date):
    '''
    This function loads data from ecmwf server and saves it as a .grib file.
    The data will be saved in VolatilityTrend.config.GRIB_FILES_DIR.
    
    Parameters
    -------
    dataset : str
        dataset name (e.g., 'era40')
    
    start_date: str
        start date in the format 'yyyy-mm-dd'
    
    end_date: str
        end date in the same format as start_date
        
    Example
    -------
    
    >>> from VolatilityTrend.dataUtils.ecmwf_dataUtils import loadGribFilesFromServer
    >>> loadGribFilesFromServer('era40','1986-02-01','1986-02-28')
    '''
    
    from ecmwfapi import ECMWFDataServer
    output=join(config.GRIB_FILES_DIR,start_date+'_to_'+end_date+'.grib')
    server = ECMWFDataServer()
    server.retrieve({
        "class": "e4",
        "dataset": dataset,
        "date": start_date+'/to/'+end_date,
        "levtype": "sfc",
        "param": "167.128",
        "step": "0",
        "stream": "oper",
        "time": "12:00:00",
        "type": "an",
        "target": output})
    
def gribToArray(gribFilename,outputFilename,destDir,
                region={'min_lat':22,'min_lon':-134,
                        'max_lat':58,'max_lon':-58}):
    
    '''
    This function converts data saved as .grib files into numpy array
    and save it as binary file. Only the data of the locations inside
    `region` are saved.
    
    After running this function two files will be saved in destDir:
        
        + a binary file named destDir/outputFilename+'_data' which contains\
        the numpy array of data. 
                
        + a file named destDir/outputFilename+'_metadata' which contains\
        a dictionary with the following fields:\
            'lats','lons','dates','n_rows','n_cols'
    
    Parameter
    --------
    
    gribFilename: str ot list
        if str, it is assumed that all data is in that single file. If
        list, it should contain the name of the .grib files.
        The grib file(s) should be in the directory specified by
        VolatilityTrend.config.GRIB_FILES_DIR.
    
    outputFilename : str
        The file name in which the numpy array will be saved.
        
    destDir: str
        The directory where the output file will be saved in.
        
    region : dict
        A dictionary containing the min and max values of the latitude
        and longitudes of a rectangle. This rectangle specifies
        a region on earth over which the analysis will be performed. The
        default values specify a region over the united states. To do the
        analysis over the earth, set ``region=None``. The values specified
        here can be obtained from google map. There is a mapping between
        the longitude values from google map and the longitude values in 
        the data set which is taken care of in this function.   
        
    Example
    -------
    >>> gribToArray('1986-01-01_to_1990-12-31',
            '1986-01-01_to_1990-12-31',
            destDir='/home/arash/datasets/ecmwf')
    '''
    
    if isinstance(gribFilename,str):
        gribFilename=[gribFilename]
    
    dataDir=config.GRIB_FILES_DIR
    
    l=[];date=[]
    for k,filename in enumerate(gribFilename):
        grbs=pygrib.open(join(dataDir,filename+'.grib'))            
        print(ctime()+'...retrieving data from file: %s...'%filename)
        
        for i,grb in enumerate(grbs.select()):
            if (i==0) and (k==0):
                #read lats and lons
                lats, lons = grb.latlons()
                n_rows=lats.shape[0];
                n_cols=lats.shape[1];
                grid_size=lats.size
                lats=lats.reshape((grid_size,1),order='F')
                lons=lons.reshape((grid_size,1),order='F')
                
                #compute the indices of the desired region
                if region is not None:
                    ind = selectRegion(region,lats,lons)
                    lats=lats[ind,];lons=lons[ind,];
                    n_rows=np.unique(lats).size;
                    n_cols=np.unique(lons).size;
                    
            #read data    
            allData=np.reshape(np.array(grb.values,'float32'),
                               (grid_size,1),order='F')
            
            #only keep the data of the desired region
            if region is not None:
                allData=allData[ind,];

            l.append(allData)
            date.append(grb.analDate)

        grbs.close()
        del grbs;gc.collect()
        dataMat=np.concatenate(l,axis=1)
        del l;gc.collect()
        
        #save data and metadata
        dataMat.tofile(join(destDir,outputFilename+'_data'))
        metadata={'lats':lats,'lons':lons,
                  'n_rows':n_rows,'n_cols':n_cols,'dates':date}
        with open(join(destDir,outputFilename+'_metadata'),'wb') as f:
            dump(metadata,f)
    
def computeAveragedData(dataDir,data_fn,metadata_fn,win_size=7):
    '''
    This function computes the average of daily data. The window size
    of averaging can be specified by the user.
    This function assumes that the data are saved in two binary files
    in `dataDir`:
        + a file named start_date+'_to_'+end_date+'_data'
        + a file named start_date+'_to_'+end_date+'_metadata'
    Run gribToArray() before running this function to create these files
    from grib files.
    
    Parameter
    --------    
        
    dataDir: str
        The directory where the data should be read from.
    
    data_fn: str
        data filename.
        
    metadata_fn: str
        metadata filename
    
    win_size: int (default 7)
        The average of daily data over `win_size` days will be computed.
        
    The results are saved as a numpy.ndarray in one binary file. A
    metadata file will also be saved.
    
    Example
    -------
    >> computeAveragedData('/home/arash/datasets/ecmwf',
                            data_fn='1986-01-01_to_1990-12-31_data',
                            metadata_fn='1986-01-01_to_1990-12-31_metadata')
    '''
    
    
    #===load data===
    with open(join(dataDir,metadata_fn)) as f:
        metadata=load(f)
    
    n_rows,n_cols,dates=(metadata['n_rows'],metadata['n_cols'],
                         metadata['dates'])
    dataMat=np.fromfile(join(dataDir,data_fn),'float32').\
                            reshape((n_rows*n_cols,len(metadata['dates'])))
    grid_size,T=dataMat.shape    
    #===load data===
    
    #===compute averaged data===
    n_win=T/win_size#number of time windows
    dataMat=dataMat[:,0:(n_win*win_size)].\
                    reshape((grid_size*n_win,win_size))
    dataMat=np.dot(dataMat,np.ones((win_size,1))/float(win_size))
    dataMat=np.array(dataMat.reshape((grid_size,n_win)),'float32')
    dataMat.tofile(join(dataDir,data_fn+'_avg'))
    #===compute averaged data===
    
    #===sub-sample from dates===
    idx=np.arange(0,(n_win*win_size),win_size)
    new_dates=[dates[i] for i in idx]
    metadata.update({'dates':new_dates})
    with open(join(dataDir,metadata_fn+'_avg'),'wb') as f:
        dump(metadata,f)
    #===sub-sample from dates=== 
    
    del dataMat;gc.collect()
    
def spatialSubSample(dataDir,data_fn,metadata_fn,ss_factor=(2,2)):
    '''
    This function takes a spatial sub-sample of the data.
    
    Parameter
    --------    
        
    dataDir: str
        The directory where the data should be read from.
    
    data_fn: str
        data filename.
        
    metadata_fn: str
        metadata filename
    
    ss_factor: tuple
        A tuple of size 2, where the first and second elements are
        the sub-sampling factor for the latitude and longitude, respectively.
        
    Example
    -------
    >> computeAveragedData('/home/arash/datasets/ecmwf',
                            data_fn='1986-01-01_to_1990-12-31_data',
                            metadata_fn='1986-01-01_to_1990-12-31_metadata',
                            ss_factor=(2,2))        
    '''
    
    #===load data===
    with open(join(dataDir,metadata_fn)) as f:
        metadata=load(f)
    
    n_rows,n_cols,lats,lons=(metadata['n_rows'],metadata['n_cols'],
                             metadata['lats'],metadata['lons'])
    dataMat=np.fromfile(join(dataDir,data_fn),'float32').\
                            reshape((n_rows*n_cols,len(metadata['dates'])))
    grid_size,T=dataMat.shape    
    #===load data===
    
    #===sub-sample===
    row_idx=np.arange(0,n_rows,ss_factor[0])
    col_idx=np.arange(0,n_cols,ss_factor[1])
    n_rows_new=row_idx.size;n_cols_new=col_idx.size;
    row_idx,col_idx=np.meshgrid(row_idx,col_idx)
    idx=np.ravel_multi_index((row_idx.flatten(),col_idx.flatten()),
                             (n_rows,n_cols),order='F')
    dataMat=dataMat[idx,:];lats=lats[idx,:];lons=lons[idx,:];
    #===sub-sample===
    
    #===save===
    dataMat.tofile(join(dataDir,data_fn+'_sub'))
    metadata.update({'n_rows':n_rows_new,'n_cols':n_cols_new,
                     'lats':lats,'lons':lons})
    with open(join(dataDir,metadata_fn+'_sub'),'wb') as f:
        dump(metadata,f)
    #===save===
    
    del dataMat;gc.collect()
    
def selectRegion(region,lats,lons):

    #map lon values from google map to dataset values
    if region['max_lon']<0:
        region['max_lon']=region['max_lon']+360
    if region['min_lon']<0:
        region['min_lon']=region['min_lon']+360
        
    ind = (lats <= region['max_lat']) &\
          (lats >= region['min_lat']) &\
          (lons <= region['max_lon']) &\
          (lons >= region['min_lon'])
    return ind.flatten()
  
    