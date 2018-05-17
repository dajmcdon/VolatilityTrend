import numpy as np
from os.path import join
from cPickle import load
import gc,datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
from time import ctime 

from .. import config
from .linearized_admm import linearizedADMM_fit
from .consensus_admm import consensusADMM_fit
from ..utils import latlon_to_rowcol,compute_D_and_lam,computeLoss

__metaclass__=type
class BaseAlgorithmClass():
    '''
    All alogrithm classes are sub-class of this class.
    '''
    
    def __init__(self,dataDir=None,destDir=None):
        if dataDir is None:
            self.dataDir=config.DATA_DIR
        else:
            self.dataDir=dataDir
        
        self.metadata={'n_rows':[],'n_cols':[],'T':[],'dates':[],
                       'lats':[],'lons':[]}
        self.dataMat=[]
        self.fittedVar=[]
        
    def loadData(self,data_fn,metadata_fn):
        
        #===load metadata===
        with open(join(self.dataDir,metadata_fn)) as f:
            self.metadata=load(f)
            
        n_rows,n_cols,T=(self.metadata['n_rows'],self.metadata['n_cols'],
                         self.metadata['T'])
        #===load metadata===

        #===load data===
        self.dataMat=np.fromfile(join(self.dataDir,data_fn),
                                 dtype='float32').reshape((n_rows*n_cols,T))
        #===load data===
        
    def analyseFittedValues(self,filepath):

        #===metadata===
        n_rows=self.metadata['n_rows'];n_cols=self.metadata['n_cols']
        gridsize=n_rows*n_cols
        T=self.metadata['T'];n_years=T/52 
        years=np.unique(np.array(map(lambda x:x.year,self.metadata['dates'])))
        self.metadata.update({'years':[datetime.date(y,06,01)\
                                       for y in years.tolist()]})
        #===metadata===
        
        #===load fitted values and compute fitted variance===        
        X=np.fromfile(filepath).reshape((gridsize,T))#???work on log(var)?
        self.fittedVar=X.copy()
        #===load fitted values and compute fitted variance===
        
        #===compute yearly average of fitted variance===        
        X=X[:,0:n_years*52].reshape((n_years*gridsize,52))
        X=np.mean(X,axis=1).reshape((gridsize,n_years))
        self.fittedVar_yearly_avg=X.copy()
        #===compute yearly average of fitted variance===
        
        #===compute the sum of the change in variance at each location===
        self.changeInVar=np.sum(np.diff(X,axis=1),axis=1).\
                        reshape((n_rows,n_cols),order='F')
        del X;gc.collect()                                    
        #===compute the sum of the change in variance at each location===
    
    def plot_solution_for_timeStamp(self,date,figsize,saveDir,suffix):
        #===metadata===
        lats,lons,n_rows,n_cols,dates=(self.metadata['lats'],
                                       self.metadata['lons'],
                                       self.metadata['n_rows'],
                                       self.metadata['n_cols'],
                                       self.metadata['dates'])
        #===metadata===
        
        #===retrieve the solution for the desired date===
        td=pd.DataFrame({'dates':dates}).dates.dt.date-\
           pd.Timestamp(date).date()        
        idx=td.abs().argmin()
        X = self.fittedVar[:,idx]
        #===retrieve the solution for the desired date===
        
        #===reshape data for map.pcolor===
        lons=((lons+180) % 360) - 180
        lats=lats.reshape((n_rows,n_cols),order='F')
        lons=lons.reshape((n_rows,n_cols),order='F')
        X=X.reshape((n_rows,n_cols),order='F')
        #===reshape data for map.pcolor===
        
        fig=plt.figure(figsize=figsize)
        map = Basemap(projection='mill',
                      llcrnrlon=lons.min(),llcrnrlat=lats.min(),
                      urcrnrlon=lons.max(),urcrnrlat=lats.max())
        map.drawcoastlines()
        map.drawparallels(np.arange(np.ceil(lats.min()),
                                    np.floor(lats.max()),20),
                          labels=[1,0,0,0])
        map.drawmeridians(np.arange(np.ceil(lons.min()),
                                    np.floor(lons.max()),40),
                          labels=[0,0,0,1])
        map.pcolor(lons,lats,data=X,latlon=True)
        plt.colorbar(fraction=.013)   
        
        fn=join(saveDir,'estimatedVar_in_{}_{}.pdf'.format(date,suffix))
        fig.savefig(fn,dpi=300,format='pdf')
        
        
    def plot_ts_of_locations(self,latList,lonList,
                             savepath,figureLayout,figsize,
                             overlayFittedSD=True):
        
        #===metadata===
        lats,lons,n_rows,n_cols,dates=(self.metadata['lats'],
                                       self.metadata['lons'],
                                       self.metadata['n_rows'],
                                       self.metadata['n_cols'],
                                       self.metadata['dates'])
        r,c=figureLayout
        #===metadata===
        
        fig=plt.figure(figsize=figsize)
        for i,lat,lon in zip(range(len(latList)),latList,lonList):
            row,col,idx=latlon_to_rowcol(lat,lon,lats,lons,n_rows,n_cols)
            if i==0:
                ax1=fig.add_subplot(r,c,i+1)
            elif r==1:
                fig.add_subplot(r,c,i+1,sharey=ax1)
            elif c==1:
                fig.add_subplot(r,c,i+1,sharex=ax1)
                
            plt.plot(dates,self.dataMat[idx,:]);
            
            if overlayFittedSD:
                years=self.metadata['years']
                plt.plot(dates,2*np.sqrt(np.exp(self.fittedVar[idx,:])/2));
                plt.plot(years,2*np.sqrt(np.exp(self.\
                                              fittedVar_yearly_avg[idx,:])/2));
                
        if overlayFittedSD:
            plt.legend(['data','estimated SD',
                                'yearly average \n of estimated SD'])
        fig.text(0.5, 0.04, 'year', ha='center', va='center')
        fig.text(0.08,0.5,'temperature (K)',ha='center',va='center',
                 rotation='vertical')
        fig.savefig(savepath,dpi=300,format='pdf')
        
    def plotAvgChangeInVariance(self,saveDir,suffix):
        
        #===metadata===
        lats,lons,n_rows,n_cols=(self.metadata['lats'],
                                       self.metadata['lons'],
                                       self.metadata['n_rows'],
                                       self.metadata['n_cols'])
        #===metadata===
        
        lons=((lons+180) % 360) - 180
        lats=lats.reshape((n_rows,n_cols),order='F')
        lons=lons.reshape((n_rows,n_cols),order='F')
        
        #===plot average change in variance===
        fig=plt.figure(figsize=(10,6))
        plt.plot()
        map = Basemap(projection='mill',
                      llcrnrlon=lons.min(),llcrnrlat=lats.min(),
                      urcrnrlon=lons.max(),urcrnrlat=lats.max())
        map.drawcoastlines()
        map.drawparallels(np.arange(np.ceil(lats.min()),
                                    np.floor(lats.max()),20),
                          labels=[1,0,0,0])
        map.drawmeridians(np.arange(np.ceil(lons.min()),
                                    np.floor(lons.max()),40),
                          labels=[0,0,0,1])
        map.pcolor(lons,lats,data=self.changeInVar,latlon=True)
        plt.colorbar(fraction=.013)
        
        fn=join(saveDir,'avg_change_logVar_{}.pdf'.format(suffix))
        fig.savefig(fn,dpi=300,format='pdf')
        #===plot average change in variance===
        
        #===plot average variance===
        fig=plt.figure(figsize=(10,6))
        plt.plot()
        map = Basemap(projection='mill',
                      llcrnrlon=lons.min(),llcrnrlat=lats.min(),
                      urcrnrlon=lons.max(),urcrnrlat=lats.max())
        map.drawcoastlines()
        map.drawparallels(np.arange(np.ceil(lats.min()),
                                    np.floor(lats.max()),20),
                          labels=[1,0,0,0])
        map.drawmeridians(np.arange(np.ceil(lons.min()),
                                    np.floor(lons.max()),40),
                          labels=[0,0,0,1])
        map.pcolor(lons,lats,data=np.mean(self.fittedVar,axis=1).\
                                       reshape((n_rows,n_cols),order='F'),
                                       latlon=True)
        plt.colorbar(fraction=.013)
        fn=join(saveDir,'avg_logVar_{}.pdf'.format(suffix))
        fig.savefig(fn,dpi=300,format='pdf')
        #===plot average variance===        
        
class LinearizedADMM(BaseAlgorithmClass):

    def fit(self,destDir,lam_t_vec,lam_s_vec,
            mu=.01,maxIter=40000,freq=100,
            ifWarmStart=True,lh_trend=True,wrapAround=True,
            earlyStopping=True,patience=2,tol=.1,
            ifAdaptMu=False,mu_adapt_rate=.95,mu_adapt_freq=100):
        
        
        linearizedADMM_fit(self.dataMat,destDir,self.metadata,
                           lam_t_vec,lam_s_vec,mu=mu,
                           maxIter=maxIter,freq=freq,
                           ifWarmStart=ifWarmStart,lh_trend=lh_trend,
                           wrapAround=wrapAround,
                           earlyStopping=earlyStopping,
                           patience=patience,tol=tol,
                           ifAdaptMu=ifAdaptMu,mu_adapt_rate=mu_adapt_rate,
                           mu_adapt_freq=mu_adapt_freq)
      
    def modelSelection(self,fittedModelsDir,lam_t_vec,lam_s_vec,mu_vec,
                       lh_trend=True,wrapAround=True):
        '''
        This function computes the value of the loss function 
        for all combinations of given sets of values of lam_t and 
        lam_s.
        '''  
        
        #===check if the data is loaded===
        if len(self.dataMat)==0:
            msg='''Data has not been loaded.
            Use self.loadData() to load the data first'''
            raise ValueError(msg)
        #===check if the data is loaded===
    
        #===metadata===
        n_rows,n_cols,T=(self.metadata['n_rows'],self.metadata['n_cols'],
                         self.metadata['T'])    
        #===metadata===        
        
        #===make sure the parameters are float===
        lam_t_vec=[float(lam_t) for lam_t in lam_t_vec]
        lam_s_vec=[float(lam_s) for lam_s in lam_s_vec]
        mu_vec=[float(mu) for mu in mu_vec]
        #===make sure the parameters are float=== 
        
        #===compute D and o2===
        D,lam_vec=compute_D_and_lam(n_rows,n_cols,T,0,0,
                                    lh_trend=lh_trend,
                                    wrapAround=wrapAround)
        o2=(self.dataMat.reshape((-1,1)))**2        
        #===compute D===
        
        #===compute loss for all combinations of values===
        self.AIC=np.zeros((len(lam_t_vec),len(lam_s_vec)))
        self.BIC=np.zeros((len(lam_t_vec),len(lam_s_vec)))
        print(ctime()+'...computing AIC and BIC for:')
        for i,lam_t in enumerate(lam_t_vec):
            for j,lam_s in enumerate(lam_s_vec):
                print('lam_t={:.2f} , lam_s={:.2f} , mu={:.3f}'.\
                      format(lam_t,lam_s,mu))
                mu=mu_vec[i]
                fn='mu_{}_lam_t_{}_lam_s_{}'.format(mu,lam_t,lam_s)
                self.analyseFittedValues(join(fittedModelsDir,'X_'+fn))
                X=self.fittedVar.reshape((-1,1))
                
                NLL=np.sum(X + o2*np.exp(-X))
#                df=np.sum(np.abs(D.dot(X))>.005)
                df = np.sum(np.abs(D.dot(X)))
                self.AIC[i,j] =  NLL + 2*df
                self.BIC[i,j] =  NLL + np.log(X.size)*df
        #===compute loss for all combinations of values===
        
        #===optimal values===
        self.opt_lamt_idx,self.opt_lams_idx=\
                np.unravel_index(self.AIC.argmin(),self.AIC.shape)
        self.opt_lamt,self.opt_lams=(lam_t_vec[self.opt_lamt_idx],
                                     lam_s_vec[self.opt_lams_idx])
        #===optimal values===
        
class ConsensusADMM(BaseAlgorithmClass):

    def fit(self,destDir,
            lam_t_vec,lam_s_vec,rho=.1,
            n_r_b=2,n_c_b=2,
            maxIter=1000,freq=100,
            lh_trend=True,wrapAround=True,
            earlyStopping=True,patience=2,tol=.1):
        
        
        consensusADMM_fit(self.dataMat,destDir,self.metadata,
                          lam_t_vec,lam_s_vec,rho=rho,
                          n_r_b=n_r_b,n_c_b=n_c_b,
                          maxIter=maxIter,freq=freq,
                          lh_trend=lh_trend,wrapAround=wrapAround,
                          earlyStopping=earlyStopping,
                          patience=patience,tol=tol)
    
    
    
    
    
    
    
    
    
    
    