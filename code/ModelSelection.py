from volatilitytrend.algorithms.base import LinearizedADMM
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

lam_t_vec=[0,2,4,8,10,15,200]
lam_s_vec=[0,.1,.5,2,5,10]
mu_vec=[.01]*6+[.001]*3
dataset='north_hemisphere'
#dataset='us'


if dataset=='us':
    data_fn='1992-08-31_to_2002-08-31_data_avg_us_detrended'
    metadata_fn='1992-08-31_to_2002-08-31_metadata_avg_us_detrended'
    fittedModelsDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/'+\
    '19920831to20020831/Fitted_LinADMM_test'
    
    
elif dataset=='north_hemisphere':
    data_fn='1960-01-01_to_2010-12-31_data_avg_sub_north_detrended'
    metadata_fn='1960-01-01_to_2010-12-31_metadata_avg_sub_north_detrended'
    fittedModelsDir='/home/arash/datasets/ecmwf/Fitted_linADMM'
    saveFigDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/globe/Figures'
    

#===load data===
print('\014')
la = LinearizedADMM()#construct linearizedADMM object
la.loadData(data_fn,metadata_fn)#load data
#===load data===

la.modelSelection(fittedModelsDir,lam_t_vec,lam_s_vec,mu_vec,
                  lh_trend=True,wrapAround=True)

print('optimal penalty parameters are: lam_t={:.2f} , lam_s={:.2f}'.\
      format(la.opt_lamt,la.opt_lams))
  





 