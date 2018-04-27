from volatilitytrend.algorithms.base import LinearizedADMM

dataset='north_hemisphere'
#dataset='us'
lam_t_vec=[4]
lam_s_vec=[0]

if dataset=='us':
    data_fn='1992-08-31_to_2002-08-31_data_avg_us_detrended'
    metadata_fn='1992-08-31_to_2002-08-31_metadata_avg_us_detrended'
    destDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/'+\
    '19920831to20020831/Fitted_LinADMM_test'
    
elif dataset=='north_hemisphere':
    data_fn='1960-01-01_to_2010-12-31_data_avg_sub_north_detrended'
    metadata_fn='1960-01-01_to_2010-12-31_metadata_avg_sub_north_detrended'
    destDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/'+\
            'globe/Fitted_linADMM'

print('\014')
la = LinearizedADMM()#construct linearizedADMM object
la.loadData(data_fn,metadata_fn)#load data
la.fit(destDir,lam_t_vec,lam_s_vec,maxIter=5000,freq=500,
       ifWarmStart=False,earlyStopping=True,mu=.01,
       ifAdaptMu=False,mu_adapt_rate=.999,mu_adapt_freq=1)



