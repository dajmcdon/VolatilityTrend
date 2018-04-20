from volatilitytrend.algorithms.base import LinearizedADMM
from os.path import join
import matplotlib.pyplot as plt

lam_t=2;lam_s=5;mu=.01
dataset='north_hemisphere'
#dataset='us'


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
    saveFigDir='/home/arash/MEGA/MEGAsync/Projects/Cloud/Data/globe/Figures'

print('\014')
la = LinearizedADMM()#construct linearizedADMM object
la.loadData(data_fn,metadata_fn)#load data

mu,lam_t,lam_s=(float(mu),float(lam_t),float(lam_s))
fn='mu_'+str(mu)+'_lam_t_'+str(lam_t)+'_lam_s_'+str(lam_s)
la.analyseFittedValues(join(destDir,'X_'+fn))

saveFigPath=join(saveFigDir,'ts_'+fn)
la.plot_ts_of_locations([39.17,32.7],[-86.52,-117.6],saveFigPath,
                        figureLayout=(1,2),figsize=(12,4))
la.plotAvgChangeInVariance(saveFigDir)

#===plot histogram of avg. change===
fig=plt.figure()
plt.hist(la.changeInVar.flatten(),bins=50)
plt.xlabel('change in variance');plt.ylabel('frequency');
fig.savefig(join(saveFigDir,'hist_avg_change.pdf'),dpi=300,format='pdf')
#===plot histogram of avg. change===

la.plot_solution_for_timeStamp('1974-01-01',figsize=(18,6),
                               saveDir=saveFigDir)





