from volatilitytrend.algorithms.base import LinearizedADMM
from os.path import join
import matplotlib.pyplot as plt

lam_t=4;lam_s=2;mu=.01
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
mu,lam_t,lam_s=(float(mu),float(lam_t),float(lam_s))
fn='mu_{}_lam_t_{}_lam_s_{}'.format(mu,lam_t,lam_s)
la = LinearizedADMM()#construct linearizedADMM object
la.loadData(data_fn,metadata_fn)#load data
#===load data===

#===analyse fitted value and compute var and ave. var===
la.analyseFittedValues(join(fittedModelsDir,'X_'+fn))
#===analyse fitted value and compute var and ave. var===

#===plot time series of temerature at different locations===
saveFigPath=join(saveFigDir,'ts_'+fn)
la.plot_ts_of_locations([39.17,32.7],[-86.52,-117.6],saveFigPath,
                        figureLayout=(1,2),figsize=(12,4))
#===plot time series of temerature at different locations===

#===plot average change in variance===
la.plotAvgChangeInVariance(saveFigDir,suffix=fn)
#===plot average change in variance===

#===plot histogram of avg. change===
fig=plt.figure()
plt.hist(la.changeInVar.flatten(),bins=50)
plt.xlabel('change in variance');plt.ylabel('frequency');
fig.savefig(join(saveFigDir,'hist_avg_change_{}.pdf'.format(fn)),
            dpi=300,format='pdf')
#===plot histogram of avg. change===

#===plot solution for a given data===
la.plot_solution_for_timeStamp('2009-01-01',figsize=(18,6),
                               saveDir=saveFigDir,suffix=fn)
#===plot solution for a given data===




