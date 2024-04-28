%matplotlib qt
import mne
from config_FaceName import Results_data_path,Ids,MEG_data_path,group_name,MRI_data_path,stats_conds_list,stats_comparisions,study_path
from custom_functions import do_cluster_permutation_1samp_test,prepare_X_and_one_data
import os.path as op
from config_FaceName import Results_data_path,stats_comparisions,MRI_data_path,Results_data_path
from custom_functions import read_stats_list, print_sig_stats
import numpy as np
import os
import matplotlib.pyplot as plt

Results_data_path=os.path.join(study_path, 'results-Rev1')


resample_rate=200
Ids.remove(18)

evks_all = {condtion: [] for condtion in stats_conds_list}
stcs_all = {condtion: [] for condtion in stats_conds_list}

for subject_id in Ids:   
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')

    evk=mne.read_evokeds(op.join(MEG_data_path,group_name+"%02d" % subject_id,'fname'+'_%02d'%(subject_id)+'_alt_rating_grouping-ave.fif'))
    for ev in evk:
        if ev.comment in stats_conds_list:
            evks_all[ev.comment].append(ev.resample(resample_rate))
    
            stc=mne.read_source_estimate(op.join(MEG_data_path,group_name+"%02d" %subject_id,'fname'+'_%02d'%(subject_id)+'-'+ev.comment+'_alt_rating_grouping-'+'dSPM'))
            stcs_all[ev.comment].append(stc.resample(resample_rate))


cluster_method='tfce'
tfce_p_sig=0.05/2.0 # adjust for sensor and source stat tests
sampling_rate=1000

#sensor stats
fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [2.2, 1.2]})

for i, comparision in enumerate(stats_comparisions):
    print(comparision)
    stats_list_evk=read_stats_list(mne.evoked.Evoked,comparision,Results_data_path,'1samp_ttest',cluster_method)
    sig_times=print_sig_stats(stats_list_evk,cluster_method,tfce_p_sig)
    sig_times=sig_times*sampling_rate
    if sig_times is not None:
            stats_list_evk[0].data[stats_list_evk[1].data>tfce_p_sig]=0.0

            mean_t_values=stats_list_evk[0].data.mean(axis=1)
            threshold=np.percentile(mean_t_values, 75)
            mask = mean_t_values >= threshold
            
    #select the evokeds for the comparision
    evks={k:v for k,v in evks_all.items() if k in comparision}
    X,one_data=prepare_X_and_one_data(evks,ch_picks='grad',time_window=(None,None))

    X=X*1e13

    #plot the ERF time course
    X1=X[0,:,np.where(mask)[0],:]
    X2=X[1,:,np.where(mask)[0],:]


    axs[i%2, i//2].plot(one_data.times*sampling_rate,X1.mean(axis=(0,1)),label=comparision[0])
    axs[i%2, i//2].plot(one_data.times*sampling_rate,X2.mean(axis=(0,1)),label=comparision[1])
    
    #fill between sig_times
    if i<3:
        axs[i%2, i//2].fill_betweenx(y=[0,X1.mean(axis=(0,1)).max()],x1=sig_times.min(),x2=sig_times.max(),color='gray',alpha=0.3)
    else:
        axs[i%2, i//2].fill_betweenx(y=[0,X1.mean(axis=(0,1)).max()],x1=279.99999702,x2=384.99999702,color='gray',alpha=0.3)
        axs[i%2, i//2].fill_betweenx(y=[0,X1.mean(axis=(0,1)).max()],x1=614.99999702,x2=994.99999702,color='gray',alpha=0.3)
    print(sig_times)
    axs[i%2, i//2].legend()

for ax in axs.flat:
    ax.set_ylim([0, ax.get_ylim()[1]])
    xmax=2000 if ax.get_xlim()[1]>1500 else 1000
    ax.set_xlim([-200, xmax])
    ax.set_yticks(range(0, int(ax.get_ylim()[1]) + 5, 5))
    ax.set_xticks(range(-200, int(ax.get_xlim()[1])+200, 200))

plt.show()
plt.savefig('S1.pdf')