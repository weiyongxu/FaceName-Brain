import mne
from config_FaceName import Results_data_path,Ids,MEG_data_path,group_name,stats_conds_list,stats_comparisions
from custom_functions import do_cluster_permutation_1samp_test
from functools import partial
import os.path as op
import mne
from config_FaceName import Results_data_path,MRI_data_path,stats_comparisions,study_path
import matplotlib.pyplot as plt
import numpy as np
from custom_functions import read_stats_list,print_sig_stats,prepare_X_and_one_data
import os

Ids.remove(18)

tfrs_all = {condtion: [] for condtion in stats_conds_list}

for subject_id in Ids:   
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
               
    tfr=mne.time_frequency.read_tfrs(fname.replace("_tsss_mc.fif", "_alt_rating_grouping_new_baseline-tfr.h5"))

    for tf in tfr: 
        if tf.comment in stats_conds_list:
            tfrs_all[tf.comment].append(tf.apply_baseline((-0.4,-0.2),'percent'))

ga_tfrs={k:mne.grand_average(tfrs_all[k]) for k in tfrs_all.keys()}
for title,ga_tfr in ga_tfrs.items():
    ga_tfr.plot_topo(baseline=None,title=title)


cluster_method='tfce'
tfce_p_sig=0.05/8.0 # adjust for sensor and source stat tests * 4 freq bands
diff=ga_tfrs['recall_hc_hit_eq_recall_lc_hit']-ga_tfrs['recall_lc_hit_eq_recall_hc_hit']


stats_list_alpha=read_stats_list(mne.time_frequency.AverageTFR,['recall_hc_hit_eq_recall_lc_hit', 'recall_lc_hit_eq_recall_hc_hit'],Results_data_path,'alpha','tfce')
sig_times=print_sig_stats(stats_list_alpha,cluster_method,tfce_p_sig)
stats_list_alpha[0].data[stats_list_alpha[1].data>tfce_p_sig]=0.0
mean_t_values=stats_list_alpha[0].data.mean(axis=(1,2))
threshold=np.percentile(mean_t_values, 25)
ch_mask_alpha = mean_t_values <= threshold

stats_list_beta=read_stats_list(mne.time_frequency.AverageTFR,['recall_hc_hit_eq_recall_lc_hit', 'recall_lc_hit_eq_recall_hc_hit'],Results_data_path,'beta','tfce')
sig_times=print_sig_stats(stats_list_beta,cluster_method,tfce_p_sig)
stats_list_beta[0].data[stats_list_beta[1].data>tfce_p_sig]=0.0
mean_t_values=stats_list_beta[0].data.mean(axis=(1,2))
threshold=np.percentile(mean_t_values, 25)
ch_mask_beta = mean_t_values <= threshold

ch_mask=np.logical_and(ch_mask_alpha,ch_mask_beta)
final_mask_beta=stats_list_beta[0].data[np.where(ch_mask)[0],:,:].mean(axis=0)<threshold
final_mask_alpha=stats_list_alpha[0].data[np.where(ch_mask)[0],:,:].mean(axis=0)<threshold


sig_mask=np.zeros(diff.data.shape[1:],dtype=bool)
alpha_idx=np.where(np.logical_and(diff.freqs>=8,diff.freqs<=13))[0]
beta_idx=np.where(np.logical_and(diff.freqs>=14,diff.freqs<=30))[0]

time_idx=np.where(np.logical_and(diff.times>=0,diff.times<=1))[0]

sig_mask[alpha_idx,100:301]=final_mask_alpha
sig_mask[beta_idx,100:301]=final_mask_beta


fig, axes = plt.subplots(3, 1, figsize=(6,8))

ga_tfrs['recall_hc_hit_eq_recall_lc_hit'].plot(picks=np.where(ch_mask)[0], combine='mean',vmin=-0.15, vmax=0.15,axes=axes[0])
axes[0].set_title('recall_hc_hit_eq_recall_lc_hit')

ga_tfrs['recall_lc_hit_eq_recall_hc_hit'].plot(picks=np.where(ch_mask)[0], combine='mean',vmin=-0.15, vmax=0.15,axes=axes[1])
axes[1].set_title('recall_lc_hit_eq_recall_hc_hit')

diff.plot(picks=np.where(ch_mask)[0], combine='mean', mask=sig_mask, mask_style='both', mask_alpha=0.5, mask_cmap='RdBu_r',vmin=-0.15, vmax=0.15,axes=axes[2])
axes[2].set_title('Difference')

plt.tight_layout()
plt.show()

#save the figure
fig.savefig('S2.pdf')