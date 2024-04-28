import mne
from config_FaceName import Results_data_path,Ids,MEG_data_path,group_name,stats_conds_list,stats_comparisions
from custom_functions import do_cluster_permutation_1samp_test
from functools import partial
import os.path as op

bands = dict(theta=[4, 7],alpha=[8, 13],beta=[14, 30],gamma=[40, 90])

Ids.remove(18)

tfrs_all = {condtion: [] for condtion in stats_conds_list}

for subject_id in Ids:   
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
               
    tfr=mne.time_frequency.read_tfrs(fname.replace("_tsss_mc.fif", "_alt_rating_grouping_new_baseline-tfr.h5"))
    for tf in tfr: 
        if tf.comment in stats_conds_list:
            tfrs_all[tf.comment].append(tf) 

cluster_method='tfce'
p_threshold_cluster = 0.05
stat_fun = partial(mne.stats.ttest_1samp_no_p, sigma=1e-3)
stat_kwargs=dict(n_permutations=1000, tail=0, stat_fun=stat_fun, n_jobs=50, seed=None)
ch_picks='grad'
time_window=(0,None)

for compare in stats_comparisions:
    tfrs={k:v for k,v in tfrs_all.items() if k in compare}
    for band_name, band in bands.items():
        stats_list_tfr=do_cluster_permutation_1samp_test(tfrs,ch_picks=ch_picks,results_folder=Results_data_path,time_window=time_window,freq_window=band,baseline=[-0.4,-0.2],
                                                        cluster_method=cluster_method,p_threshold_cluster=p_threshold_cluster,stats_test_name=band_name,**stat_kwargs)