import mne
from config_FaceName import Results_data_path,Ids,MEG_data_path,group_name,MRI_data_path,stats_conds_list,stats_comparisions
from custom_functions import do_cluster_permutation_1samp_test
from functools import partial
import os.path as op

bands = dict(theta=[4, 7],alpha=[8, 13],beta=[14, 30],gamma=[40, 90])

resample_rate=200
Ids.remove(18)

stcs_tfrs_all = {condtion+'_'+band: [] for condtion in stats_conds_list for band in bands }

for subject_id in Ids:   
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
    for cond in stats_conds_list:
        for b in bands:
            stcs_tfrs_all[cond+'_'+b].append(mne.read_source_estimate(fname.replace("_tsss_mc.fif", '-%s-%s-%s_alt_rating_grouping_new_baseline' %(cond[0].upper(),cond,b))))                

cluster_method='tfce'
p_threshold_cluster = 0.05
stat_fun = partial(mne.stats.ttest_1samp_no_p, sigma=1e-3)
stat_kwargs=dict(n_permutations=1000, tail=0, stat_fun=stat_fun, n_jobs=50, seed=None)
src_fname = MRI_data_path+'/fsaverage/bem/fsaverage-ico-4-src.fif'
src = mne.read_source_spaces(src_fname)
time_window=(0,None)

for compare in stats_comparisions:
    for band in bands:
        stcs_tfrs={k:v for k,v in stcs_tfrs_all.items() if k in [c+'_'+band for c in compare]}
        
        stats_list_stc=do_cluster_permutation_1samp_test(stcs_tfrs,results_folder=Results_data_path,src=src,time_window=time_window,
                                                          cluster_method=cluster_method,p_threshold_cluster=p_threshold_cluster,**stat_kwargs)