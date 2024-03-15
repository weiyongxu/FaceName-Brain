import mne
from config_FaceName import Results_data_path,Ids,MEG_data_path,group_name,MRI_data_path,stats_conds_list,stats_comparisions
from custom_functions import do_cluster_permutation_1samp_test
from functools import partial
import os.path as op

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

src_fname = MRI_data_path+'/fsaverage/bem/fsaverage-ico-4-src.fif'
src = mne.read_source_spaces(src_fname)

cluster_method='tfce'
p_threshold_cluster = 0.05
stat_fun = partial(mne.stats.ttest_1samp_no_p, sigma=1e-3)
stat_kwargs=dict(n_permutations=1000, tail=0, stat_fun=stat_fun, n_jobs=50, seed=None,spatial_exclude=None)
ch_picks='grad'
time_window=(0,None)

for compare in stats_comparisions:

    evokeds={k:v for k,v in evks_all.items() if k in compare}
    stats_list_evk=do_cluster_permutation_1samp_test(evokeds,time_window=time_window,ch_picks=ch_picks,results_folder=Results_data_path,
                                                      cluster_method=cluster_method,p_threshold_cluster=p_threshold_cluster,**stat_kwargs)

    stcs={k:v for k,v in stcs_all.items() if k in compare}
    stats_list_stc=do_cluster_permutation_1samp_test(stcs,time_window=time_window,ch_picks=ch_picks,results_folder=Results_data_path,src=src,
                                                      cluster_method=cluster_method,p_threshold_cluster=p_threshold_cluster,**stat_kwargs)
