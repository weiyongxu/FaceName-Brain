from config_FaceName import group_name,Ids,RSA_data_path,Results_data_path,MRI_data_path
import mne
import os.path as op
from custom_functions import prepare_X_and_one_data,prepare_stats_list,write_stats_list,generate_adjacency_matrix
from scipy import stats as stats
import numpy as np
from mne.stats import permutation_cluster_1samp_test

Ids.remove(18)

conditions=['pixel_gray','ID','age','gender','correct','rating','rt_memory','rt_rating']
cluster_method='tfce'
tfce_p_sig=0.05/2.0
time_window=(0,None)
adjacency_matrix=generate_adjacency_matrix(MRI_data_path=MRI_data_path)

for task in ['E','R']:
    
    evks_all = {condtion: [] for condtion in conditions}

    for subject_id in Ids:
        
        subject = group_name+"%02d" % subject_id
        print("processing subject: %s" % subject)            
        RSAs=mne.read_evokeds(op.join(RSA_data_path,'fname'+'_%02d'%(subject_id)+'-%s-rsa_ROI_normal_source-ave.fif'%(task)))
        for ev in RSAs:
            if ev.comment in conditions:
                evks_all[ev.comment].append(ev)

    for comparision in conditions:
        evokeds={k:v for k,v in evks_all.items() if k == comparision}
    
        X,one_data=prepare_X_and_one_data(evokeds,ch_picks='misc',baseline=None,cmb_grad=False,time_window=time_window)
        data_type=type(one_data)   
    
        X=np.squeeze(X)
        X=np.arctanh(X)
        X=np.transpose(X,[0,2,1]) # observations × time × space
    
        t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(X,threshold=dict(start=0, step=0.2),adjacency=adjacency_matrix)
        stats_list=prepare_stats_list(t_obs, clusters, cluster_pv, H0,one_data,cluster_method,tfce_p_sig)    
        
        write_stats_list(stats_list,data_type,[comparision],Results_data_path,stats_test_name='RSA_ROI_normal_'+task,stats_threshold_type=cluster_method)
        