from config_FaceName import group_name,Ids,RSA_data_path,MRI_data_path,Results_data_path
import mne
import os.path as op
from custom_functions import print_sig_stats,read_stats_list
from scipy import stats as stats
import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
from mne_rsa.source_level import backfill_stc_from_rois

Ids.remove(18)

conditions=['pixel_gray','ID','age','gender','correct','rating','rt_memory','rt_rating']

conditions=['pixel_gray','ID','gender','age']
CB_color_cycle = ['#4daf4a','#377eb8', '#984ea3', '#ff7f00']
colors = {condition: CB_color_cycle[i % len(CB_color_cycle)] for i, condition in enumerate(conditions)}


cluster_method='tfce'
tfce_p_sig=0.05/2.0 # adjust for sensor and source stat tests
labels = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=MRI_data_path)
src=mne.read_source_spaces(MRI_data_path+'/fsaverage/bem/fsaverage-ico-4-src.fif')

fig,ax=plt.subplots(len(conditions),2,squeeze=False,gridspec_kw={'width_ratios': [2,1]},figsize=(8,6))

for task,task_idx in zip(['E','R'],[0,1]):
    
    evks_all = {condtion: [] for condtion in conditions}

    for subject_id in Ids:
        
        subject = group_name+"%02d" % subject_id
        print("processing subject: %s" % subject)            
        RSAs=mne.read_evokeds(op.join(RSA_data_path,'fname'+'_%02d'%(subject_id)+'-%s-rsa_ROI_normal_source-ave.fif'%(task)))
        for ev in RSAs:
            if ev.comment in conditions:
                evks_all[ev.comment].append(ev)

    
    for cmp_idx,comparision in enumerate(conditions):
        evokeds={k:v for k,v in evks_all.items() if k == comparision}
        stats_list=read_stats_list(mne.evoked.Evoked,[comparision],Results_data_path,'RSA_ROI_normal_'+task,'tfce')

        sig_times=print_sig_stats(stats_list,cluster_method,tfce_p_sig)
        print(comparision)
        spatial_sig_idx, temporal_sig_idx = np.where(stats_list[-1].data<tfce_p_sig)
        spatial_sig_idx = np.unique(spatial_sig_idx).tolist()
        temporal_sig_idx = np.unique(temporal_sig_idx).tolist()

        sig_channels = [ev.ch_names[i] for i in spatial_sig_idx]
        sig_t_values = stats_list[0].data[spatial_sig_idx, :][:, temporal_sig_idx].mean(axis=1)

        picks=sig_channels if len(sig_channels)>0 else 'misc'
        #[-0.005, 0.010] memory
        mne.viz.plot_compare_evokeds({comparision:evks_all[comparision]},picks=picks,ylim=dict(misc=[-0.005, 0.005]), ci=True,axes=ax[cmp_idx,task_idx],title='', show_sensors=False,truncate_yaxis=False, truncate_xaxis=False ,combine='mean',colors={comparision:colors[comparision]})
        if sig_times is not None:
            ax[cmp_idx,task_idx].plot(sig_times,[0.0]*len(sig_times),'.',color=colors[comparision],markersize=5)

            sig_labels=[label for label in labels[:-1] if label.name in sig_channels]
            stc=backfill_stc_from_rois(sig_t_values, sig_labels, src, subject='fsaverage')
            d_max=np.abs(sig_t_values).max()
            d_min=np.abs(sig_t_values).min()

            brain=stc.plot(views=['lat','med'], hemi='split', size=(600, 600), subject='fsaverage',
                    clim=dict(kind='value',pos_lims=[d_min,(d_max+d_min)/2,d_max]),
                    time_viewer=False, show_traces=False,background='w',colorbar=True,colormap='mne',
                    add_data_kwargs=dict(colorbar_kwargs=dict(label_font_size=10)))
            print(comparision)
            print(d_min,(d_min+d_max)/2,d_max)
            brain.save_image('RSA_%s_%s.pdf'%(task,comparision))

fig.savefig('RSA_source_stimuli1.pdf')

