import mne
import mne_rsa
import os.path as op
import numpy as np
import pandas as pd
from skimage import io, color
from mne.minimum_norm import (make_inverse_operator, apply_inverse_epochs)
from config_FaceName import MEG_data_path,group_name,Ids,stim_folder,stimuli_list,RSA_data_path,MRI_data_path

rsa_metric='partial-spearman'
tasks=["E","R"]

src = mne.read_source_spaces(MRI_data_path+'/fsaverage/bem/fsaverage-ico-4-src.fif')
resample_rate=200

for task in tasks:
    for subject_id in Ids:

        subject = group_name+"%02d" % subject_id
        print("processing subject: %s" % subject)        
        fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')

        epochs=mne.read_epochs(fname.replace("_tsss_mc", "-epo_%s"%task))
            
        epochs=epochs['stim=="F"'].resample(resample_rate).pick_types(meg=True)
        
        epochs.drop(epochs.metadata.correct.isna() | epochs.metadata.rating.isna() | epochs.metadata.RT_memory.isna() | epochs.metadata.RT_rating.isna() )
        epochs.events[:,2] = np.arange(len(epochs))
    
        epochs.metadata['UID']=epochs.metadata.trigger_code-(epochs.metadata.trigger_code/100).astype(int)*100
        epochs.metadata=pd.merge(epochs.metadata,stimuli_list,how='left',on='UID')

        # stimuli related DSM
        df_gray=epochs.metadata['File_path_rel'].apply(lambda x: np.ravel(color.rgb2gray(io.imread(op.join(stim_folder,x)))))
        dsm_pixel_gray=mne_rsa.compute_rdm(np.array(df_gray.to_list()),metric='correlation')        
        dsm_ID = mne_rsa.compute_rdm(epochs.metadata[['Name']],metric=lambda a, b: 0 if a==b else 1)
        dsm_age = mne_rsa.compute_rdm(epochs.metadata[['Age']],metric='euclidean')
        dsm_gender = mne_rsa.compute_rdm(epochs.metadata[['Gender']],metric=lambda a, b: 0 if a==b else 1)

        # learning related DSM
        dsm_correct = mne_rsa.compute_rdm(epochs.metadata[['correct']],metric='euclidean')
        dsm_rating = mne_rsa.compute_rdm(epochs.metadata[['rating']],metric='euclidean')
        dsm_rt_memory = mne_rsa.compute_rdm(epochs.metadata[['RT_memory']],metric='euclidean')
        dsm_rt_rating = mne_rsa.compute_rdm(epochs.metadata[['RT_rating']],metric='euclidean')
        
        dsm_models_sel=[dsm_pixel_gray,dsm_ID,dsm_age,dsm_gender, dsm_correct,dsm_rating,dsm_rt_memory,dsm_rt_rating] #
        dsm_models_sel_name=['pixel_gray','ID','age','gender','correct','rating','rt_memory','rt_rating'] # 

        # sensor
        cov = mne.read_cov(fname.replace('_tsss_mc.fif','-cov.fif'))
        evoked_rsa = mne_rsa.rsa_epochs(epochs.copy(), dsm_models_sel, noise_cov=cov, rsa_metric=rsa_metric,spatial_radius=None, temporal_radius=0.01,picks='meg', verbose=True,n_jobs=-1) 
                                        
        for idx,evk in enumerate(evoked_rsa):
            evk.comment=dsm_models_sel_name[idx]
        
        mne.write_evokeds(op.join(RSA_data_path,'fname'+'_%02d'%(subject_id)+'-%s-rsa_sensor-%s_all_chs.fif'%(task,rsa_metric)), evoked_rsa,overwrite=True)
        
        # source ROI
        cov = mne.read_cov(fname.replace('_tsss_mc.fif','-cov.fif'))
        forward = mne.read_forward_solution(fname.replace('_tsss_mc.fif','-meg-ico4-fwd.fif'))
        inv = make_inverse_operator(epochs.info, forward, cov, loose=1.0, depth=0.8) 
        epochs_stc = apply_inverse_epochs(epochs, inv, lambda2=0.1111, pick_ori='normal')

        rois = mne.read_labels_from_annot(parc="aparc", subject=subject, subjects_dir=MRI_data_path)
        rsa_vals, stcs = mne_rsa.rsa_stcs_rois(
            epochs_stc,
            dsm_models_sel,
            inv["src"],
            rois[:-1],
            temporal_radius=0.01,
            n_jobs=-1,
            verbose=True,
        )

        evokeds=dict()
        for idx,cond in enumerate(dsm_models_sel_name):
            #save timecourse into an evoked object
            info = mne.create_info(ch_names=[label.name for label in rois[:-1]], sfreq=1.0 / stcs[idx].tstep, ch_types='misc')
            evoked = mne.EvokedArray(rsa_vals[:,:,idx], info, tmin=stcs[idx].tmin, comment=cond)
            evokeds[cond]=evoked
        mne.write_evokeds(op.join(RSA_data_path,'fname'+'_%02d'%(subject_id)+'-%s-rsa_ROI_normal_source-ave.fif'%(task)), list(evokeds.values()),overwrite=True)