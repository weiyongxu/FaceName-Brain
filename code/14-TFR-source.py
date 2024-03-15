import mne
import numpy as np
import pandas as pd
import os.path as op
from config_FaceName import MEG_data_path,group_name,Ids,bids_root,MRI_data_path,conditions_encode,conditions_recall,eq_conditions_encode,eq_conditions_recall
from mne.preprocessing import read_ica
from mne_bids import BIDSPath
from mne.minimum_norm import make_inverse_operator, source_band_induced_power
from custom_functions import eq_cond_name

src_fname = MRI_data_path+'/fsaverage/bem/fsaverage-ico-4-src.fif'
src = mne.read_source_spaces(src_fname)

bands = dict(theta=[4, 7],alpha=[8, 13],beta=[14, 30],gamma=[40, 90])
n_cycles =5

for subject_id in Ids:
    
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
    bids_path = BIDSPath(subject='%02d'%(subject_id), task='FaceName', suffix='meg', datatype='meg', processing='tsss', split=1, extension='.fif', root=bids_root)
    
    raw_tsss_mc=mne.io.read_raw_fif(bids_path.fpath,preload=True)  
    
    ica = read_ica(fname.replace("_tsss_mc", "-ica"))
    ica.exclude = np.load(fname.replace("tsss_mc.fif",'ICA_excludes_EOG_EKG.npy')).tolist()  #EOG_EKG
    raw_tsss_mc = ica.apply(raw_tsss_mc, exclude=ica.exclude)    
    df=pd.read_csv(fname.replace('tsss_mc.fif','events_with_artifacts_marked.csv'))        
    
    cov = mne.read_cov(fname.replace('_tsss_mc.fif','-cov.fif'))
    forward = mne.read_forward_solution(fname.replace('_tsss_mc.fif','-meg-ico4-fwd.fif')) 
    
    for task,t_max,conds_list,eq_conds_list in zip(['E','R'],[3,2],[conditions_encode,conditions_recall],[eq_conditions_encode,eq_conditions_recall]):
        
        df2=df.query('artifacts==False & task=="%s" & stim=="F" & face_test==False'%(task))   
        events = df2[["stim_onset", "button_press", "trigger_code"]].astype(int).values
        picks = mne.pick_types(raw_tsss_mc.info, meg=True, eeg=False, eog=False, stim=False)
        epochs = mne.Epochs(raw_tsss_mc, events=events, event_id=None, tmin=-1.5, tmax=t_max, picks=picks,baseline=None, preload=True,metadata=df2)
        inv = make_inverse_operator(epochs.info, forward, cov, loose=1.0, depth=0.8)       
    
        for eq_conds in eq_conds_list:
            epo_li=[epochs[conds_list[cond]] for cond in eq_conds]
            mne.epochs.equalize_epoch_counts(epo_li)
            for epo,cond in zip(epo_li,eq_conds):                
                epo.subtract_evoked()
                stcs =source_band_induced_power(epo, inv, bands=bands, n_cycles=n_cycles,n_jobs=5,baseline=(-0.2,0),baseline_mode='percent',decim=5)
                for b, stc in stcs.items():
                    stc=stc.crop(-0.5,t_max-1)                
                    stc_fsaverage = mne.compute_source_morph(stc,subjects_dir=MRI_data_path,src_to=src,smooth=15).apply(stc)                
                    stc_fsaverage.save(fname.replace("_tsss_mc.fif", '-%s-%s-%s_alt_rating_grouping' %(task,eq_cond_name(eq_conds,cond,join_name='_eq_'),b)), overwrite=True)       