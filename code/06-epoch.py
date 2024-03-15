import mne
import os.path as op
from config_FaceName import MEG_data_path,group_name,Ids,bids_root
from mne.preprocessing import read_ica
import numpy as np
import pandas as pd
from mne_bids import BIDSPath


for subject_id in Ids:
    
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
    
    bids_path = BIDSPath(subject='%02d'%(subject_id), task='FaceName', suffix='meg', datatype='meg', processing='tsss', split=1, extension='.fif', root=bids_root)
    raw_tsss_mc=mne.io.read_raw_fif(bids_path.fpath,preload=True)
    
    raw_tsss_mc.filter(l_freq=0.1, h_freq=40.0)  # low-pass filter data
    ica = read_ica(fname.replace("_tsss_mc", "-ica"))
    ica.exclude = np.load(fname.replace("tsss_mc.fif",'ICA_excludes_EOG_EKG.npy')).tolist()     #EOG_EKG
    raw_tsss_mc = ica.apply(raw_tsss_mc, exclude=ica.exclude)

    events=pd.read_csv(fname.replace('_tsss_mc.fif','-events.csv'))

    df=pd.read_csv(fname.replace('tsss_mc.fif','events_with_artifacts_marked.csv'))    
    
    df=df[df['artifacts']==False]
    df_E=df[(df['stim']=='F') & (df['task']=='E')]
    df_R=df[(df['stim']=='F') & (df['task']=='R') &(df['face_test']==False)]
    
    events_E = df_E[["stim_onset", "button_press", "trigger_code"]].astype(int).values
    events_R = df_R[["stim_onset", "button_press", "trigger_code"]].astype(int).values

    picks = mne.pick_types(raw_tsss_mc.info, meg=True, eeg=False, stim=False, eog=True,exclude='bads')

    epochs_E = mne.Epochs(raw_tsss_mc, events_E, event_id=None, tmin=-0.2, tmax=2.0, picks=picks, baseline=(None,0), preload=True,reject=None,proj=True,metadata=df_E)        
    epochs_R = mne.Epochs(raw_tsss_mc, events_R, event_id=None, tmin=-0.2, tmax=1.0, picks=picks, baseline=(None,0), preload=True,reject=None,proj=True,metadata=df_R)        
    
    epochs_E.save(fname.replace("_tsss_mc", "-epo_E"),overwrite=True)
    epochs_R.save(fname.replace("_tsss_mc", "-epo_R"),overwrite=True)
    