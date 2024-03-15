import mne
import os.path as op
import numpy as np
from config_FaceName import MEG_data_path,group_name,Ids,bids_root
from mne.preprocessing import read_ica
import pandas as pd
from mne_bids import BIDSPath

for subject_id in Ids:
    
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
    
    bids_path = BIDSPath(subject='%02d'%(subject_id), task='FaceName', suffix='meg', datatype='meg', processing='tsss', split=1, extension='.fif', root=bids_root)
    raw_tsss_mc=mne.io.read_raw_fif(bids_path.fpath,preload=True)
            
    if op.isfile(fname.replace("tsss_mc", "annot")):
        raw_tsss_mc.set_annotations(mne.read_annotations(fname.replace("tsss_mc", "annot")))
        print("Annotion loaded!")
    raw_tsss_mc.filter(l_freq=1, h_freq=40.0)  # band-pass filter data      
    
    
    ica=read_ica(fname.replace("_tsss_mc.fif",'-ica.fif'))
    ICA_reject_threshold=np.load(fname.replace("tsss_mc.fif",'ICA_reject_threshold.npy'))        
    
    df=pd.read_csv(fname.replace('_tsss_mc.fif','-events.csv'))
    df=df[df['stim']=='F'] #only select face trials
    events=df[["stim_onset", "button_press", "trigger_code"]].astype(int).values
    epochs=mne.Epochs(raw_tsss_mc, events=events, tmin=-0.2, tmax=1,decim=5,reject=dict(grad=ICA_reject_threshold[0], mag=ICA_reject_threshold[1])).drop_bad()

    #remove EOG+ECG              
    eog_idx,eog_scores=ica.find_bads_eog(raw_tsss_mc)
    ecg_idx,ecg_scores=ica.find_bads_ecg(raw_tsss_mc)
    
    print('EOG index: %s'%eog_idx)
    ica.plot_scores(eog_scores, exclude=eog_idx, labels='eog')
    print('ECG index: %s'%ecg_idx)
    ica.plot_scores(ecg_scores, exclude=ecg_idx, labels='ecg')    
    
    print ('---'*20)
    print ('Pick component related to the EOG and ECG')
    
    if op.isfile(fname.replace("tsss_mc.fif",'ICA_excludes_EOG_EKG.npy')):
        ICA_excludes_EOG_EKG=np.load(fname.replace("tsss_mc.fif",'ICA_excludes_EOG_EKG.npy')).tolist()
        print('Load previous EOG+ECG index: %s'%ICA_excludes_EOG_EKG)
        ica.exclude.extend(ICA_excludes_EOG_EKG)
        ica.exclude=list(set(ica.exclude))
    
    print('EOG+ECG index: %s'%ica.exclude)
    ica.plot_sources(raw_tsss_mc,block = True,overview_mode='hidden')   

    print('EOG+ECG index: %s'%ica.exclude)
    ica.plot_components(inst=epochs)
    
    print('EOG+ECG index: %s'%ica.exclude)
    ica.plot_sources(raw_tsss_mc,block = True,overview_mode='hidden')   

    ICA_excludes_EOG_EKG=ica.exclude.copy() # !use copy method, otherwise the ICA_excludes_EOG_EKG will change later
    np.save(fname.replace("tsss_mc.fif",'ICA_excludes_EOG_EKG.npy'),ICA_excludes_EOG_EKG) 
    print ('ICA_excludes_EOG_EKG saved:%s'%ICA_excludes_EOG_EKG)