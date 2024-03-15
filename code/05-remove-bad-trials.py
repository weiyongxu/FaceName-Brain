import mne
import pandas as pd
import os.path as op
import numpy as np
from config_FaceName import MEG_data_path,group_name,Ids,bids_root
from mne.preprocessing import read_ica
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
    
    raw_tsss_mc.filter(l_freq=None,h_freq=40.0)  # band-pass filter data      

    df=pd.read_csv(fname.replace('_tsss_mc.fif','-events.csv'))
    events = df[["stim_onset", "button_press", "trigger_code"]].astype(int).values
    epochs=mne.Epochs(raw_tsss_mc, events=events, tmin=-0.5, tmax=1.5,preload=True,baseline=None,picks=['meg','eog'])
    
    ica=read_ica(fname.replace("_tsss_mc.fif",'-ica.fif'))
    ica.exclude = np.load(fname.replace("tsss_mc.fif",'ICA_excludes_EOG_EKG.npy')).tolist()     #EOG+EKG    
    ica.apply(epochs,exclude=ica.exclude)
    
    picks=mne.pick_channels(epochs.ch_names,[epochs.ch_names[0]]+epochs.ch_names[1::9]+epochs.ch_names[5::9]+epochs.ch_names[6::9])   
            
    epochs.drop_bad(reject=dict(grad=1500e-13, mag=5e-12))
    
    epochs.plot(n_channels=103,n_epochs=40,block = True,picks=picks, scalings='auto',overview_mode='hidden')

    epochs.plot_drop_log(subject=subject).savefig(fname.replace("tsss_mc.fif",'drop_bads_Manual.png'))
    
    df['artifacts']=~df['stim_onset'].isin(epochs.events[:,0])
    
    df.to_csv(fname.replace('tsss_mc.fif','events_with_artifacts_marked.csv'),index=False)