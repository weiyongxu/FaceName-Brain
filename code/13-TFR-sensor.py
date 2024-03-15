import mne
import numpy as np
import pandas as pd
import os.path as op
from config_FaceName import MEG_data_path,group_name,Ids,bids_root,conditions_encode,conditions_recall,eq_conditions_encode,eq_conditions_recall
from mne.preprocessing import read_ica
from mne.time_frequency import tfr_morlet
from mne_bids import BIDSPath
from custom_functions import eq_cond_name

freqs = np.logspace(*np.log10([4, 90]), num=30)
n_cycles =5 # fixed number of cycles    

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

    tfrs=list()
    for task,t_max,conds_list,eq_conds_list in zip(['E','R'],[3,2],[conditions_encode,conditions_recall],[eq_conditions_encode,eq_conditions_recall]):
        
        df2=df.query('artifacts==False & task=="%s" & stim=="F" & face_test==False'%(task))   
        events = df2[["stim_onset", "button_press", "trigger_code"]].astype(int).values
        picks = mne.pick_types(raw_tsss_mc.info, meg='grad', eeg=False, eog=False, stim=False)
        epochs = mne.Epochs(raw_tsss_mc, events=events, event_id=None, tmin=-1.5, tmax=t_max, picks=picks,baseline=None, preload=True,metadata=df2)
        
        for eq_conds in eq_conds_list:
            epo_li=[epochs[conds_list[cond]] for cond in eq_conds]
            mne.epochs.equalize_epoch_counts(epo_li)
            for epo,cond in zip(epo_li,eq_conds):                
                epo.subtract_evoked()
                power=tfr_morlet(epo, freqs=freqs, n_cycles=n_cycles, return_itc=False, decim=5,n_jobs=-1,average=True)
                power.comment=eq_cond_name(eq_conds,cond,join_name='_eq_')
                power.crop(-0.5,t_max-1)
                tfrs.append(power)
      
    mne.time_frequency.write_tfrs(fname.replace("_tsss_mc.fif", "_alt_rating_grouping-tfr.h5"),tfr=tfrs, overwrite=True)