import mne
import os.path as op
import numpy as np
from config_FaceName import MEG_data_path,group_name,Ids,bids_root
from mne.preprocessing import ICA
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
    
    ica = ICA(method='fastica',random_state=97)
        
    picks = mne.pick_types(raw_tsss_mc.info, meg=True, eeg=False, eog=False,stim=False, exclude='bads')
    
    ICA_reject_threshold=np.load(fname.replace("tsss_mc.fif",'ICA_reject_threshold.npy'))
    
    print(ICA_reject_threshold)
    
    ica.fit(raw_tsss_mc, picks=picks, reject=dict(grad=ICA_reject_threshold[0], mag=ICA_reject_threshold[1]),decim=5,reject_by_annotation=True) 
    ica.save(fname.replace("_tsss_mc.fif",'-ica.fif'))