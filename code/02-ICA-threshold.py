import mne
import os.path as op
import numpy as np
from config_FaceName import MEG_data_path,group_name,Ids,bids_root
print(Ids)
from mne_bids import BIDSPath


for subject_id in Ids:
    
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
    bids_path = BIDSPath(subject='%02d'%(subject_id), task='FaceName', suffix='meg', datatype='meg', processing='tsss', split=1, extension='.fif', root=bids_root)
    RAW=mne.io.read_raw_fif(bids_path.fpath,preload=True)
        
    if op.isfile(fname.replace("tsss_mc", "annot")):
        RAW.set_annotations(mne.read_annotations(fname.replace("tsss_mc", "annot")))
        print("Annotion loaded!")
    
    RAW.filter(l_freq=1, h_freq=40.0)  # band-pass filter data      
    
    epochs=mne.Epochs(RAW, events=mne.make_fixed_length_events(RAW),tmin=0, tmax=1, baseline=None,reject=None)
    
    ICA_reject_threshold = dict(grad=1500e-13, mag=4.00e-12)
    
    # less than 5% excluding the bad move artifacts
    while mne.Epochs(RAW, events=mne.make_fixed_length_events(RAW), tmin=0, tmax=1, baseline=None,reject=ICA_reject_threshold).drop_bad().drop_log_stats(ignore=('BAD_move','BAD_ACQ_SKIP')) >=5:        
        ICA_reject_threshold['mag']=ICA_reject_threshold['mag']+0.25e-12
        
    print(ICA_reject_threshold)
    
    epochs=mne.Epochs(RAW, events=mne.make_fixed_length_events(RAW), tmin=0, tmax=1, baseline=None,reject=ICA_reject_threshold).drop_bad()
    mne.viz.plot_drop_log(epochs.drop_log,subject=subject,show=False).savefig(fname.replace("tsss_mc.fif",'drop_bads_ICA.png'))
    
    np.save(fname.replace("tsss_mc.fif",'ICA_reject_threshold.npy'),np.array(list(ICA_reject_threshold.values())))