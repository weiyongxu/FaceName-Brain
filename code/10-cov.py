import mne
import os.path as op
from config_FaceName import MEG_data_path,group_name,Ids

for subject_id in Ids:
    
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
           
    epo_E=mne.read_epochs(fname.replace("_tsss_mc", "-epo_E"))
    epo_R=mne.read_epochs(fname.replace("_tsss_mc", "-epo_R"))

    epo_E.apply_baseline((None,0)) 
    epo_R.apply_baseline((None,0)) 

    rank = mne.compute_rank(epo_R, tol=1e-6, tol_kind='relative')
    cov = mne.compute_covariance([epo_E,epo_R],n_jobs=20, tmin=None,tmax=0, method='shrunk',rank=rank)
    
    cov.save(fname.replace('_tsss_mc.fif','-cov.fif'))