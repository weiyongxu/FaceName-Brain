import os.path as op
import mne
from mne.minimum_norm import (make_inverse_operator, apply_inverse)
from config_FaceName import MEG_data_path,group_name,Ids,MRI_data_path,stats_conds_list
src_fname = MRI_data_path+'/fsaverage/bem/fsaverage-ico-4-src.fif'
src = mne.read_source_spaces(src_fname)

for subject_id in Ids:
    
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')

    evokeds = mne.read_evokeds(fname.replace("_tsss_mc", "_alt_rating_grouping-ave"),stats_conds_list)

    cov = mne.read_cov(fname.replace('_tsss_mc.fif','-cov.fif'))
    forward = mne.read_forward_solution(fname.replace('_tsss_mc.fif','-meg-ico4-fwd.fif'))
    inverse_operator = make_inverse_operator(evokeds[0].info, forward, cov, loose=1.0, depth=0.8) 

    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    
    methods=['dSPM']
    for method in methods:
        for evoked in evokeds:
            
            stc = apply_inverse(evoked.apply_baseline(baseline=(None,0)), inverse_operator, lambda2, method=method, pick_ori=None)
            stc_fsaverage = mne.compute_source_morph(stc,subjects_dir=MRI_data_path,src_to=src,smooth=15).apply(stc)
                           
            stc_fsaverage.save(op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'-'+evoked.comment+'_alt_rating_grouping-'+method),overwrite=True)
            
            
    