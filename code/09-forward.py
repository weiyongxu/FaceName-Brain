import mne
import os.path as op
from config_FaceName import MEG_data_path,group_name,Ids,MRI_data_path,bids_root 
from mne_bids import BIDSPath

for subject_id in Ids:
    
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
    bids_path = BIDSPath(subject='%02d'%(subject_id), task='FaceName', suffix='meg', datatype='meg', processing='tsss', split=1, extension='.fif', root=bids_root)

    src = mne.setup_source_space(subject, spacing='ico4',subjects_dir=MRI_data_path,add_dist=True,n_jobs=24)
    
    mne.write_source_spaces(fname.replace('tsss_mc.fif','ico4-src.fif'), src,overwrite=True)
    
    #without MRI
    info = mne.io.read_info(bids_path.fpath)
    trans_fname = fname.replace('_tsss_mc.fif','-trans.fif')
    bem_file=op.join(MRI_data_path,subject,'bem','%s-inner_skull-bem.fif'%(subject))            
    bem_sol=mne.make_bem_solution(bem_file)
    fwd = mne.make_forward_solution(info, trans=trans_fname, src=src, bem=bem_sol, meg=True, eeg=False)  
                             
    mne.write_forward_solution(fname.replace('_tsss_mc.fif','-meg-ico4-fwd.fif'), fwd, overwrite=True)
    
