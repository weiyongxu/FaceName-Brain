import os.path as op
import numpy as np

import mne
from mne.coreg import Coregistration
from mne.io import read_info
from config_FaceName import MEG_data_path,MRI_data_path,group_name,Ids,bids_root 
from mne_bids import BIDSPath

for subject_id in Ids:

    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
    subjects_dir=MRI_data_path

    bids_path = BIDSPath(subject='%02d'%(subject_id), task='FaceName', suffix='meg', datatype='meg', processing='tsss', split=1, extension='.fif', root=bids_root)

    info = read_info(bids_path.fpath)
    plot_kwargs = dict(subject=subject, subjects_dir=subjects_dir,
                    surfaces=['brain','head'], dig=True, eeg=[],
                    meg=False, show_axes=True,
                    coord_frame='meg')
    view_kwargs = dict(azimuth=45, elevation=90, distance=0.6,
                    focalpoint=(0., 0., 0.))

    fiducials = "estimated"  # get fiducials from fsaverage
    coreg = Coregistration(info, 'fsaverage', subjects_dir, fiducials=fiducials)
    coreg.fit_fiducials(verbose=True)
    coreg.set_scale_mode('uniform') # 3-axis
    coreg.fit_icp(verbose=True)
    coreg.omit_head_shape_points(distance=10. / 1000)  # distance is in meters
    coreg.fit_icp(verbose=True)
    mne.scale_mri(subject_from='fsaverage', subject_to=subject, scale=coreg.scale, overwrite=True, subjects_dir=MRI_data_path, skip_fiducials=False, labels=True, annot=True)

    fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
    mne.viz.set_3d_view(fig, **view_kwargs)

    dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
    print(
        f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
        f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
    )

    mne.write_trans(fname.replace('_tsss_mc.fif','-trans.fif'), coreg.trans,overwrite=True)

    #check coreg
    trans = op.join(fname.replace('_tsss_mc.fif','-trans.fif'))
    info = mne.io.read_info(bids_path.fpath)
    aln = mne.viz.plot_alignment(info, trans, subject=subject, subjects_dir=MRI_data_path,dig=True,meg='helmet',surfaces=['brain','head'])
    