import mne
import re
import os.path as op
import os
import subprocess
from config_FaceName import MEG_data_path,group_name,Ids

def mark_bad_channel(fname):
    RAW=mne.io.Raw(fname,allow_maxshield=True,preload=True).pick_types(meg=True)
    RAW.filter(None,80).plot(duration=20,n_channels=102,block=True) #,block=True
    if RAW.info['bads']:
        print(*[int(re.findall('\d+', s)[0]) for s in RAW.info['bads']],file=open(fname+'_bad.txt', 'a'))    
    
def do_maxfilter(MEG_data_path,subject,fname):      
    os.chdir(op.join(MEG_data_path,subject,""))
    command="maxfilter-3.0 -corr 0.95 -autobad off -v -force -st -regularize svd -movecomp -hpicons -frame head -origin fit -f "+fname+" -bad $(<"+fname+'_bad.txt)'+" | tee "+fname+'_maxfilter_log.txt'
    subprocess.run(command,shell=True)    
    

def annotate_bads(fname):    
    RAW=mne.io.Raw(fname,allow_maxshield=True,preload=True).pick_types(meg=True)

    if os.path.isfile(fname.replace("tsss_mc", "annot")):
        RAW.set_annotations(mne.read_annotations(fname.replace("tsss_mc", "annot")))

    mne.Epochs(RAW, events=mne.make_fixed_length_events(RAW), tmin=0, tmax=1, baseline=None,reject = dict(grad=4000e-13, mag=4e-12)).drop_bad().plot_drop_log().savefig(fname.replace("tsss_mc.fif",'drop_bads.png'))

    RAW.plot(lowpass=80,duration=60,n_channels=102,block=True) #,block=True

    if RAW.info['bads']:
        print(*[int(re.findall('\d+', s)[0]) for s in RAW.info['bads']],file=open(fname+'_bad.txt', 'a'))

    if RAW.annotations:
        RAW.annotations.save(fname.replace("tsss_mc", "annot"))    

        
for subject_id in Ids:
    
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)            
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'.fif')
    fname_tsss=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
    
    mark_bad_channel(fname)
    do_maxfilter(MEG_data_path,subject,fname)
    annotate_bads(fname_tsss)    
    
