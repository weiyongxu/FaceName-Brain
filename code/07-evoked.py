import mne
import os.path as op
from config_FaceName import MEG_data_path,group_name,Ids,conditions_encode,conditions_recall,eq_conditions_encode,eq_conditions_recall
from custom_functions import eq_cond_name


for subject_id in Ids:
    
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
    evokeds=dict()
    
    for task,conds_list,eq_conds_list in zip(['E','R'],[conditions_encode,conditions_recall],[eq_conditions_encode,eq_conditions_recall]):  
        
        epochs=mne.read_epochs(fname.replace("_tsss_mc", "-epo_%s"%task))
    
        for eq_conds in eq_conds_list:
            epo_list=[epochs[conds_list[cond]] for cond in eq_conds]
            mne.epochs.equalize_epoch_counts(epo_list)
            for epo,cond in zip(epo_list,eq_conds):
                eq_cond=eq_cond_name(eq_conds,cond,join_name='_eq_')
                evokeds[eq_cond]=epo.average()
                evokeds[eq_cond].comment=eq_cond

    mne.write_evokeds(fname.replace("_tsss_mc", "_alt_rating_grouping-ave"), list(evokeds.values()),overwrite=True)