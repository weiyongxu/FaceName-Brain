#MNE version: 1.1.0

import os 
import pandas as pd
from custom_functions import eq_cond_name

study_path='/nashome1/wexu/Projects/FaceName-Brain/'
bids_root = '/nashome1/wexu/BIDS_Datasets/FaceName BIDS/derivatives/maxfilter/'

group_name='FN'

MRI_data_path = os.path.join(study_path,'data','subjects')
MEG_data_path = os.path.join(study_path,'data', 'MEG')
RSA_data_path = os.path.join(study_path,'data', 'RSA')
Results_data_path=os.path.join(study_path, 'results')

stim_folder=os.path.join(study_path,'experiment')
stimuli_list=pd.read_csv(os.path.join(study_path,'experiment','FN_stimuli_block_1.csv'))

os.environ["SUBJECTS_DIR"] = MRI_data_path


Ids=list(range(1,34))
Ids.remove(6) #6 is not measured

delay=25 #trigger delay(ms)

conditions_encode=dict()
conditions_encode['encode_hc_hit']='task=="E" & stim=="F" & rating>3 & correct==True'
conditions_encode['encode_lc_hit']='task=="E" & stim=="F" & rating<= 3 & correct==True'
conditions_encode['encode_hc_miss']='task=="E" & stim=="F" & rating>3 & correct==False'
conditions_encode['encode_lc_miss']='task=="E" & stim=="F" & rating<= 3 & correct==False'

conditions_recall=dict()
conditions_recall['recall_hc_hit']='task=="R" & stim=="F" & rating>3 & correct==True'
conditions_recall['recall_lc_hit']='task=="R" & stim=="F" & rating<= 3 & correct==True'
conditions_recall['recall_hc_miss']='task=="R" & stim=="F" & rating>3 & correct==False'
conditions_recall['recall_lc_miss']='task=="R" & stim=="F" & rating<= 3 & correct==False'

eq_conditions_encode=[['encode_hc_hit','encode_lc_hit'],['encode_lc_hit','encode_lc_miss']]
eq_conditions_recall=[['recall_hc_hit','recall_lc_hit'],['recall_lc_hit','recall_lc_miss']]

stats_conds_list=[eq_cond_name(eq_conds,cond,join_name='_eq_') for eq_conds in eq_conditions_encode+eq_conditions_recall for cond in eq_conds]
stats_comparisions=[[eq_cond_name(eq_conds,cond,join_name='_eq_') for cond in eq_conds] for eq_conds in eq_conditions_encode+eq_conditions_recall]
