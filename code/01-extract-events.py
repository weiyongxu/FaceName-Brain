import mne
import numpy as np
import pandas as pd
import os.path as op
from config_FaceName import MEG_data_path,group_name,Ids,delay,bids_root,stimuli_list
from mne_bids import BIDSPath

for subject_id in Ids:

    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
    
    bids_path = BIDSPath(subject='%02d'%(subject_id), task='FaceName', suffix='meg', datatype='meg', processing='tsss', split=1, extension='.fif', root=bids_root)
    raw=mne.io.read_raw_fif(bids_path.fpath)
    events=mne.find_events(raw,stim_channel='STI101', verbose=True,min_duration=0.003)
    events[:,0]=events[:,0]+delay #fix delay
    events[:,2]=events[:,2]-events[:,1]
    
    df=pd.DataFrame(events,columns=['stim_onset','button_press','trigger_code']) #
    df.loc[df['trigger_code']==1024,'trigger_code']=1
    df.loc[df['trigger_code']==2048,'trigger_code']=2
    df.loc[df['trigger_code']==4096,'trigger_code']=3
    df.loc[df['trigger_code']==8192,'trigger_code']=4
    
    df['face_test']=False
    
    df[['repeat','stim','task','correct','rating','RT_memory','RT_rating']]=np.nan
    
    for idxx, row in stimuli_list.iterrows():
         for rep in [1,2,3,4,5,6]:
             # if the trial is a test trial
             idx=df.loc[df[df['trigger_code']==row['UID']+rep*100].index+1,'trigger_code'] == 40 
             df.loc[idx[idx==True].index-1,'face_test']=True 
             
             if len(df[df['trigger_code']==(row['UID']+rep*100)])==5:
                 df.loc[df['trigger_code']==row['UID']+rep*100,'repeat']=[rep,rep,rep,rep,rep]
                 df.loc[df['trigger_code']==row['UID']+rep*100,'stim']=['F','N','F','F','F']         
                 df.loc[df['trigger_code']==row['UID']+rep*100,'task']=['E','E','R','R','R']
                 
                 tmp=(df.loc[df[df['trigger_code']==row['UID']+rep*100].index+2,'trigger_code'] <= 4) # find the Response trigger 1,2,3,4
                 idx=tmp.index[tmp == True].tolist() #idx=resp trigger
                 
                 if len(idx)>=1:# if there was response given                                              
                     df.loc[df['trigger_code']==row['UID']+rep*100,'correct']=(df.loc[idx[0]+1,'trigger_code']==71) #71=correct resp
                     
                     if df.loc[idx[0]+2,'trigger_code']<= 4: #rating might be 2 or 3 triggers after test resp
                         df.loc[df['trigger_code']==row['UID']+rep*100,'rating']=df.loc[idx[0]+2,'trigger_code']
                         if df.loc[idx[0]+1,'trigger_code'] == 50:# RT_rating=resp - rating trial start(50)   
                             df.loc[df['trigger_code']==row['UID']+rep*100,'RT_rating']=df.loc[idx[0]+2,'stim_onset']-df.loc[idx[0]+1,'stim_onset']  
                     if df.loc[idx[0]+3,'trigger_code']<= 4: #rating might be 2 or 3 triggers after test resp
                         df.loc[df['trigger_code']==row['UID']+rep*100,'rating']=df.loc[idx[0]+3,'trigger_code']
                         if df.loc[idx[0]+2,'trigger_code'] == 50:# RT_rating=resp - rating trial start(50)   
                             df.loc[df['trigger_code']==row['UID']+rep*100,'RT_rating']=df.loc[idx[0]+3,'stim_onset']-df.loc[idx[0]+2,'stim_onset']                           
                     if df.loc[idx[0]-1,'trigger_code'] == 40:# RT_memory=resp - test trial start(40)   
                         df.loc[df['trigger_code']==row['UID']+rep*100,'RT_memory']=df.loc[idx[0],'stim_onset']-df.loc[idx[0]-1,'stim_onset']

                 else:# no response given, time out
                     df.loc[df['trigger_code']==row['UID']+rep*100,'correct']=False                     
                     df.loc[df['trigger_code']==row['UID']+rep*100,'rating']=1                   

    df=df[(df['trigger_code']>=109) & (df['trigger_code']<=1000)]
    df.to_csv(fname.replace('_tsss_mc.fif','-events.csv'),index=False)
    
    