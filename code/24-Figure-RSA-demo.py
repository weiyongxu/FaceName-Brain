from sklearn.manifold import MDS
import mne
import mne_rsa
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, color
from config_FaceName import MEG_data_path,group_name,Ids,stim_folder,stimuli_list,MRI_data_path
src_fname = MRI_data_path+'/fsaverage/bem/fsaverage-ico-4-src.fif'
src = mne.read_source_spaces(src_fname)


Ids.remove(18)
model_RSAs=[]
tasks=["E"]
resample_rate=200

for task in tasks:
    
    for subject_id in Ids:
    
        subject = group_name+"%02d" % subject_id
        print("processing subject: %s" % subject)        
        fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
        epochs=mne.read_epochs(fname.replace("_tsss_mc", "-epo_%s"%task))
            
        epochs=epochs['stim=="F"'].resample(resample_rate)
        
        epochs.drop(epochs.metadata.correct.isna() | epochs.metadata.rating.isna() | epochs.metadata.RT_memory.isna() | epochs.metadata.RT_rating.isna() )
        epochs.events[:,2] = np.arange(len(epochs))
    
        epochs.metadata['UID']=epochs.metadata.trigger_code-(epochs.metadata.trigger_code/100).astype(int)*100
        epochs.metadata=pd.merge(epochs.metadata,stimuli_list,how='left',on='UID')
    
        df_lab=epochs.metadata['File_path_rel'].apply(lambda x: np.ravel(color.rgb2lab(io.imread(op.join(stim_folder,x)))))
        df_gray=epochs.metadata['File_path_rel'].apply(lambda x: np.ravel(color.rgb2gray(io.imread(op.join(stim_folder,x)))))
        dsm_pixel_lab=mne_rsa.compute_dsm(np.array(df_lab.to_list()),metric='correlation')
        dsm_pixel_gray=mne_rsa.compute_dsm(np.array(df_gray.to_list()),metric='correlation')
       
        dsm_correct = mne_rsa.compute_dsm(epochs.metadata[['correct']],metric='euclidean')
        dsm_rating = mne_rsa.compute_dsm(epochs.metadata[['rating']],metric='euclidean')
        dsm_rt_memory = mne_rsa.compute_dsm(epochs.metadata[['RT_memory']],metric='euclidean')
        dsm_rt_rating = mne_rsa.compute_dsm(epochs.metadata[['RT_rating']],metric='euclidean')
     
        dsm_ID = mne_rsa.compute_dsm(epochs.metadata[['Name']],metric=lambda a, b: 0 if a==b else 1)
        dsm_age = mne_rsa.compute_dsm(epochs.metadata[['Age']],metric='euclidean')
        dsm_gender = mne_rsa.compute_dsm(epochs.metadata[['Gender']],metric=lambda a, b: 0 if a==b else 1)

        dsm_models_all=[dsm_pixel_gray,dsm_ID,dsm_age,dsm_gender,dsm_correct,dsm_rating,dsm_rt_memory,dsm_rt_rating]
        dsm_models_all_name=['pixel_gray','ID','age','gender','correct','rating','rt_memory','rt_rating']
        
        if subject_id == Ids[0]:
            fig = mne_rsa.plot_dsms(dsm_models_all,names=dsm_models_all_name,n_rows=2)
            fig.savefig(op.join(MEG_data_path,'fname'+'_%02d'%(subject_id)+'-DSM.png'))
            plt.close(fig)

        model_RSA=mne_rsa.rsa(dsm_models_all,dsm_models_all)    
        model_RSAs.append(model_RSA) 

GA_model_RSA=np.mean(np.array(model_RSAs),0)



# Create the MDS model
# X is a distance matrix (n x n)
X = 1-GA_model_RSA

labels=dsm_models_all_name
# Create the MDS model
mds = MDS(n_components=2, dissimilarity='precomputed')

# Fit the model to the distance matrix
X_transformed = mds.fit_transform(X)

import matplotlib.pyplot as plt

plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
plt.show()

for x, y, label in zip(X_transformed[:, 0], X_transformed[:, 1], labels):
    plt.text(x, y, label)

# Create the scatter plot
plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
plt.show()

plt.savefig('2dMDS.pdf')
        

# plot the ROI on the brain
import mne
from config_FaceName import MRI_data_path
Brain = mne.viz.get_brain_class()

brain = Brain(
    "fsaverage",
    "lh",
    "inflated",
    subjects_dir=MRI_data_path,
    cortex="low_contrast",
    background="white",
    size=(800, 600),
)
brain.add_annotation("aparc")

aparc = mne.read_labels_from_annot("fsaverage", "aparc", subjects_dir=MRI_data_path, verbose=False)
label=[l for l in aparc if l.name=='supramarginal-lh']

brain.add_label(label[0], color="red", borders=False)
brain.save_image("brain.pdf")