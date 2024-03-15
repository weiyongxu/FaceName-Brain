import matplotlib.pyplot as plt
import mne
import os.path as op
import numpy as np
from mne.channels.layout import _merge_grad_data
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test
import glob 
from copy import deepcopy
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

def prepare_X_and_one_data(data,time_window=(0,None),freq_window=(None,None),ch_picks=None,baseline=(-0.2, 0.0),baseline_mode='percent',cmb_grad=True):
    conditions=list(data.keys())
    one_data=data[conditions[0]][0].copy()
    data_type=type(one_data)
    
    if data_type==mne.time_frequency.AverageTFR: # n_channels, n_freqs, n_times
        X=np.array([[tfr.copy().pick(ch_picks).apply_baseline(mode=baseline_mode, baseline=baseline).crop(time_window[0],time_window[1],freq_window[0],freq_window[1]).data for tfr in data[cond]] for cond in conditions])
        one_data=one_data.pick(ch_picks).apply_baseline(mode=baseline_mode, baseline=baseline).crop(time_window[0],time_window[1],freq_window[0],freq_window[1])
    elif data_type==mne.evoked.Evoked and ch_picks=='grad': # n_channels, n_times
        if cmb_grad==True:
            X=np.array([[_merge_grad_data(evk.copy().pick(ch_picks).apply_baseline(baseline=baseline).crop(time_window[0],time_window[1]).data) for evk in data[cond]] for cond in conditions])
            # !combined grad use 'mag'
            one_data=one_data.pick('mag').apply_baseline(baseline=baseline).crop(time_window[0],time_window[1]) 
        else:
            X=np.array([[evk.copy().pick(ch_picks).apply_baseline(baseline=baseline).crop(time_window[0],time_window[1]).data for evk in data[cond]] for cond in conditions])
            one_data=one_data.pick(ch_picks).apply_baseline(baseline=baseline).crop(time_window[0],time_window[1])            
    elif data_type==mne.evoked.Evoked and ch_picks in ['mag','eeg','misc']: # n_channels, n_times
        X=np.array([[evk.copy().pick(ch_picks).apply_baseline(baseline=baseline).crop(time_window[0],time_window[1]).data for evk in data[cond]] for cond in conditions])
        one_data=one_data.pick(ch_picks).apply_baseline(baseline=baseline).crop(time_window[0],time_window[1])
    elif data_type==mne.source_estimate.SourceEstimate: # n_dipoles, n_times
        X=np.array([[stc.copy().crop(time_window[0],time_window[1]).data for stc in data[cond]] for cond in conditions])
        one_data=one_data.crop(time_window[0],time_window[1])  
    
    return X, one_data


def prepare_adjacency(one_data,src=None):
    data_type=type(one_data)
    if data_type==mne.source_estimate.SourceEstimate:
        adjacency = mne.spatial_src_adjacency(src)    
    else:
        channel_types=one_data.get_channel_types()
        if len(set(channel_types))>1:
            raise ValueError('mixed types of sensors: %s'%set(channel_types)) #make sure to use one sensor type
    
        if data_type==mne.time_frequency.tfr.AverageTFR:  
            sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(one_data.info, channel_types[0])
            adjacency = mne.stats.combine_adjacency(len(one_data.times),len(one_data.freqs),sensor_adjacency) #time × frequencies × space
            assert adjacency.shape[0] == adjacency.shape[1] == len(one_data.ch_names) * len(one_data.freqs) * len(one_data.times)   
        else:    
            adjacency, ch_names = find_ch_adjacency(one_data.info, channel_types[0]) #combined grad also use mag adjacency in one_data
    return adjacency


def prepare_stats_list(t_obs, clusters, cluster_pv, H0,one_data,stats_threshold_type,cluster_p_sig):
    stats_list=list()
    stats_t_obs=one_data.copy()
    stats_t_obs.data=t_obs.T
    stats_t_obs.comment='t_obs'
    stats_list.append(stats_t_obs)
            
    if stats_threshold_type=='tfce':
        p_tfce=cluster_pv.reshape(t_obs.shape)
        stats_p_tfce=one_data.copy()
        stats_p_tfce.data=p_tfce.T
        stats_p_tfce.comment='p_tfce'
        stats_list.append(stats_p_tfce)        
    else:
        good_cluster_inds = np.where(cluster_pv < cluster_p_sig)[0]        
        sig_ps=cluster_pv[cluster_pv<cluster_p_sig]
        print('-'*60);print('found %d significant cluster(s), p value(s) : %s'%(len(sig_ps),sig_ps));print('-'*60)        
        if len(good_cluster_inds)>0:      
            t_obs_cmb = np.nan * np.ones_like(t_obs)
            for ii, cluster_ind in enumerate(good_cluster_inds):            
                t_obs_temp = np.nan * np.ones_like(t_obs)
                t_obs_temp[clusters[cluster_ind]] = t_obs[clusters[cluster_ind]]
                stats_t_cluster=one_data.copy()
                stats_t_cluster.data=t_obs_temp.T
                stats_t_cluster.comment='C#%d-p=%.05f'%(ii+1,cluster_pv[cluster_ind])
                stats_list.append(stats_t_cluster)
                
                t_obs_cmb[clusters[cluster_ind]] = t_obs[clusters[cluster_ind]]
           
            stats_t_cluster_cmb=one_data.copy()
            stats_t_cluster_cmb.data=t_obs_cmb.T
            stats_t_cluster_cmb.comment='all_C'
            stats_list.append(stats_t_cluster_cmb)
    
    return stats_list

def print_sig_stats(stats_list,stats_threshold_type,p_sig_tfce=0.05):
    
    if len(stats_list)<=1:
        print('-'*60);print('Not Significant');print('-'*60)
    else:
        if stats_threshold_type=='tfce':
            stat=deepcopy(stats_list[-1])
            stat.data[stat.data >p_sig_tfce] = np.nan
            t_idx=np.where(~np.isnan(stat.data))[-1]
            t_idx = np.unique(t_idx)
            sig_times=stat.times[t_idx]
            if len(sig_times)>0:
                print('-'*60);print('sig_times:%0.2f-%0.2f'%(sig_times.min(),sig_times.max()));print('-'*60)
                return sig_times
            else:
                print('-'*60);print('Not Significant');print('-'*60)
        if stats_threshold_type=='cluster':
            stats_clusters=deepcopy(stats_list[1:-1])
            for stats_cluster in stats_clusters:
                t_idx=np.where(~np.isnan(stats_cluster.data))[-1]
                t_idx = np.unique(t_idx)
                sig_times=stats_cluster.times[t_idx]
                if len(sig_times)>0:
                    print('-'*60);print(stats_cluster.comment+' sig_times:%0.2f-%0.2f'%(sig_times.min(),sig_times.max()));print('-'*60)
                    return sig_times
                else:
                    print('-'*60);print('Not Significant');print('-'*60)
                

def write_stats_list(stats_list,data_type,conditions,results_folder,stats_test_name,stats_threshold_type):    
    comparision_name='--'.join(conditions) if len(conditions)>1 else conditions[0]   
    fname_base="%s-%s-%s"%(stats_test_name,stats_threshold_type,comparision_name)
    if data_type==mne.evoked.Evoked:
        mne.write_evokeds(op.join(results_folder,fname_base+'-evk.fif'), stats_list,overwrite=True)
    elif data_type==mne.time_frequency.tfr.AverageTFR:
        mne.time_frequency.write_tfrs(op.join(results_folder,fname_base+'-tfr.h5'),tfr=stats_list, overwrite=True)
    elif data_type==mne.source_estimate.SourceEstimate:
        for stc in stats_list:
            stc.save(op.join(results_folder,fname_base+'-'+stc.comment),overwrite=True)    


def read_stats_list(data_type,conditions,results_folder,stats_test_name,stats_threshold_type):
    comparision_name='--'.join(conditions)    
    fname_base="%s-%s-%s"%(stats_test_name,stats_threshold_type,comparision_name)

    if data_type==mne.evoked.Evoked: 
        stats_list=mne.read_evokeds(op.join(results_folder,fname_base+'-evk.fif'))         
    elif data_type==mne.time_frequency.AverageTFR:
        stats_list=mne.time_frequency.read_tfrs(op.join(results_folder,fname_base+'-tfr.h5'))
    elif data_type==mne.source_estimate.SourceEstimate:
        stats_list=list()
        files = glob.glob(op.join(results_folder,fname_base+'*-lh.stc'))
        files=sorted(files, key = op.getmtime)    
        for file in files:
            stc=mne.read_source_estimate(file)
            stc.comment=op.basename(file).replace(fname_base+'-','').replace('-lh.stc','')
            stats_list.append(stc)
    
    return stats_list

def do_cluster_permutation_1samp_test(data,time_window=(0,None),freq_window=(None,None),ch_picks=None,src=None,
                                       baseline=(-0.2, 0.0),baseline_mode='percent',
                                       cluster_method='cluster',p_threshold_cluster = 0.05,cluster_p_sig=0.05,
                                       results_folder='/',stats_test_name='1samp_ttest',**stat_kwargs):
    """ 
    1.do cluster-based permutation 1-sample tests for sensor (ERF,TFR) and soucece data(SourceEstimate) data.
    2.save the stats results as the same data type of input data.    
    Parameters
    ----------
    data : dict
        dict with length of 1 or 2
        If len(data) is 2, then calculate the difference 
        values of the dict must be one of the follwoing data types: evokeds,tfrs,stcs 
    time_window : tuple 
        time window for the stats
    freq_window : tuple | None    
        frequency window for the stats (for tfr data)        
    ch_picks : 'grad' | 'mag' | 'eeg'  | None (all channels)
        channels for the stats (sensor level)
    src : instance of SourceSpaces | None
    baseline :  tuple of length 2
        baseline for the sensor level stats
    baseline_mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
        baseline mode for tfr data
    cluster_method: string
        cluster method for the stats, either 'cluster' or 'tfce'
    p_threshold_cluster: float
        p val for the cluster forming threshold
    cluster_p_sig : float
        p values for significant clusters
    results_folder: string
        location to save the results, default to the current folder
    stats_test_name: string
    stat_kwargs : dict
        stats parameters for the spatio_temporal_cluster_1samp_test function

    Returns
    -------
    stats_list: list 
        list of data type as input but with stats results
    Notes
    -----
    Source TF analysis is not implemented.
    """
          
    X,one_data=prepare_X_and_one_data(data,time_window,freq_window,ch_picks,baseline,baseline_mode)
    data_type=type(one_data)   
    adjacency=prepare_adjacency(one_data,src) 

    X=X[0,...]-X[1,...] if X.shape[0]==2 else np.squeeze(X)    
    if data_type==mne.time_frequency.tfr.AverageTFR:
        X=np.transpose(X,[0,3,2,1]) # observations × time × frequencies × space
    else:
        X=np.transpose(X,[0,2,1]) # observations × time × space

    if cluster_method=='cluster':
        threshold = -stats.distributions.t.ppf(p_threshold_cluster / (1.+(stat_kwargs['tail']==0)), len(X) - 1)
    elif cluster_method=='tfce':
        threshold = dict(start=0, step=0.2)
    else:
        threshold=None
        
    t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(X,threshold=threshold,adjacency=adjacency,**stat_kwargs)
    
    stats_list=prepare_stats_list(t_obs, clusters, cluster_pv, H0,one_data,cluster_method,cluster_p_sig)
    
    write_stats_list(stats_list,data_type,list(data.keys()),results_folder,stats_test_name=stats_test_name,stats_threshold_type=cluster_method)
    
    return stats_list

def eq_cond_name(eq_conditions,cond,join_name='_eq_'):
    reorder_list=eq_conditions.copy()
    reorder_list.remove(cond)
    reorder_list.insert(0, cond)
    eq_cond=join_name.join(reorder_list)
    return eq_cond

def generate_adjacency_matrix(distance_threshold=0.04,MRI_data_path=None,plot=False):
    labels = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=MRI_data_path, verbose=False)
    centroids = []
    hemispheres = []
    for label in labels[:-1]:
        center_of_mass_vertex = label.center_of_mass()
        center_of_mass_pos = label.pos[np.where(label.vertices == center_of_mass_vertex)[0][0]]
        centroids.append(center_of_mass_pos)
        hemispheres.append(label.hemi)
    centroids_array = np.array(centroids)
    distances = pdist(centroids_array, metric='euclidean')
    distance_matrix = squareform(distances)
    adjacency_matrix = distance_matrix < distance_threshold
    for i in range(len(hemispheres)):
        for j in range(i+1, len(hemispheres)):
            if hemispheres[i] != hemispheres[j]:
                adjacency_matrix[i, j] = 0
                adjacency_matrix[j, i] = 0

    if plot==True:
        #plot the distance matrix
        fig, ax = plt.subplots()
        cax = ax.matshow(distance_matrix, cmap='viridis')
        ax.set_xticks(range(len(labels)-1))
        ax.set_yticks(range(len(labels)-1))
        ax.set_xticklabels([label.name for label in labels[:-1]], rotation=90)
        ax.set_yticklabels([label.name for label in labels[:-1]])
        plt.colorbar(cax)
        plt.show()

        #plot the adjacency matrix with labels
        fig, ax = plt.subplots()
        cax = ax.matshow(adjacency_matrix, cmap='gray')
        ax.set_xticks(range(len(labels)-1))
        ax.set_yticks(range(len(labels)-1))
        ax.set_xticklabels([label.name for label in labels[:-1]], rotation=90)
        ax.set_yticklabels([label.name for label in labels[:-1]])
        plt.colorbar(cax)
        plt.show()

    #Convert to a SciPy sparse matrix
    sparse_adj_matrix = csr_matrix(adjacency_matrix)

    return sparse_adj_matrix

