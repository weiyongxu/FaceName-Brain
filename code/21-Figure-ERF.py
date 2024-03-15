import mne
from config_FaceName import Results_data_path,stats_comparisions,MRI_data_path,Results_data_path
from custom_functions import read_stats_list, print_sig_stats

cluster_method='tfce'
tfce_p_sig=0.05/2.0 # adjust for sensor and source stat tests

#sensor stats
for comparision in stats_comparisions:
    print(comparision)
    stats_list_evk=read_stats_list(mne.evoked.Evoked,comparision,Results_data_path,'1samp_ttest',cluster_method)
    
    sig_times=print_sig_stats(stats_list_evk,cluster_method,tfce_p_sig)

    #check the sig_time first and then segment the sig_times list into two lists: one before 0.5s and one after 0.5s if needed
    if sig_times is not None:
        sig_times_before=sig_times[sig_times<=0.5]
        sig_times_after=sig_times[sig_times>=0.5]
        # define sig_times as a list of sig_times_before and sig_times_after if both lists are not empty
        if len(sig_times_before)>0 and len(sig_times_after)>0:
            sig_times=[sig_times_before,sig_times_after]
        else:
            sig_times=[sig_times]

    if sig_times is not None:
        
        stats_list_evk[0].data[stats_list_evk[1].data>tfce_p_sig]=0

        fig=stats_list_evk[0].plot_topomap(times=[(sig_time.max()+sig_time.min())/2 for sig_time in sig_times],average=[sig_time.max()-sig_time.min() for sig_time in sig_times],time_format='%0.2f s',units='t-TFCE',scalings=dict(mag=1,grad=1),
                        mask=stats_list_evk[1].data<tfce_p_sig,mask_params=dict(markersize=8, markerfacecolor='y'),contours=0) #vlim=[-30,30]

        #save the topomap
        fig.savefig(' '.join(comparision)+'-ERF-topomap.pdf')

#source stats
for comparision in stats_comparisions:
    print(comparision)
    stats_list_stc=read_stats_list(mne.source_estimate.SourceEstimate,comparision,Results_data_path,'1samp_ttest',cluster_method)
    sig_times=print_sig_stats(stats_list_stc,cluster_method,tfce_p_sig)
    if sig_times is not None:
        sig_times_before=sig_times[sig_times<=0.5]
        sig_times_after=sig_times[sig_times>=0.5]
        if len(sig_times_before)>0 and len(sig_times_after)>0:
            sig_times=[sig_times_before,sig_times_after]
        else:
            sig_times=[sig_times]

        TWs=[[sig_time.min(),sig_time.max()] for sig_time in sig_times]
        
        stats_list_stc[0].data[stats_list_stc[1].data>tfce_p_sig]=0
 
        stats_bins=[]
        for TW in TWs:
            stats_bins.append(stats_list_stc[0].bin(width=TW[1]-TW[0],tstart=TW[0],tstop=TW[1]+stats_list_stc[0].tstep))

        #find the max and min value of the stats_bins
        d_max=max([stats_bin.data.max() for stats_bin in stats_bins])
        d_min=min([stats_bin.data[stats_bin.data!=0].min() for stats_bin in stats_bins])
        #plot the source space
        for TW,stats_bin in zip(TWs,stats_bins):           
            brain =stats_bin.plot(views=['lat','med'], hemi='split', size=(600, 600), subject='fsaverage',
                                    subjects_dir=MRI_data_path, initial_time=stats_bin.times[0], 
                                    clim=dict(kind='value',lims=[d_min,(d_min+d_max)/2,d_max]) ,                                                              
                                    time_viewer=False, show_traces=False,background='w',colorbar=True,
                                    add_data_kwargs=dict(colorbar_kwargs=dict(label_font_size=10)))
            print(d_min,(d_min+d_max)/2,d_max)
            brain.save_image(' '.join(comparision)+'-ERF-'+str(TW[0])+'-'+str(TW[1])+'_cb.pdf')    
            brain.close()