import mne
from config_FaceName import Results_data_path,MRI_data_path,stats_comparisions
import matplotlib.pyplot as plt
import numpy as np
from custom_functions import read_stats_list,print_sig_stats

cluster_method='tfce'
bands = dict(theta=[4, 7],alpha=[8, 13],beta=[14, 30],gamma=[40, 90])
tfce_p_sig=0.05/8.0 # adjust for sensor and source stat tests * 4 freq bands

#sensor space
for compare in stats_comparisions:
    for band_name, band in bands.items():
        stats_list_tfr=read_stats_list(mne.time_frequency.AverageTFR,compare,Results_data_path,band_name,cluster_method)
        sig_times=print_sig_stats(stats_list_tfr,cluster_method,tfce_p_sig)

        if sig_times is not None:
            vmax=np.abs(stats_list_tfr[0].copy().crop(tmin=sig_times[0],tmax=sig_times[-1]).data).max()
            sig_times_before=sig_times[sig_times<=0.5]
            sig_times_after=sig_times[sig_times>=0.5]
            # define sig_times as a list of sig_times_before and sig_times_after if both lists are not empty
            if len(sig_times_before)>0 and len(sig_times_after)>0:
                sig_times=[sig_times_before,sig_times_after]
            else:
                sig_times=[sig_times]

            TWs=[[sig_time.min(),sig_time.max()] for sig_time in sig_times]

            for TW in TWs:
                mask_temp=stats_list_tfr[1].copy().crop(tmin=TW[0],tmax=TW[1]).data<tfce_p_sig
                mask_temp=mask_temp.any(axis=1).all(axis=1)
                mask_temp=np.stack([mask_temp[0::2],mask_temp[1::2]]).any(axis=0)

                stats_list_tfr[0].data[stats_list_tfr[1].data>tfce_p_sig]=0

                stats_list_tfr[0].plot_topomap(tmin=TW[0],tmax=TW[1],mask=mask_temp,mask_params=dict(markersize=4, markerfacecolor='y'),vlim=(-vmax,vmax),contours=0)

                plt.savefig(' '.join(compare)+'-TFR-'+band_name+str(TW[0])+'-'+str(TW[1])+'.pdf')

#source space
for compare in stats_comparisions:
    for band in bands:
        comparision=[c+'_'+band for c in compare]    
        stats_list_stc=read_stats_list(mne.source_estimate.SourceEstimate,comparision,Results_data_path,'1samp_ttest',cluster_method)
        print(comparision)

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
            d_max=max([np.abs(stats_bin.data[stats_bin.data!=0]).max() for stats_bin in stats_bins])
            d_min=min([np.abs(stats_bin.data[stats_bin.data!=0]).min() for stats_bin in stats_bins])
            #plot the source space
            for TW,stats_bin in zip(TWs,stats_bins):           
                brain =stats_bin.plot(views=['lat','med'], hemi='split', size=(600, 600), subject='fsaverage',
                                        subjects_dir=MRI_data_path, initial_time=stats_bin.times[0], 
                                        clim=dict(kind='value',pos_lims=np.abs([d_min,(d_min+d_max)/2,d_max])) ,                                                              
                                        time_viewer=False, show_traces=False,background='w',colorbar=True,
                                        colormap='mne',add_data_kwargs=dict(colorbar_kwargs=dict(label_font_size=10)))
                print(d_min,(d_min+d_max)/2,d_max)
                brain.save_image(' '.join(comparision)+'-TFR-'+str(TW[0])+'-'+str(TW[1])+'_cb.pdf')    
                brain.close()
            