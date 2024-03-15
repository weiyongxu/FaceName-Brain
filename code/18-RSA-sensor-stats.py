from config_FaceName import group_name,Ids,RSA_data_path
import mne
import os.path as op
from custom_functions import prepare_X_and_one_data,prepare_stats_list,print_sig_stats
from scipy import stats as stats
import numpy as np
from mne.stats import permutation_cluster_1samp_test
import matplotlib.pyplot as plt

Ids.remove(18)

rsa_metric='partial-spearman'
conditions=['pixel_gray','ID','gender','age']
#conditions=['correct','rating','rt_memory','rt_rating']
CB_color_cycle = ['#4daf4a','#377eb8', '#984ea3', '#ff7f00']
colors = {condition: CB_color_cycle[i % len(CB_color_cycle)] for i, condition in enumerate(conditions)}

fig, ax = plt.subplots(len(conditions), 2, squeeze=False, gridspec_kw={'width_ratios': [2, 1]}, figsize=(8, 6))

for task, task_idx in zip(['E', 'R'], [0, 1]):

    evks_all = {condtion: [] for condtion in conditions}

    for subject_id in Ids:

        subject = group_name + "%02d" % subject_id
        print("processing subject: %s" % subject)
        RSAs = mne.read_evokeds(op.join(RSA_data_path, 'fname' + '_%02d' % (subject_id) + '-%s-rsa_sensor-%s_all_chs.fif' % (task, rsa_metric)))
        for ev in RSAs:
            if ev.comment in conditions:
                evks_all[ev.comment].append(ev)

    cluster_method = 'tfce'
    p_threshold_cluster = 0.05
    cluster_p_sig = 0.05 / 2.0
    time_window = (0, None)

    for cmp_idx, comparision in enumerate(conditions):
        evokeds = {k: v for k, v in evks_all.items() if k == comparision}

        X, one_data = prepare_X_and_one_data(evokeds, ch_picks='misc', baseline=None, cmb_grad=False, time_window=time_window)
        data_type = type(one_data)

        X = np.squeeze(X)
        X = np.arctanh(X)

        t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(X, threshold=dict(start=0, step=0.2), adjacency=None, seed=1)
        stats_list = prepare_stats_list(t_obs, clusters, cluster_pv, H0, one_data, cluster_method, cluster_p_sig)
        sig_times = print_sig_stats(stats_list, cluster_method, cluster_p_sig)
        print(comparision)
        #[-0.025, 0.05] [-0.02, 0.02],,[-0.030, 0.030]
        mne.viz.plot_compare_evokeds({comparision: evks_all[comparision]}, picks='misc', ci=True, axes=ax[cmp_idx, task_idx], show_sensors=False, title='',truncate_yaxis=False, truncate_xaxis=False ,ylim=dict(misc=[-0.02, 0.02]), colors={comparision: colors[comparision]})
        if sig_times is not None:
            ax[cmp_idx, task_idx].plot(sig_times, [0.0] * len(sig_times), '.',color=colors[comparision], markersize=5)

fig.savefig('RSA_sensor_stimuli.pdf')