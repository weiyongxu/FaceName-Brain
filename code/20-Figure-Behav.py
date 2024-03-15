
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as op
import pandas as pd
import numpy as np
from config_FaceName import MEG_data_path,group_name,Ids
import scipy.stats as stats
sns.reset_defaults()

def auroc2(accuracies:list, confidences:list, number_of_confidence_ratings:int) -> float:
    """Compute area under type 2 ROC to assess calibration of confidence ratings.
    Parameters
    ----------
    accuracies : list
        List of decisions, with 1 indicating correct decisions, 0 indicating errors.
    confidences : list
        Confidence ratings associated to each decision.
    number_of_confidence_ratings : int
        How many confidence levels are available. Assuming they go from 1 to this number.
    Returns
    -------
    float
       Area under the type 2 ROC.
    """
    assert len(confidences) == len(accuracies)
    accuracies = np.array(accuracies)
    confidences = np.array(confidences)
    hit_rates = np.zeros((number_of_confidence_ratings))
    false_alarm_rates = np.zeros((number_of_confidence_ratings))
    for c in range(1, number_of_confidence_ratings+1):
        hit_rates[number_of_confidence_ratings-c] = np.sum([(accuracies == 1) & (confidences == c)])+0.5
        false_alarm_rates[number_of_confidence_ratings-c] = np.sum([(accuracies == 0) & (confidences == c)])+0.5
    hit_rates /= np.sum(hit_rates)
    false_alarm_rates /= np.sum(false_alarm_rates)
    cumsum_hit = [0]+list(np.cumsum(hit_rates))
    cumsum_false_alarm = [0]+list(np.cumsum(false_alarm_rates))
    k = []
    for c in range(number_of_confidence_ratings):
        k.append((cumsum_hit[c+1]-cumsum_false_alarm[c])**2-(cumsum_hit[c]-cumsum_false_alarm[c+1])**2)
    auroc2 = 0.5 + 0.25*np.sum(k)
    return auroc2



df_list=[]
for idx,subject_id in enumerate(Ids):
    
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
 
    df=pd.read_csv(fname.replace('_tsss_mc.fif','_events_with_artifacts_marked.csv'))
    df=df[(df['task']=='E') & (df['stim']=='F')]
    df['ID_conti']=idx+1 #for plotting continous bars 
    df['ID']=subject_id #real Ids
    df=df[df['rating'].isin([1.0,2.0,3.0,4.0])]

    df['confidence']=df.apply(lambda x: 'HC' if x.rating >3 else 'LC', axis=1)
    df['accuracy']=df.apply(lambda x: 'Hit' if x.correct ==True else 'Miss', axis=1)
    df['Confidence_Accuracy']=df['confidence']+'_'+df['accuracy']    
    df_list.append(df)

df_all=pd.concat(df_list)
df_all=df_all.reset_index()

#get counts of each subject's 4 ratings
rating_counts=df_all.groupby(['ID','rating'])['rating'].count()
#get mean of the 4 ratings across subjects
rating_mean=rating_counts.groupby(['rating']).mean()
#get std of the 4 ratings across subjects
rating_std=rating_counts.groupby(['rating']).std()
#print out the mean and std for each ratings with 4 levels
rating_m=pd.concat([rating_mean,rating_std],axis=1)
rating_m.columns=['mean','std']
print(rating_m)

for id in Ids:  
    acc_rating=df_all[df_all['ID']==id][['correct','rating']].astype(int)
    df_all.loc[df_all['ID']==id,['auroc2']]=auroc2(acc_rating['correct'],acc_rating['rating'],4)

df_counts=df_all.groupby(['ID_conti','Confidence_Accuracy'])['Confidence_Accuracy'].count()
df_counts=df_counts.rename('counts').unstack(level=-1)
epsilon = 0.5
# Adjust counts for H2 and FA2 calculations
df_counts['Adjusted_HC_Hit'] = df_counts['HC_Hit'].fillna(0).add(epsilon)
df_counts['Adjusted_LC_Hit'] = df_counts['LC_Hit'].fillna(0).add(epsilon)
df_counts['Adjusted_HC_Miss'] = df_counts['HC_Miss'].fillna(0).add(epsilon)
df_counts['Adjusted_LC_Miss'] = df_counts['LC_Miss'].fillna(0).add(epsilon)

total_hits = df_counts['Adjusted_HC_Hit'] + df_counts['Adjusted_LC_Hit']
total_misses = df_counts['Adjusted_HC_Miss'] + df_counts['Adjusted_LC_Miss']

# Calculate H2 and FA2 with adjustments
df_counts['H2'] = df_counts['Adjusted_HC_Hit'] / total_hits
df_counts['FA2'] = df_counts['Adjusted_HC_Miss'] / total_misses

# Ensure H2 and FA2 are within the valid range (0,1) for z-score calculation
df_counts['H2'] = df_counts['H2'].clip(epsilon / total_hits, 1 - epsilon / total_hits)
df_counts['FA2'] = df_counts['FA2'].clip(epsilon / total_misses, 1 - epsilon / total_misses)

# Calculate d-prime (dprime2)
df_counts['dprime2'] = df_counts.apply(lambda x: stats.norm.ppf(x['H2']) - stats.norm.ppf(x['FA2']), axis=1)
df_both=df_counts.join(df_all.groupby(['ID_conti'])['auroc2'].mean())
df_both=df_both.reset_index()

fig, ax = plt.subplots(3,1,figsize=(5,8),sharex=True,gridspec_kw={'width_ratios': [1],'height_ratios':[1,1,0.5]})

sns.histplot(
     df_all,
     x="ID_conti", hue="rating",
     multiple="stack",
     discrete=True,
     ax=ax[0] ,palette=sns.color_palette(['#377eb8','#9bbfdc','#ffbf80','#ff7f00'])
     )

sns.histplot(
     df_all,
     x="ID_conti", hue="Confidence_Accuracy",
     multiple="stack",
     discrete=True,
     ax=ax[1] ,palette=sns.color_palette(['#9bbfdc','#377eb8','#ffbf80','#ff7f00']),
     hue_order=['LC_Miss','LC_Hit','HC_Miss','HC_Hit']
     )
sns.move_legend(ax[0], "upper right", bbox_to_anchor=(1.08, 1), ncol=1, title='rating', frameon=True)
sns.move_legend(ax[1], "upper right", bbox_to_anchor=(1.13, 1), ncol=1, title='rating_acc', frameon=True)
plt.show()

sns.lineplot(x='ID_conti', y='dprime2',data=df_both,linewidth = 1,alpha=0.75,marker="o",label='dprime2',ax=ax[2],color='#1f1f1f',linestyle='dashed',markersize=6)
sns.lineplot(x='ID_conti', y='auroc2',data=df_both,linewidth = 1,alpha=0.75,marker="s",label='auroc2',ax=ax[2].twinx(),color='#1f1f1f',linestyle='dashed',markersize=6)

fig.tight_layout()
fig.show()

plt.savefig('figure1b.pdf')





df_list=[]
for subject_id in Ids:
    
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
 
    df=pd.read_csv(fname.replace('_tsss_mc.fif','_events_with_artifacts_marked.csv'))
    
    df=df[(df['task']=='E') & (df['stim']=='N')]
    dfgb=df.groupby(['repeat'])[['correct','RT_memory','rating','RT_rating']]
    dfgb=dfgb.apply(np.sum)/dfgb.count()
    
    dfgb=dfgb.reset_index()
    dfgb['ID']=subject_id
    df_list.append(dfgb)

df_avg=pd.concat(df_list)
df_avg=df_avg.reset_index()

df_list=[]
for subject_id in Ids:
    
    subject = group_name+"%02d" % subject_id
    print("processing subject: %s" % subject)        
    fname=op.join(MEG_data_path,subject,'fname'+'_%02d'%(subject_id)+'_tsss_mc.fif')
 
    df=pd.read_csv(fname.replace('_tsss_mc.fif','_events_with_artifacts_marked.csv'))
    
    df=df[(df['task']=='E') & (df['stim']=='N')]
    dfgb=df.groupby(['rating'])[['correct']]
    dfgb=dfgb.apply(np.sum)/dfgb.count()
    
    dfgb=dfgb.reset_index()
    dfgb['ID']=subject_id
    df_list.append(dfgb)

df_avg2=pd.concat(df_list)
df_avg2=df_avg2.reset_index()

fig, ax = plt.subplots(3,1,figsize=(5,8),gridspec_kw={'width_ratios': [1],'height_ratios':[1,1,1]})

sns.lineplot(x='repeat', y='correct',data=df_avg,ax=ax[1],err_style="bars",color='#377eb8',linewidth = 3,alpha=0.75,marker="o",label='Accuracy')
sns.lineplot(x='repeat', y='rating',data=df_avg,ax=ax[1].twinx(),err_style="bars",color='#ff7f00',linewidth = 3,alpha=0.75,marker="o",label='Confidence Rating')

sns.lineplot(x='repeat', y='RT_memory',data=df_avg,ax=ax[0],err_style="bars",color='#377eb8',linewidth = 3,alpha=0.75,marker="o",label='RT_memory')
sns.lineplot(x='repeat', y='RT_rating',data=df_avg,ax=ax[0].twinx(),err_style="bars",color='#ff7f00',linewidth = 3,alpha=0.75,marker="o",label='RT_rating')

ax[0].set_xticks([1,2,3,4,5,6])
ax[1].set_xticks([1,2,3,4,5,6])

sns.lineplot(x='rating', y='correct',data=df_avg2, hue="ID",ax=ax[2],marker="o",linestyle='dashed',palette = sns.color_palette(['#999999'],n_colors=32),linewidth = 1,alpha=1,legend=False,markersize=5)
sns.lineplot(x='rating', y='correct',data=df_avg2,ax=ax[2],err_style=None,linewidth = 3,alpha=0.75,color='#1f1f1f',marker="o",markersize=10)

ax[2].set_ylabel('Accuracy')
ax[2].set_xlabel('Rating')
ax[2].set_xticks([1,2,3,4])

fig.tight_layout()
fig.show()

plt.savefig('figure1a.pdf')


def find_SD_outliers(data, num_sd):
    """
    Find and return the outlier values in a list of values based on the number of standard deviations (SD).
    
    Parameters:
    data (list or numpy array): The list of values to be checked for outliers.
    num_sd (int): The number of standard deviations to use as a cutoff.
    
    Returns:
    (bool, list): A tuple containing a boolean indicating whether outliers were found and a list of outlier values.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    
    lower_threshold = mean - (num_sd * std_dev)
    upper_threshold = mean + (num_sd * std_dev)
    
    outliers = [value for value in data if value < lower_threshold or value > upper_threshold]
    
    return len(outliers) > 0, outliers

def find_iqr_outliers(data, k=1.5):
    """
    Find and return the outlier values in a list of values using the Interquartile Range (IQR) method.
    
    Parameters:
    data (list or numpy array): The list of values to be checked for outliers.
    k (float): Multiplier for determining the bounds. Default is 1.5, a common choice.
    
    Returns:
    list: A list of outlier values.
    """
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    return list(outliers)



data = df_both['auroc2'].to_numpy()

#Find outliers using the standard deviation method
num_sd_cutoff = 3  # You can change this to 3 for a 3 SD cutoff
contains_outliers_result, outlier_values = find_SD_outliers(data, num_sd_cutoff)
if contains_outliers_result:
    print(f"The list contains outliers (using {num_sd_cutoff} SD cutoff).")
    print(f"Outlier values: {outlier_values}")
else:
    print(f"The list does not contain outliers (using {num_sd_cutoff} SD cutoff).")

# Find outliers using the IQR method
iqr_outliers = find_iqr_outliers(data)
print("IQR Outliers:", iqr_outliers)