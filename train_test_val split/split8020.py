import json
import os
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from balance_40_30_30_forAFL import count_streamlines


data_dir = '/nilab-nexus/datasets/GLIODEF'
script_dir = '/home/thanh/gliodef_script'
output_dir = '/home/thanh/output'

# all subjects relevant to this bundle
with open(os.path.join(output_dir,'bundle_idx.json'),'r') as f:
            bundle_idx = json.load(f)
stream_indices_per_sub = {}
for d in bundle_idx['AF_L']:
            stream_indices_per_sub[d['path']] = d
for inner in stream_indices_per_sub.values():
            del inner['path']
paths = list(stream_indices_per_sub.keys())


######## tumor severity
positive_streamlines = {path: None for path in paths}

with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
    futures = {ex.submit(count_streamlines,path,bundle='AF_L'): path for path in paths}

    for fut in as_completed(futures):
        path = futures[fut]
        positive, *_ = fut.result() 
        positive_streamlines[path] = int(positive['AF_L'])

# get tum_ids & sub_ids of warped subjects who passed the threshold
pattern = re.compile(r"sub-(\d+)_tum-(\d+)")

# *: 0 match or more
sub, tum = zip(*(pattern.search(p).groups() for p in paths))
sub, tum = np.array(sub), np.array(tum)

sub_ids = np.unique(sub)
tum_ids = np.unique(tum)


## create a dict wher"""  """ """e key being tum_id and 
# values are another dict with key being path/sub & value being positive streams of that sub
tums = {tum:{} for tum in tum_ids}

pattern = re.compile(r"tum-(\d+)")
for path in paths:
    id = pattern.search(path).group(1) 
    for tum in tum_ids:
        if tum == id:
            tums[tum][path] = positive_streamlines[path]

average_severity_per_tum = {tum: None for tum in tum_ids}

for tum in tum_ids:
    average_severity_per_tum[tum] = sum(tums[tum].values())/len(tums[tum].values())

n_bins = 5
severity_values = list(average_severity_per_tum.values())
rank_order = np.argsort(severity_values)
severity_bins = np.array_split(rank_order, n_bins)


########### split
split_config = {
    'train' : 0.8,
    'test'  : 0.2
}
np.random.seed(2)
# tum
cv_folds_tum = {'train':[],'test':[]}
for bin in severity_bins:
    shuffle_bin = np.random.permutation(bin)
    number_train_samples = round(split_config['train']*len(shuffle_bin))
    cv_folds_tum['train'].extend(tum_ids[shuffle_bin[:number_train_samples]])
    cv_folds_tum['test'].extend(tum_ids[shuffle_bin[number_train_samples:]])

assert set(cv_folds_tum['train']).isdisjoint(cv_folds_tum['test'])

# sub
cv_folds_sub = {'train':[],'test':[]}
shuffled_subs = np.random.permutation(sub_ids)
split = int(split_config['train']*len(shuffled_subs))
cv_folds_sub['train'], cv_folds_sub['test'] = np.split(shuffled_subs, [split])

assert set(cv_folds_tum['train']).isdisjoint(cv_folds_tum['test'])

# or
# from sklearn.model_selection import train_test_split
# train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

with open(os.path.join(output_dir,"cv_folds_sub_8020.json"),"w") as f:
    json.dump(cv_folds_sub,f,indent=2)

with open(os.path.join(output_dir,"cv_folds_tum_8020.json"),"w") as f:
    json.dump(cv_folds_tum,f,indent=2)