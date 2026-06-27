import json
import os
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
home_path=os.environ["HOME"]
sys.path.append(home_path,'/balance\n dataset')
from balance_40_30_30 import count_streamlines

# load subjects
data_dir = '/nilab-nexus/datasets/GLIODEF'
script_dir = '/home/thanh/gliodef_script'
output_dir = '/home/thanh/output'

with open(os.path.join(output_dir,'bundle_idx.json'),'r') as file:
    bundle_idx = json.load(file)

# positive streamlines per pair sub x tumor for AFL
paths = [bundle_idx['AF_L'][sub]['path'] for sub in range(len(bundle_idx['AF_L']))]
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


## create a dict where key being tum_id and 
# values are another dict with key being path/sub & value being positive streams of that sub
tums = {tum:{} for tum in tum_ids}

pattern = re.compile(r"tum-(\d+)")
for path in paths:
    id = pattern.search(path).group(1) 
    for tum in tum_ids:
        if tum == id:
            tums[tum][path] = positive_streamlines[path]


## bins by severity
average_severity_per_tum = {tum: None for tum in tum_ids}

# severity here is average of positive streams of that tum over all subjects associated with the tum
for tum in tum_ids:
    average_severity_per_tum[tum] = sum(tums[tum].values())/len(tums[tum].values())

n_bins = 5
severity_values = list(average_severity_per_tum.values())
rank_order = np.argsort(severity_values)

severity_bins = np.array_split(rank_order, n_bins)

# cv folds
n_folds = 5

# tum folds
cv_folds_tum = [[] for _ in range(n_folds)]
for bin in severity_bins:
    shuffle_bin = np.random.permutation(bin)
    split_shuffle_bins = np.array_split(shuffle_bin, n_folds)
    for fold_ind in range(n_folds):
        cv_folds_tum[fold_ind].extend([tum_ids[i] for i in split_shuffle_bins[fold_ind]])

# sub folds
shuffled_subs = np.random.permutation(sub_ids)
cv_folds_sub = [list(f) for f in np.array_split(shuffled_subs, n_folds)]

with open(os.path.join(output_dir,"cv_folds_sub.json"),"w") as f:
    json.dump(cv_folds_sub,f,indent=2)

with open(os.path.join(output_dir,"cv_folds_tum.json"),"w") as f:
    json.dump(cv_folds_tum,f,indent=2)
