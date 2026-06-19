import json
import os
import pandas as pd
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# input: bundle_idx where its a dict with keys() being bundles
# items() being path to subjects whose have positive streamlines more than 100 for the bundle

# (1) count how many positive streamlines there are in each bundle of each subject
# (2) compute the ratio of positives vs total for that subject for that bundle

# (3) if it's 40% ok
# (4) if it's not 40% -> rescale
# (5) does each subject's 40-30-30 pass 1500?
# what do i want for the output?
# each subject has a file with indices of 40-30-30

# (6) output: json file for 3 lists of indices into warped tractogram for positive/negative streams

rootdir = '/nilab-nexus/datasets/GLIODEF'
with open(os.path.join('/home/thanh/gliodef_script','bundle_idx.json'),'r') as file:
    data = json.load(file)

balance_config = {
    'positive': 0.4,
    'hard_negative': 0.3,
    'soft_negative': 0.3
}

# (1) for each bundle, each subject, i calculate % of positive streamlines
bundles = list(data.keys())

def count_streamlines(path, bundles):

    df = pd.read_csv(path,usecols=bundles)
    positive_streamlines_all_bundles = {}
    negative_streamlines_all_bundles = {}

    for b in bundles:
        if path in data[b]:
            positive_streamlines = df[b].sum() # a scalar
            negative_streamlines = len(df[b]) - df[b].sum() # a scalar
            positive_streamlines_all_bundles[b] = positive_streamlines
            negative_streamlines_all_bundles[b] = negative_streamlines
            positive_indices = list(np.where(df[b]==True))[0]
            negative_indices = list(np.where(df[b]==False))[0]
    return positive_streamlines_all_bundles, negative_streamlines_all_bundles, positive_indices, negative_indices

def compute_ratio(count_streamlines,path,bundles):
    positive_stream,negative_stream, positive_idx, negative_idx = count_streamlines(path,bundles)
    bundles_sub = list(positive_stream.keys())
    positive_ratio = {}
    negative_ratio = {}
    for bundle in bundles_sub:
        total = positive_stream[bundle] + negative_stream[bundle]
        positive_ratio[bundle] = positive_stream[bundle] / total
        negative_ratio[bundle] = negative_stream[bundle] / total
    return positive_stream, negative_stream, positive_ratio, negative_ratio, positive_idx, negative_idx


def sample_streams(pos_idx,positive_should_be):
    positive = np.random.choice(pos_idx, size = positive_should_be, replace=False)
    negative = 

if __name__ == "__main__":
    subjects = np.unique([path for paths in data.values() for path in paths])
    bundle_idx = {b: [] for b in bundles}

    with ProcessPoolExecutor(max_workers=os.cpu.count()) as ex:
        futures = {ex.submit(compute_ratio,count_streamlines,path,bundles): path for path in subjects}

        for fut in as_completed(futures):
            positive_stream, negative_stream, positive_ratio, negative_ratio, positive_idx, negative_idx = fut.result()
            balance_set = {}
            for bundle in list(positive_stream.keys()):
                if positive_ratio[bundle] < balance_config['positive']:
                    assert negative_stream[bundle] >= round((balance_config['hard_negative']+balance_config['soft_negative']) *negative_stream[bundle])
                    total_should_be = int(round(positive_stream[bundle]/balance_config['positive']))
                    positive_should_be = positive_stream[bundle]
                    negative_should_be = total_should_be - positive_should_be
                elif negative_ratio[bundle] <= (balance_config['hard_negative'] + balance_config['soft_negative']):
                    assert positive_stream >= round(balance_config['positive'] *positive_stream[bundle])
                    total_should_be = int(round(negative_stream[bundle]/(balance_config['hard_negative'] + balance_config['soft_negative'])))
                    negative_should_be = negative_stream
                    positive_should_be = total_should_be - negative_should_be

                if total_should_be < 1500:
                    continue
                balance_set[bundle] = [total_should_be,positive_should_be,negative_should_be]


                positive = random(positive_idx, positive_should_be)


