import json
import os
import pandas as pd
import numpy as np
import random
from scipy.spatial import KDTree
from utils import load_streamlines, resample_streamlines, change_path
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
# (6) sample hard vs soft negatives
    # sample every line into 40 points
    # then for one bundle, we have eg 500 pos & 2000 neg
    # for every negative streamline, we want to know the closest bundle

# (7) output: json file for 3 lists of indices into warped tractogram for positive/negative streams

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
    """args
    inputs: 
        bundles
        paths of subjects that are relevant to each bundle

    outputs:
        how many negative, positive streamlines that belong to relevant bundles of each subject 
        indices of those streamlines
    """

    df = pd.read_csv(path,usecols=bundles)
    positive_streamlines_all_bundles = {}
    negative_streamlines_all_bundles = {}
    positive_indices_pool = {}
    negative_indices_pool = {}

    for b in bundles:
        if path in data[b]:
            positive_streamlines = df[b].sum() # a scalar
            negative_streamlines = len(df[b]) - df[b].sum() # a scalar
            positive_streamlines_all_bundles[b] = positive_streamlines
            negative_streamlines_all_bundles[b] = negative_streamlines
            positive_indices_pool[b] = np.where(df[b]==True)[0]
            negative_indices_pool[b] = np.where(df[b]==False)[0]
    return positive_streamlines_all_bundles, negative_streamlines_all_bundles, positive_indices_pool, negative_indices_pool

def compute_ratio(count_streamlines,path,bundles):
    """args
    inputs: count_streamlines()
    outputs: count_streamlines() outputs + ratio of positive/negative vs total
    """
    positive_stream,negative_stream, positive_indices_pool, negative_indices_pool = count_streamlines(path,bundles)
    bundles_sub = list(positive_stream.keys())
    positive_ratio = {}
    negative_ratio = {}
    for bundle in bundles_sub:
        total = positive_stream[bundle] + negative_stream[bundle]
        positive_ratio[bundle] = positive_stream[bundle] / total
        negative_ratio[bundle] = negative_stream[bundle] / total
    return positive_stream, negative_stream, positive_ratio, negative_ratio, positive_indices_pool, negative_indices_pool

# should i sample positive and negative points separately
# should i return indices or point coordinates
def balance_40_30_30(positive_stream, negative_stream, positive_ratio, negative_ratio):
    """
    args
    inputs: 
    output: {'bundle': total, positive, negative} after balancing 
    """
    balance_set = {}
    for bundle in list(positive_stream.keys()):
        if positive_ratio[bundle] < balance_config['positive']:
            assert negative_stream[bundle] >= round((balance_config['hard_negative']+balance_config['soft_negative']) *negative_stream[bundle])
            total_should_be = int(round(positive_stream[bundle]/balance_config['positive']))
            positive_should_be = positive_stream[bundle]
            negative_should_be = total_should_be - positive_should_be
        elif negative_ratio[bundle] <= (balance_config['hard_negative'] + balance_config['soft_negative']):
            assert positive_stream[bundle] >= round(balance_config['positive'] *positive_stream[bundle])
            total_should_be = int(round(negative_stream[bundle]/(balance_config['hard_negative'] + balance_config['soft_negative'])))
            negative_should_be = negative_stream[bundle]
            positive_should_be = total_should_be - negative_should_be

        if total_should_be < 1500:
            continue
        balance_set[bundle] = [total_should_be,positive_should_be,negative_should_be]
    return balance_set

# load_streamline() return an outer tuple shape (m,) for m streamlines
# each inner tuple has shape (n,3) which is the 3d coords for all points on a streamline.

def sample_hard_negative(path,positive_indices_pool, negative_indices_pool,balance_set,n_pts):
    hard_neg_indices = {}
    path_coords = change_path(path)
    for b in list(negative_indices_pool.keys()):
        n__neg_streamlines = round(balance_set[b][2]/2) # should be for the balance

        # query space: negative streams coords both directions of a sequence
        negative = load_streamlines(path_coords,idxs=negative_indices_pool[b], container='array')
        n_neg_streams_total = negative.shape[0]
        negative_resampled = resample_streamlines(negative,n_pts)
        Q_40 = np.array(negative_resampled)
        Q_40_2d = Q_40.reshape(n_neg_streams_total,-1) # flatten (n_streams,40,3) to (n_streams,120)
        flipped_stream = Q_40[:,::-1,:].reshape(n_neg_streams_total,-1)
        Q_40_2d_flip_invariant = np.concatenate((Q_40_2d,flipped_stream),axis=0)

        # search space: positive streams coords
        positive = load_streamlines(path_coords,idxs=positive_indices_pool[b], container='array')
        n_pos_stream_total = positive.shape[0]
        positive_resampled = resample_streamlines(positive,n_pts)
        S_40 = np.array(positive_resampled)
        S_40_2d = S_40.reshape(n_pos_stream_total,-1) # flatten into 2d

        # build the tree
        # KDTree expects 2d array shape(n_fibers,n_d), hence the reshape above
        T_kdt = KDTree(S_40_2d)

        # run query with query space on the tree that built on search space
        # return two vectors
        # one is distance of the query to its nearest match in the search space
        # one is the index of the match 
        distance,indices = T_kdt.query(Q_40_2d_flip_invariant)
        neg_dist = distance[:n_neg_streams_total]
        neg_dist_flipped = distance[n_neg_streams_total:]
        real_distance = np.zeros(n_neg_streams_total)
        real_indices = np.zeros(n_neg_streams_total)
        for i in range(n_neg_streams_total):
            if neg_dist[i] <= neg_dist_flipped[i]:
                real_distance[i] = neg_dist[i] 
                real_indices[i] = indices[:n_neg_streams_total][i]
            elif neg_dist[i] > neg_dist_flipped[i]:
                real_distance[i] = neg_dist_flipped[i]
                real_indices[i] = indices[n_neg_streams_total:][i]
        idx_neg_nn_bdl = np.argsort(real_distance)[:n__neg_streamlines]
        idx_neg_nn_bdl_real = negative_indices_pool[b][idx_neg_nn_bdl]
        hard_neg_indices[b] = idx_neg_nn_bdl_real
    return hard_neg_indices


def sample_soft_negative(hard_neg_indices,negative_indices_pool, balance_set):
    soft_neg_indices = {}
    for b in list(negative_indices_pool.keys()):
        n__neg_streamlines = balance_set[b][2]/2
        soft_neg_pool = np.setdiff1d(negative_indices_pool[b], hard_neg_indices[b])
        soft_neg_indices[b] = np.random.choice(soft_neg_pool, size = round(n__neg_streamlines), replace=False)
    return soft_neg_indices


def sample_pos_streams(positive_indices_pool,balance_set):
    """args: we have indices of positive streams, so now in all those indices,
    we sample n (positive_should_be) streams
    """
    positive_indices = {}
    for b in list(positive_indices_pool.keys()):
        n__pos_streamlines = balance_set[b][1]/2
        positive_indices[b] = np.random.choice(positive_indices_pool[b], size = round(n__pos_streamlines), replace=False)
    return positive_indices

def process(path):
    positive_stream, negative_stream, positive_ratio, negative_ratio, positive_indices_pool, negative_indices_pool = compute_ratio(count_streamlines,path,bundles)
    balance_set = balance_40_30_30(positive_stream, negative_stream, positive_ratio, negative_ratio)
    hard_neg_indices = sample_hard_negative(path,positive_indices_pool, negative_indices_pool,balance_set,n_pts=40)
    soft_neg_indices = sample_soft_negative(hard_neg_indices,negative_indices_pool, balance_set)
    positive_indices = sample_pos_streams(positive_indices_pool,balance_set)
    return hard_neg_indices,soft_neg_indices,positive_indices,path

if __name__ == "__main__":
    subjects = np.unique([path for paths in data.values() for path in paths])
    bundle_idx = {b: [] for b in bundles}

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = {ex.submit(process,path): path for path in subjects}

        for fut in as_completed(futures):
            hard_neg_indices,soft_neg_indices,positive_indices,path = fut.result()
            relevant_bundles = list(hard_neg_indices.keys())
            for b in relevant_bundles:
                bundle_idx[b].append({
                            'path': path,
                            'hard_neg_indices': hard_neg_indices[b].tolist(),
                            'soft_neg_indices': soft_neg_indices[b].tolist(),
                            'positive_indices': positive_indices[b].tolist()
                        })            

    with open("bundle_idx.json", "w") as f:
        json.dump(bundle_idx,f,indent=2)

