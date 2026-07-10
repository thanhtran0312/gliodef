import json
import os
import pandas as pd
import numpy as np
import random
import time
import faiss
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
data_dir = '/nilab-nexus/datasets/GLIODEF'
script_dir = '/home/thanh/gliodef_script'
output_dir = '/home/thanh/output'

with open(os.path.join(output_dir,'subjects_survived_100str_thresh.json'),'r') as file:
    data = json.load(file)

balance_config = {
    'positive': 0.4,
    'hard_negative': 0.3,
    'soft_negative': 0.3
}

# (1) for each bundle, each subject, i calculate % of positive streamlines
bundle = list(data.keys())[4]

def count_streamlines(path, bundle):
    """args
    inputs: 
        bundles
        paths of subjects that are relevant to each bundle

    outputs:
        how many negative, positive streamlines that belong to relevant bundles of each subject 
        indices of those streamlines
    """
    print('started counting')
    df = pd.read_csv(path,usecols=[bundle])
    positive_streamlines_all_bundles = {}
    negative_streamlines_all_bundles = {}
    positive_indices_pool = {}
    negative_indices_pool = {}

    if path in data[bundle]:
            positive_streamlines = df[bundle].sum() # a scalar
            negative_streamlines = len(df[bundle]) - df[bundle].sum() # a scalar
            positive_streamlines_all_bundles[bundle] = positive_streamlines
            negative_streamlines_all_bundles[bundle] = negative_streamlines
            positive_indices_pool[bundle] = np.where(df[bundle]==True)[0]
            negative_indices_pool[bundle] = np.where(df[bundle]==False)[0]
    print('counted')

    return positive_streamlines_all_bundles, negative_streamlines_all_bundles, positive_indices_pool, negative_indices_pool

def compute_ratio(count_streamlines,path,bundle):
    """args
    inputs: count_streamlines()
    outputs: count_streamlines() outputs + ratio of positive/negative vs total
    """
    print('started computing')
    positive_stream,negative_stream, positive_indices_pool, negative_indices_pool = count_streamlines(path,bundle)
    positive_ratio = {}
    negative_ratio = {}
    total = positive_stream[bundle] + negative_stream[bundle]
    positive_ratio[bundle] = positive_stream[bundle] / total
    negative_ratio[bundle] = negative_stream[bundle] / total
    print('computed')
    return positive_stream, negative_stream, positive_ratio, negative_ratio, positive_indices_pool, negative_indices_pool

# should i sample positive and negative points separately
# should i return indices or point coordinates
def balance_40_30_30(positive_stream, negative_stream, positive_ratio, negative_ratio):
    """
    args
    inputs: 
    output: {'bundle': total, positive, negative} after balancing 
    """
    print('start balancing')
    balance_set = {}
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
    
    balance_set[bundle] = [total_should_be,positive_should_be,negative_should_be]
    print('balanced')
    return balance_set

# load_streamline() return an outer tuple shape (m,) for m streamlines
# each inner tuple has shape (n,3) which is the 3d coords for all points on a streamline.

def faiss_nn(S, Q):
    S_f = np.ascontiguousarray(S, dtype='float32')
    Q_f = np.ascontiguousarray(Q, dtype='float32')
    d = S_f.shape[1]
    
    # GPU
    res = faiss.StandardGpuResources()
    index_cpu = faiss.IndexFlatL2(d)
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    
    index.add(S_f)
    distances, indices = index.search(Q_f, k=1)
    return distances.squeeze(), indices.squeeze()

def sample_hard_negative(path,positive_indices_pool, negative_indices_pool,balance_set,n_pts):
    print('start sampling hard neg')
    hard_neg_indices = {}
    path_coords = change_path(path)
    n__neg_streamlines = round(balance_set[bundle][2]/2) # should be for the balance


    # load tractogram
    n_pos = len(positive_indices_pool[bundle])
    n_neg = len(negative_indices_pool[bundle])
    combined_idx = np.concatenate([positive_indices_pool[bundle], negative_indices_pool[bundle]])
    t = time.time(); 

    combined = load_streamlines(path_coords, idxs=combined_idx, container='array')
    print(f"load streams: {time.time()-t:.2f}s")
    positive = combined[:n_pos]
    negative = combined[n_pos:]
        # query space: negative streams coords both directions of a sequence
    n_neg_streams_total = negative.shape[0]
    t = time.time(); 

    negative_resampled = resample_streamlines(negative,n_pts)
    print(f"resample negative: {time.time()-t:.2f}s")
    Q_40 = np.array(negative_resampled)
    Q_40_2d = Q_40.reshape(n_neg_streams_total,-1) # flatten (n_streams,40,3) to (n_streams,120)
    flipped_stream = Q_40[:,::-1,:].reshape(n_neg_streams_total,-1)
    Q_40_2d_flip_invariant = np.concatenate((Q_40_2d,flipped_stream),axis=0)

    print('build search space')
        # search space: positive streams coords
    n_pos_stream_total = positive.shape[0]
    positive_resampled = resample_streamlines(positive,n_pts)
    S_40 = np.array(positive_resampled)
    S_40_2d = S_40.reshape(n_pos_stream_total,-1) # flatten into 2d
    print('search')
        # run query with query space on the tree that built on search space
        # return two vectors
        # one is distance of the query to its nearest match in the search space
        # one is the index of the match 
    t = time.time(); 
    distance, indices = faiss_nn(S_40_2d,Q_40_2d_flip_invariant)

    print(f"query: {time.time()-t:.2f}s")
    neg_dist = distance[:n_neg_streams_total]
    neg_dist_flipped = distance[n_neg_streams_total:]
    flip_is_smaller = neg_dist_flipped < neg_dist
    real_distance = np.where(flip_is_smaller, neg_dist_flipped, neg_dist)
    real_indices  = np.where(flip_is_smaller, indices[n_neg_streams_total:], indices[:n_neg_streams_total])
    idx_neg_nn_bdl = np.argsort(real_distance)[:n__neg_streamlines]
    idx_neg_nn_bdl_real = negative_indices_pool[bundle][idx_neg_nn_bdl]
    hard_neg_indices[bundle] = idx_neg_nn_bdl_real
    print('finish searching')
    return hard_neg_indices


def sample_soft_negative(hard_neg_indices,negative_indices_pool, balance_set):
    soft_neg_indices = {}
    n__neg_streamlines = balance_set[bundle][2]/2
    soft_neg_pool = np.setdiff1d(negative_indices_pool[bundle], hard_neg_indices[bundle])
    soft_neg_indices[bundle] = np.random.choice(soft_neg_pool, size = round(n__neg_streamlines), replace=False)
    return soft_neg_indices


def sample_pos_streams(positive_indices_pool,balance_set):
    """args: we have indices of positive streams, so now in all those indices,
    we sample n (positive_should_be) streams
    """
    positive_indices = {}
    n__pos_streamlines = balance_set[bundle][1]
    positive_indices[bundle] = np.random.choice(positive_indices_pool[bundle], size = round(n__pos_streamlines), replace=False)
    return positive_indices

def process(path):
    t0 = time.time()
    positive_stream, negative_stream, positive_ratio, negative_ratio, positive_indices_pool, negative_indices_pool = compute_ratio(count_streamlines,path,bundle)
    balance_set = balance_40_30_30(positive_stream, negative_stream, positive_ratio, negative_ratio)
    if balance_set[bundle][0] > 1500:
        hard_neg_indices = sample_hard_negative(path,positive_indices_pool, negative_indices_pool,balance_set,n_pts=40)
        soft_neg_indices = sample_soft_negative(hard_neg_indices,negative_indices_pool, balance_set)
        positive_indices = sample_pos_streams(positive_indices_pool,balance_set)
        return hard_neg_indices,soft_neg_indices,positive_indices,path
    print(f"query: {time.time()-t0:.2f}s")
    return None,None,None,path

if __name__ == "__main__":
    subjects = np.unique([path for path in data['AF_L']])
    bundle_idx = {'AF_L':[]}

    with ProcessPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(process,path): path for path in subjects}

        for fut in as_completed(futures):
            hard_neg_indices,soft_neg_indices,positive_indices,path = fut.result()
            if hard_neg_indices is None:
                print(f"Skipped {path} (under 1500 threshold)")
                continue

            bundle_idx[bundle].append({
                            'path': path,
                            'hard_neg_indices': hard_neg_indices[bundle].tolist(),
                            'soft_neg_indices': soft_neg_indices[bundle].tolist(),
                            'positive_indices': positive_indices[bundle].tolist()
                        })            
            print(f"Processed {path}")

    with open("bundle_idx.json", "w") as f:
        json.dump(bundle_idx,f,indent=2)

