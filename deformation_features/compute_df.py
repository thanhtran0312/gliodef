"""
the first deformation feature is the distance between skeleton of the bundle atlas and deformed streamlines. 

one skeleton per each bundle. the skeleton were estimated by RecoBundleX atlas and resampled so that points are spaced 1 mm between each other
streamline - computed per each pair (brats x tracto) per bundle.

if the skeleton has 40 points & the streamline 1 of af_l on that participant has 36 points. 
each point (except for the first and last point) on the streamline has 3d as 3 def features; 

## deformation feature 1 is, for point 1, I compute euclidean distance between that point
## and all the points on the skeleton and take min
"""
import numpy as np
import re
import json
from pathlib import Path
import argparse

import nibabel as nib
from scipy.ndimage import binary_erosion
from scipy.spatial import KDTree

from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamlinespeed import length
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from nibabel.streamlines.array_sequence import ArraySequence
from training_script.utils.utils import load_streamlines, change_to_trk

def spacing_tovox(path, streamlines, target_spacing_mm=1):
    """ for all streamlines from indices given of one tractogram, this function do 2 things
    (1) enforce equal space spacing 
    (2) convert to voxel space 
    
    args: streamlines' coords, spacing value
    """
    resampled_streamlines = []
    for sl in streamlines:
        total_len_mm = length(sl)
        n_points =  int(round(total_len_mm / target_spacing_mm)) + 1
        resampled_streamlines.append(set_number_of_points(sl, n_points))
    new_streamlines = ArraySequence(resampled_streamlines)
    new_sft = StatefulTractogram(streamlines=new_streamlines, reference=change_to_trk(path), space=Space.RASMM)
    new_sft.to_vox() 
    spaced_vox_streamlines = new_sft.streamlines
    return spaced_vox_streamlines

def class_query(sub_tum,kdt_skeleton,label):   
        """class here can be hard/soft neg or pos"""    
        # shape (streams x points x 3), but space should be 2d for kdtree.query()
        path = list(sub_tum.keys())[0]
        spaced_tovox = spacing_tovox(path,sub_tum[path][label], target_spacing_mm=5)
        # list of number of points of each streamline
        n_points_each_stream = [s.shape[0] for s in spaced_tovox]
        flattened_query = np.concatenate(spaced_tovox, axis=0) 
        dist, idx = kdt_skeleton.query(flattened_query, k=1, workers=-1)
        ## split back per streamline
        splits = np.cumsum(n_points_each_stream)[:-1]
        dist_per_streamline = np.split(dist, splits)
        dist_per_streamline_internal_points = [d[1:-1] for d in dist_per_streamline]
        # pair = [{indices[i]: dist_per_streamline_internal_points[i]} for i in range(len(indices))]
        return dist_per_streamline_internal_points


def deformation_feature_1(sub_tum,tum_id,sub_id):
    """
    this function runs for one subject relative to one bundle at a time.

    args: 
            streamlines of the bundle
            centroid tractogram
    output: 

    # step 1: get the skeleton of the bundle - 1mm spaced
    # step 2: for the bundle, load subjects' relevant streamlines - 5mm spaced
    # step 3: for each internal point of each streamline,
              i compute euclidean distance of on that point to all the points on the skeleton
    # step 4: i take the min
    """
    path = f"/nilab-nexus/datasets/GLIODEF/sub-{sub_id}/tractography/sub-{sub_id}_tum-{tum_id}_bundle.csv"

    skeleton_path = f'/home/thanh/gliodef_script/training_script/deformation_features/original_MNIatlas_space_5bdl/centroids/AF_L_centroid.trk'
    skeleton_streamlines = load_streamlines(skeleton_path,container='array')
    bundle_skeleton = spacing_tovox(path=skeleton_path,streamlines=skeleton_streamlines,target_spacing_mm=1) # return Array Sequence len of 1
    search_space = np.array(bundle_skeleton[0])
    kdt_skeleton = KDTree(search_space)

    sub_df1 = []
    """here i query all streamlines together instead of one streamline at a time like the old script"""
        # shape (n_streams, n_points) , value for each cell is "smallest_distance"
        # for each stream, it returns a list of distance for all points of that stream to the closest skeleton point 
    dist_hard_per_streamline = class_query(sub_tum,kdt_skeleton,label='hard_neg_streams')
    dist_soft_per_streamline = class_query(sub_tum,kdt_skeleton,label='soft_neg_streams')
    dist_pos_per_streamline = class_query(sub_tum,kdt_skeleton,label='positive_streams')
    sub_df1 = {
            path: {
                'hard_neg_df1': dist_hard_per_streamline,
                'soft_neg_df1': dist_soft_per_streamline,
                'positive_df1': dist_pos_per_streamline,
            }
        }
    return sub_df1


def get_center_of_mass(path_nii):
    img = nib.load(path_nii)
    mask = img.get_fdata() > 0

    coords = np.argwhere(mask) # gives all voxel coords that belong to the tumor
    cx, cy, cz = coords.mean(axis=0) # take the mean coord across all tumor voxels
    return cx, cy, cz

def get_border_points(path_nii):
    img = nib.load(path_nii)
    mask = img.get_fdata() > 0    # the whole mask - 3d boolean array - 
    # cell value is true/false whether voxel at that index belongs to the lesion or not

    eroded = binary_erosion(mask) # the interior -  3d boolean array
    border = mask & ~eroded       # = the whole mask - the interior - 3d boolean array

    border_points = np.argwhere(border) # n points x 3d coords
    return border_points


def deformation_feature_2():
    """
    output: CT/CX
            C: center of the tumor
            X: one point on the streamline
            CX: ray from tumor center to the point on the the streamline
            T: where CX crosses the tumor
            CT: tumor size on the direction of CX

    algorithm:
            step 1: find where the tumor boundary is in the direction of X, where CX cut the tumor boundary is T
            step 2: compute the ratio

    input:
            tumor center
            tumor border points
            streamline points
    """


   

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]      # root/
    src_dir      = script_dir.parents[1]      # root/src/
    data_dir   = '/nilab-nexus/datasets/GLIODEF'
    output_dir = src_dir / "output"

    parser.add_argument("--bundle", required=True, help="e.g. AF_L")
    args = parser.parse_args() 
    
    with (output_dir / f"bundle_idx_{args.bundle}.json").open("r") as f:
        bundle_idx = json.load(f) 
    subjects = {
                 sub['path']: {
                     "hard_neg_indices": sub['hard_neg_indices'],
                     "soft_neg_indices": sub['soft_neg_indices'],
                     "positive_indices": sub['positive_indices']}
                for sub in bundle_idx[args.bundle]}
    
    subjects_pool = [(bundle_idx[args.bundle ][j]['path']) for j in range(len(bundle_idx[args.bundle ]))]    

    # to get tum_ids & sub_ids
    pattern = re.compile(r"sub-(\d+)_tum-(\d+)")
    sub, tum = zip(*(pattern.search(p).groups() for p in subjects_pool))
    sub, tum = np.array(sub), np.array(tum)

    sub_ids = np.unique(sub)
    tum_ids = np.unique(tum)

    for tum in tum_ids:
        for sub in sub_ids:
            path = f"{data_dir}/sub-{sub}/tractography/sub-{sub}_tum-{tum}_bundle.csv"
            if path in subjects_pool:
                indices = subjects[path]['hard_neg_indices']+subjects[path]['soft_neg_indices']+subjects[path]['positive_indices']
                len_hard = len(subjects[path]['hard_neg_indices'])
                len_soft = len(subjects[path]['soft_neg_indices'])
                streamlines = load_streamlines(change_to_trk(path), idxs = indices, container = 'array')
                sub_tum = {path:
                    {'hard_neg_streams': streamlines[:len_hard],
                    'soft_neg_streams': streamlines[len_hard:len_hard+len_soft],
                    'positive_streams': streamlines[-len_soft:]}
                }
                assert len(streamlines[:len_hard]) + len(streamlines[len_hard:-len_pos]) + len(streamlines[-len_pos:]) == len(subjects[path]['hard_neg_indices']) + len(subjects[path]['soft_neg_indices']) + len(subjects[path]['positive_indices'])
                sub_df1 = deformation_feature_1(sub_tum,tum,sub)
                for each_stream_line:
                    sub_df2 = deformation_feature_2()
                    sub_df3 = deformation_feature_3()

     
