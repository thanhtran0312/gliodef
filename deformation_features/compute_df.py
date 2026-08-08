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
from scipy.spatial import KDTree

from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamlinespeed import length
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from nibabel.streamlines.array_sequence import ArraySequence
from dipy.io.streamline import load_tractogram
from training_script.utils.utils import load_streamlines, change_path

def spacing_tovox(path,idx=None,target_spacing_mm=1):
    """ for all streamlines from indices given of one tractogram, this function do 2 things
    (1) enforce equal space spacing 
    (2) convert to voxel space 
    
    args: path in trk, spacing value
    """
    streamlines = load_streamlines(path, idx, container = 'array')
    resampled_streamlines = []
    for sl in streamlines:
        total_len_mm = length(sl)
        n_points =  int(round(total_len_mm / target_spacing_mm)) + 1
        resampled_streamlines.append(set_number_of_points(sl, n_points))
    new_streamlines = ArraySequence(resampled_streamlines)
    new_sft = StatefulTractogram(streamlines=new_streamlines, reference=path, space=Space.RASMM)
    new_sft.to_vox() 
    spaced_vox_streamlines = new_sft.streamlines
    return spaced_vox_streamlines

def class_query(sub,kdt_skeleton,label):   
        """class here can be hard/soft neg or pos"""    
        # shape (streams x points x 3), but space should be 2d for kdtree.query()
        label_query = spacing_tovox(change_path(next(iter(sub.keys()))), idx = next(iter(sub.values()))[label], target_spacing_mm=5)
        # list of number of points of each strealine
        length = [s.shape[0] for s in label_query]
        flattened_query = np.concatenate(label_query, axis=0) 
        dist, idx = kdt_skeleton.query(flattened_query, k=1, workers=-1)
        ## split back per streamline
        splits = np.cumsum(length)[:-1]
        dist_per_streamline = np.split(dist, splits)
        return dist_per_streamline

def deformation_feature_1(bundle_idx, bundle_centroid_tractogram_path):
    """
    this function runs for one bundle at a time.

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
    bundle_skeleton = spacing_tovox(bundle_centroid_tractogram_path,target_spacing_mm=1) # return Array Sequence len of 1
    search_space = np.array(bundle_skeleton[0])
    kdt_skeleton = KDTree(search_space)
    bundle = next(iter(bundle_idx.keys()))
    subjects = {
        sub['path']: {
            "hard_neg_indices": sub['hard_neg_indices'],
            "soft_neg_indices": sub['soft_neg_indices'],
            "positive_indices": sub['soft_neg_indices']}
            for sub in bundle_idx[bundle]
        }
    allsub_df1 = []
    for sub in subjects:
        """here i query all streamlines together instead of one streamline at a time like the old script"""
        # shape (n_streams, n_points) , value for each cell is "smallest_distance"
        # for each stream, it returns a list of distance for all points of that stream to the closest skeleton point 
        dist_hard_per_streamline = class_query(sub,kdt_skeleton,label='hard_neg_indices')
        dist_soft_per_streamline = class_query(sub,kdt_skeleton,label='soft_neg_indices')
        dist_pos_per_streamline = class_query(sub,kdt_skeleton,label='positive_indices')
        subject = {
            sub['path']: {
                'hard_neg_df1': dist_hard_per_streamline,
                'soft_neg_df1': dist_soft_per_streamline,
                'positive_df1': dist_pos_per_streamline,
            }
        }
        allsub_df1.append(sub)
    return allsub_df1


