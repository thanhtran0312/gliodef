"""
to run on the cluster, each job is a combination of sub_id x tum_id x bundle

"""

import numpy as np
import re
import json
import pickle

from pathlib import Path
import argparse
import sklearn
import nibabel as nib
from scipy.ndimage import binary_erosion
from scipy.spatial import KDTree

from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamlinespeed import length
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from nibabel.streamlines.array_sequence import ArraySequence
from training_script.utils.utils import load_streamlines, change_to_trk, irbio_path


def ray_intersections_all_streamlines(
    streamlines,
    tumor_center_mm,
    lesion_img,
    step_mm=0.1
):
    """
    Find tumor-surface intersection T for every point of every streamline.

    Parameters
    ----------
    streamlines : iterable
        Each streamline has shape (n_points, 3), in RASMM / MNI mm.
    tumor_center_mm : array-like, shape (3,)
        Tumor center in RASMM / MNI mm.
    lesion_img : nibabel Nifti1Image
        Binary lesion mask.
    step_mm : float
        Step size along each C -> P ray, in mm.

    Returns
    -------
    all_tumor_points : list of arrays
        One array per streamline, shape (n_points, 3).
        Each row is the tumor-surface point corresponding to that
        streamline point, in RASMM / MNI mm.
    """

    mask = lesion_img.get_fdata() > 0
    inv_affine = np.linalg.inv(lesion_img.affine)
    C = np.asarray(tumor_center_mm, dtype=float)
    all_tumor_points = []

    for stream in streamlines:
        stream = np.asarray(stream, dtype=float)

        # C -> P direction for every point on this streamline
        directions = stream - C                         # (n_points, 3)
        distances = np.linalg.norm(directions, axis=1) # (n_points,)

        # unit directions
        unit_dirs = directions / distances[:, None]

        # initially all rays start at tumor center
        T = np.repeat(C[None, :], len(stream), axis=0)

        # rays still being followed
        active = np.ones(len(stream), dtype=bool)

        t = 0.0

        while np.any(active):
            t += step_mm

            # points at distance t along every active ray
            points_mm = C + t * unit_dirs[active]

            # RASMM -> lesion voxel coordinates
            points_vox = nib.affines.apply_affine(inv_affine, points_mm)

            idx = np.round(points_vox).astype(int)

            # check whether voxel index is inside image
            inside_bounds = np.all((idx >= 0) & (idx < np.array(mask.shape)), axis=1)
            inside_tumor = np.zeros(len(idx), dtype=bool)
            valid_idx = idx[inside_bounds]
            inside_tumor[inside_bounds] = mask[
                valid_idx[:, 0],
                valid_idx[:, 1],
                valid_idx[:, 2]
            ]

            active_indices = np.where(active)[0]
            # rays that are still inside tumor:
            # save current position as latest valid surface candidate
            still_inside = active_indices[inside_tumor]
            T[still_inside] = points_mm[inside_tumor]
            # rays that have exited tumor are finished
            exited = active_indices[~inside_tumor]
            active[exited] = False
        all_tumor_points.append(T)
    return all_tumor_points

def spacing(path, streamlines, target_spacing_mm=1):
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
    new_sft = StatefulTractogram(streamlines=new_streamlines, reference=path, space=Space.RASMM)
    spaced_streamlines = new_sft.streamlines
    return spaced_streamlines

def class_query(all_streamlines,kdt_skeleton):   
        """class here can be hard/soft neg or pos"""    
        # shape (streams x points x 3), but space should be 2d for kdtree.query()
        # list of number of points of each streamline
        n_points_each_stream = [s.shape[0] for s in all_streamlines]
        flattened_query = np.concatenate(all_streamlines, axis=0) 
        dist, idx = kdt_skeleton.query(flattened_query, k=1, workers=-1)
        ## split back per streamline
        splits = np.cumsum(n_points_each_stream)[:-1]
        dist_per_streamline = np.split(dist, splits)
        dist_per_streamline_internal_points = [d[1:-1] for d in dist_per_streamline]
        # pair = [{indices[i]: dist_per_streamline_internal_points[i]} for i in range(len(indices))]
        return dist_per_streamline_internal_points

def get_center_of_mass(path_nii):
    img = nib.load(path_nii)
    mask = img.get_fdata() > 0

    coords = np.argwhere(mask) # gives all voxel coords that belong to the tumor
    center_vox = coords.mean(axis=0) # take the mean coord across all tumor voxels

    center_rasmm = nib.affines.apply_affine(img.affine, center_vox)
    return center_rasmm

def get_border_points(path_nii):
    img = nib.load(path_nii)
    mask = img.get_fdata() > 0    # the whole mask - 3d boolean array - 
    # cell value is true/false whether voxel at that index belongs to the lesion or not

    eroded = binary_erosion(mask) # the interior -  3d boolean array
    border = mask & ~eroded       # = the whole mask - the interior - 3d boolean array

    border_points = np.argwhere(border) # n points x 3d coords
    border_rasmm = nib.affines.apply_affine(img.affine, border_points)
    return border_rasmm

def deformation_feature_1(bundle,all_streamlines):
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

    skeleton_path = (
        PC_DIR
        / "training_script"
        / "deformation_features"
        / "original_MNIatlas_space_5bdl"
        / "centroids"
        / f"{bundle}_centroid.trk")
    skeleton_streamlines = load_streamlines(skeleton_path,container='array')
    bundle_skeleton = spacing(path=skeleton_path,streamlines=skeleton_streamlines,target_spacing_mm=1) # return Array Sequence len of 1
    search_space = np.array(bundle_skeleton[0])
    kdt_skeleton = KDTree(search_space)
    """here i query all streamlines together instead of one streamline at a time like the old script"""
        # shape (n_streams, n_points) , value for each cell is "smallest_distance"
        # for each stream, it returns a list of distance for all points of that stream to the closest skeleton point 
    dist_per_streamline = class_query(all_streamlines,kdt_skeleton)
    return dist_per_streamline

def deformation_feature_2(all_streamlines, tumor_center, all_streams_tumor_points):
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
    all_streams = []
    for i,stream in enumerate(all_streamlines):
        # # cosine between streamline point i direction and border point j direction 
        # # so row is all tumor points for one point on the sreamline (n_stream_points, n_border_points)
        # cos_sim = sklearn.metrics.pairwise.cosine_similarity(stream,tumor_border_points)
        # # tumor points for all points on this stream, shape = n_points on the stream
        # tumor_points = tumor_border_points[np.argmax(cos_sim,axis=1)]
        tumor_points = all_streams_tumor_points[i]
        tumor_size = np.linalg.norm(tumor_center - tumor_points, axis=1)
        tumor_distance = np.linalg.norm(tumor_center - stream, axis=1)
        ratio = tumor_size / tumor_distance
        all_streams.append(ratio[1:-1])
    
    return all_streams

def deformation_feature_3(all_streamlines,all_streams_tumor_points):
    all_streams = []
    for i,stream in enumerate(all_streamlines):
        P = stream[:-2]      # j-1
        Q = stream[1:-1]     # j
        S = stream[2:]       # j+1        
        Pt = all_streams_tumor_points[i][:-2]
        Qt = all_streams_tumor_points[i][1:-1]     
        St = all_streams_tumor_points[i][2:]

        # alpha
        PQ = P - Q
        SQ = S - Q
        cos_alpha = np.diagonal(sklearn.metrics.pairwise.cosine_similarity(PQ,SQ))
        alpha = np.arccos(np.clip(cos_alpha, -1, 1))

        PtQt = Pt - Qt
        StQt = St - Qt
        cos_beta = np.diagonal(sklearn.metrics.pairwise.cosine_similarity(PtQt,StQt))
        beta = np.arccos(np.clip(cos_beta, -1, 1))
        df3 = alpha - beta
        all_streams.append(df3)
    return all_streams

     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="e.g. AF_L")
    parser.add_argument("--sub", required=True)
    parser.add_argument("--tum", required=True)
    args = parser.parse_args()
    bundle, sub, tum = args.bundle, args.sub, args.tum

    MNT_DIR = Path("/home/thuythienthanh.tran/mnt")

    # PC dir, mounted with SSHFS
    PC_DIR = MNT_DIR / "pc"

    # nilab-nexus, mounted with SSHFS
    NILAB_DIR = MNT_DIR / "mount_point"

    # Large GLIODEF dataset lives on nilab-nexus
    data_dir = NILAB_DIR / "datasets" / "GLIODEF"

    # Bundle index JSONs are on PC
    output_dir = PC_DIR / "output"

    # Generated feature files should go to pc
    feature_output_dir = MNT_DIR / "gliodef_backup"

    with (output_dir / f"bundle_idx_{bundle}.json").open("r") as f:
        bundle_idx = json.load(f)
    bundle_idx = irbio_path(bundle_idx, bundle)

    subjects = {
        s['path']: {
            "hard_neg_indices": s['hard_neg_indices'],
            "soft_neg_indices": s['soft_neg_indices'],
            "positive_indices": s['positive_indices']}
        for s in bundle_idx[bundle]}

    path = str(
        data_dir
        / f"sub-{sub}"
        / "tractography"
        / f"sub-{sub}_tum-{tum}_bundle.csv"
    )
    out_dir = feature_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sub-{sub}_tum-{tum}_{bundle}.pkl"

    if out_path.exists():
        print(f"Skipped, already computed: {out_path}")
        exit()
    if path not in subjects:
        print(f"{path} not in bundle {bundle} pool, skipping.")
        exit()

    path_nii = data_dir / "sub-MNI" / "lesion" / f"sub-MNI_tum-{tum}_lesion.nii.gz"
    lesion_img = nib.load(path_nii)
    tumor_center = get_center_of_mass(path_nii)

    indices = subjects[path]['hard_neg_indices'] + subjects[path]['soft_neg_indices'] + subjects[path]['positive_indices']
    len_hard = len(subjects[path]['hard_neg_indices'])
    len_soft = len(subjects[path]['soft_neg_indices'])
    len_pos = len(subjects[path]['positive_indices'])

    streamlines = load_streamlines(change_to_trk(path), idxs=indices, container='array')
    streamlines = spacing(change_to_trk(path), streamlines, target_spacing_mm=5)

    all_streams_tumor_points = ray_intersections_all_streamlines(streamlines, tumor_center, lesion_img, step_mm=0.1)
    sub_df1 = deformation_feature_1(bundle, streamlines)
    sub_df2 = deformation_feature_2(streamlines, tumor_center, all_streams_tumor_points)
    sub_df3 = deformation_feature_3(streamlines, all_streams_tumor_points)

    sub_df = {
        'hard_neg_features': [np.stack([a, b, c], axis=1) for a, b, c in zip(sub_df1[:len_hard], sub_df2[:len_hard], sub_df3[:len_hard])],
        'soft_neg_features': [np.stack([a, b, c], axis=1) for a, b, c in zip(sub_df1[len_hard:len_hard+len_soft], sub_df2[len_hard:len_hard+len_soft], sub_df3[len_hard:len_hard+len_soft])],
        'positive_features': [np.stack([a, b, c], axis=1) for a, b, c in zip(sub_df1[-len_pos:], sub_df2[-len_pos:], sub_df3[-len_pos:])],
    }
    with open(out_path, "wb") as f:
        pickle.dump(sub_df, f)
    print(f"Saved: {out_path}")
