import numpy as np
import nibabel as nib
from numba import jit
import os
from time import time
from dipy.tracking.streamlinespeed import length, set_number_of_points
from dipy.tracking.streamline import select_random_set_of_streamlines, set_number_of_points, length, Streamlines

def change_path(path_bundle):
    """arg: change path from *bundle.csv to *.trk"""
    path_trk = path_bundle.replace("bundle.csv","track.trk")
    return path_trk

def resample_streamlines(streamlines, n_pts=16):
    resampled = []
    for sl in streamlines:
        resampled.append(set_number_of_points(sl, n_pts))

    return resampled

def parse_lengths(buffer, lengths, point_size, n_properties):
    pointer = 0
    for idx in range(lengths.size):
        l = buffer[pointer]
        lengths[idx] = l
        pointer += 1 + l * point_size + n_properties
    return lengths

def parse_streamlines(buffer,
                      idxs,
                      split_points,
                      n_floats,
                      affine,
                      apply_affine=True,
                      rescale_factor=None,
                      reorient=False):
    streamlines = []
    rotation_zoom_shear = affine[:3, :3].copy(
    )  # necessary to create a C-contiguous array
    translation = affine[:3, 3:4].copy(
    )  # necessary to create a C-contiguous array
    for idx in idxs:  # range(n_floats.size):
        s = buffer[split_points[idx]:split_points[idx] +
                   n_floats[idx]].reshape(-1, 3)
        if apply_affine:
            s = (np.dot(rotation_zoom_shear, s.T) + translation).T
        if rescale_factor is not None:
            s *= rescale_factor
        if reorient:
            o = np.zeros(3) # origin
            start = np.argmin(np.array([np.sqrt(np.sum((o - s[0])**2)),
                np.sqrt(np.sum((o - s[-1])**2))]))
            if start == 1:
                s = s[::-1]

        streamlines.append(s)

    return streamlines

def load_streamlines(trk_fn,
                     idxs=None,
                     apply_affine=True,
                     resample=False,
                     container='array_flat',
                     replace=False,
                     verbose=False,
                     load_twice=True,
                     rescale=False,
                     reorient=False,
                     return_len=False):
    """Load streamlines from a .trk file. If a list of indices (idxs) is
    given, this function just loads and returns the requested
    streamlines, skipping all the non-requested ones.

    This function is sort of similar to nibabel.streamlines.load() but
    extremely FASTER. It is very convenient if you need to load only
    some streamlines in large tractograms. Like 100x faster than what
    you can get with nibabel.
    """

    if verbose:
        print("Loading %s" % trk_fn)

    lazy_trk = nib.streamlines.load(trk_fn, lazy_load=True)
    header = lazy_trk.header
    header_size = header['hdr_size']
    nb_streamlines = header['nb_streamlines']
    n_scalars = header['nb_scalars_per_point']
    n_properties = header['nb_properties_per_streamline']
    aff = nib.streamlines.trk.get_affine_trackvis_to_rasmm(header)
    vol_size = header['dimensions']
    vox_size = header['voxel_sizes']

    if idxs is None:
        idxs = np.arange(nb_streamlines, dtype=np.int32)
    elif isinstance(idxs, int):
        if verbose:
            print('Sampling %s streamlines uniformly at random' % idxs)

        if idxs > nb_streamlines and (not replace):
            print('WARNING: Sampling with replacement')

        idxs = np.random.choice(np.arange(nb_streamlines),
                                idxs,
                                replace=replace)
    elif isinstance(idxs, list):
        idxs = np.array(idxs, dtype=int)

    ## See: http://www.trackvis.org/docs/?subsect=fileformat
    length_bytes = 4
    point_size = 3 + n_scalars
    nb_bytes_float32 = np.dtype('<f').itemsize

    if verbose:
        print("Loading the whole data blob.")
        t0 = time()

    buffer = np.empty((os.path.getsize(trk_fn) // nb_bytes_float32),
                      np.float32)
    with open(trk_fn, 'rb') as f:
        f.seek(1000)  # 1000 is the size of the header in bytes
        f.readinto(buffer)

    if verbose:
        print("%s values read in %s sec." % (buffer.size, time() - t0))

    if verbose:
        print("Parsing lengths of %s streamlines" % nb_streamlines)
        t0 = time()

    lengths = np.empty(nb_streamlines, dtype=np.int32)
    buffer_int32 = buffer.view(dtype=np.int32)
    lengths = parse_lengths(buffer_int32, lengths, point_size, n_properties)

    if verbose:
        print("%s sec." % (time() - t0))

    n_floats = lengths * point_size
    split_points = (n_floats + 1 +
                    n_properties).cumsum() - n_floats - n_properties

    if verbose:
        print("Extracting %s streamlines" % nb_streamlines)
        if apply_affine:
            print("and applying the affine")

        t0 = time()

    scaling = 1/(vol_size * vox_size) if rescale else None

    streamlines = parse_streamlines(buffer, idxs, split_points, n_floats, aff,
                                    apply_affine, scaling, reorient)

    if verbose:
        print("%s sec." % (time() - t0))

    if verbose:
        print("Converting all streamlines to the container %s" % container)
        t0 = time()

    if resample == 'fixed_step':
        for i, s in enumerate(streamlines):
            l = length(s)
            lengths[idxs[i]] = int(l/5)
            streamlines[i] = set_number_of_points(s, lengths[idxs[i]])
    elif resample:
        streamlines = set_number_of_points(streamlines, resample)
        lengths[:] = resample

    if container == 'array':
        streamlines = np.array(streamlines, dtype=object)
    elif container == 'ArraySequence':
        streamlines = nib.streamlines.ArraySequence(streamlines)
    elif container == 'list':
        pass
    elif container == 'array_flat':
        streamlines = np.concatenate(streamlines, axis=0)
        return_len = True
    else:
        raise Exception

    if verbose:
        print("%s sec." % (time() - t0))

    if return_len:
        return streamlines, lengths[idxs]
    else:
        return streamlines


if __name__ == '__main__':

    np.random.seed(0)

    trk_fn = 'sub-100206_var-FNAL_tract.trk'
    trk_fn = 'sub-599469_var-10M_tract.trk'

    # idxs = np.random.choice(500000, 200000, replace=True)
    # idxs.sort()
    idxs = None  # This is for loading all streamlines
    # idxs = 1000

    streamlines, header, lengths, idxs = load_streamlines(trk_fn,
                                                          idxs,
                                                          apply_affine=True,
                                                          container='list',
                                                          verbose=True)

    print("Done.")

import os
import torch
from datetime import date
import numpy as np
import random
import configparser

def is_float(val):
    try:
        num = float(val)
    except ValueError:
        return False
    return True

def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True

def get_cfg_value(value):
    if value[0] == '[' and value[-1] == ']':
        value = [get_cfg_value(v) for v in value[1:-1].split()]
        return value
    if value == 'y':
        return True
    if value == 'n':
        return False
    if is_int(value):
        return int(value)
    if is_float(value):
        return float(value)
    return value

def set_exp_name(cfg, modelname, dataname):
    exp = cfg['experiment_name']
    exp = exp.replace('DATE', str(date.today()))
    exp = exp.replace('MODEL', modelname.lower())
    exp += '_data-{}'.format(dataname.lower())
    cfg['experiment_name'] = exp
    return

def print_cfg(cfg, fileobj=None):
    for k in sorted(cfg.keys()):
        line = '%s : %s' % (k, cfg[k])
        if fileobj is None:
            print(line)
        else:
            fileobj.write(line + '\n')

def save_dict_to_file(dic, filename):
    f = open(filename, 'w')
    f.write(str(dic))
    f.close()

def load_dict_from_file(filename):
    f = open(filename, 'r')
    data = f.read()
    f.close()
    return eval(data)

def initialize_metrics():
    metrics = {}
    metrics['acc'] = []
    metrics['iou'] = []
    metrics['prec'] = []
    metrics['recall'] = []
    metrics['mse'] = []
    metrics['abse'] = []

    return metrics


def update_metrics(metrics, prediction, target, task='classification'):

    if task == 'classification':
        prediction = prediction.data.int().cpu()
        target = target.data.int().cpu()

        correct = prediction.eq(target).sum().item()
        acc = correct / float(target.size(0))

        tp = torch.mul(prediction, target).sum().item() + 0.00001
        fp = prediction.gt(target).sum().item()
        fn = prediction.lt(target).sum().item()
        tn = correct - tp

        iou = float(tp) / (tp + fp + fn)
        prec = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        
        metrics['prec'].append(prec)
        metrics['recall'].append(recall)
        metrics['acc'].append(acc)
        metrics['iou'].append(iou)
    else:
        prediction = prediction.data.cpu()
        target = target.data.cpu()

        abs_err = torch.mean(abs(prediction-target))
        mserr = torch.mean((target-prediction)**2)
        
        metrics['abse'].append(abs_err)
        metrics['mse'].append(mserr)


def get_metrics_inline(metrics, type='avg'):
    s = ''
    if type == 'avg':
        s = ', '.join(['%s : %.4f' % (k[:3], torch.tensor(v).mean())
                     for k, v in metrics.items() if len(v) > 0])
    elif type == 'last':
        s = ', '.join(['%s : %.4f' % (k[:3], v[-1])
                     for k, v in metrics.items() if len(v) > 0])
    return s


def log_avg_metrics(writer, metrics, prefix, epoch):
    for k, v in metrics.items():
        if type(v) == list:
            v = torch.tensor(v)
        if len(v) == 0:
            continue
        writer.add_scalar('%s/epoch_%s' % (prefix, k), v.mean().item(), epoch)
        #writer.add_scalar('%s/epoch_%s' % (prefix, k), v.float().mean().item(), epoch)

def batched_cdist_l2(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_nrom = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(x2_norm.transpose(-2, -1),
                        x1,
                        x2.transpose(-2, -1),
                        alpha=-2).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
