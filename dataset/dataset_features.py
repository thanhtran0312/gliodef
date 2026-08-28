import numpy as np
import pickle
import torch
from torch_geometric.data import Data as gData
from torch_geometric.data import Dataset as gDataset
from training_script.utils.utils import change_to_trk, load_streamlines

class GlioDefData(gData):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'bvec':
            return int(self.lengths.numel())
        return super().__inc__(key, value, *args, **kwargs)
    
class GlioDefDataset(gDataset):
    """
    input is a list of subject ids and tumor ids already split and combined for training or testing
    it loads the whole set for training or testing, not just one fold
    eg: sub_xxxx-tum_xxxxx
    """
    def __init__(self, 
                 all_subs, 
                 bundle_idx,
                 deformation_features,
                 transform,
                 return_edges,
                 with_gt,
                 permute,
                 permute_type='flip'):
        # the config: where the subject list is (sub_file), where the data lives (root_dir)
        # whether its for training or testing, what augmentation/sampling to apply (transforms)
        # whether labels exist (with_gt), whether to build graph edges (return_edges), 
        # whether to exhaustively drain each subject across multiple calls (split_obj)
        # where labels live (labels_dir,labels_name), whether to apply flip augmentation (permute,permute_type)
        self.all_subs = all_subs
        # it reads the subjects' file paths, no trk
        bundle = list(bundle_idx.keys())[0]
        self.path_to_entry = {entry['path']: entry for entry in bundle_idx[bundle]}
        self.deformation_feature_paths = deformation_features
        self.return_edges = return_edges
        self.with_gt = with_gt
        self.transform = transform
        self.permute = permute
        self.permute_type = permute_type 

    def __len__(self):
        return len(self.all_subs)
    
    def __getitem__(self,idx):
        item = self.getitem(idx)
        return item
    
    def getitem(self,idx):
        path_bd = self.all_subs[idx]        
        path_trk = change_to_trk(path_bd)
        entry = self.path_to_entry[self.all_subs[idx]]
        # shape(num of streamlines,)
        indices = np.concatenate([entry['hard_neg_indices'], entry['soft_neg_indices'], entry['positive_indices']])
        sample = {'points': indices}

        if self.with_gt:
            labels = np.concatenate([np.zeros(len(entry['hard_neg_indices'])+len(entry['soft_neg_indices'])), np.ones(len(entry['positive_indices']))])
            sample['gt'] = labels
        if self.transform:
            sample = self.transform(sample)
        chosen_idx = sample['chosen_idx']
        # 1 trk/subject at a time; 
        # load_streamlines() returns all points of all streamlines of the subject concatenated
        # and lengths = number of points of each streamline
        streams, lengths = load_streamlines(path_trk,sample['points'],container='array_flat')
        deformation_features = self.load_deformation_features(idx, chosen_idx)
        ## it appends the reversed points of each streamline
        if self.permute:
            streams_perm = self.permute_pts(
                np.split(streams, np.cumsum(lengths))[:-1], type=self.permute_type)
            streams = np.concatenate(streams_perm, axis=0)
            df_perm = self.permute_pts(
                            np.split(deformation_features, np.cumsum(lengths))[:-1], type=self.permute_type)
            deformation_features = np.concatenate(df_perm, axis=0)

        # gt_tensor = torch.from_numpy(sample['gt']).long() if self.with_gt else None
        sample['points'] = self.build_graph_sample(
            streams=streams,
            deformation_features=deformation_features,
            lengths=lengths,
            gt=sample.get("gt"))
        del sample['gt']
        return sample
    def load_deformation_features(self, idx, chosen_idx):
        path = self.deformation_feature_paths[idx]

        with open(path, "rb") as f:
            features = pickle.load(f)

        return features[chosen_idx]
    ## it doesn't make the model invariant, but it changes the training data
    ## data augmentation function, training with data with random flips teaches
    ## the model that direction shouldn't matter for the label (because flipped or not, it has the same gt)
    def permute_pts(self, sl_list, type='rand'):
        perm_sl_list = []
        for sl in sl_list:
            ## if specified 'flip', flip
            if type == 'flip':
                perm_sl_list.append(sl[::-1])
            ## otherwise, shuffle the order of points -> point cloud style
            else:
                perm_idx = torch.randperm(len(sl)).tolist()
                perm_sl_list.append(sl[perm_idx])
        return perm_sl_list

    def build_graph_sample(self, streams, deformation_features, lengths, gt=None):
        ### create graph structure
        lengths = torch.from_numpy(lengths).long()
        batch_vec = torch.arange(len(lengths)).repeat_interleave(lengths)
        batch_slices = torch.cat([torch.tensor([0]), lengths.cumsum(dim=0)])
        slices = batch_slices[1:-1]
        streams = torch.from_numpy(streams)
        l = streams.shape[0]
        graph_sample = GlioDefData(x=deformation_features,
                     lengths=lengths,
                     bvec=batch_vec,
                     pos=streams)
        if self.return_edges:
            e1 = set(np.arange(0,l-1)) - set(slices.numpy()-1)
            e2 = set(np.arange(1,l)) - set(slices.numpy())
            edges = torch.tensor([list(e1)+list(e2),list(e2)+list(e1)],
                            dtype=torch.long)
            graph_sample['edge_index'] = edges
            num_edges = graph_sample.num_edges
            edge_attr = torch.ones(num_edges,1)
            graph_sample['edge_attr'] = edge_attr
        if gt is not None:
            graph_sample['y'] = torch.from_numpy(gt).long()

        return graph_sample
