# to train a neural network, i need
    # (1) data preparation/loading
    # (2) neural network construction - verifyber here with its architecture and forward pass
    # (3) loss function
    # (4) maybe optimizer?

    # (5) the algorithm - for testing loop
                        # iterate over a given set of data
                        # forward the data through the neural network
                        # compare the network output with the ground truth labels to compute the loss/evaluation metrics
                        
                        # for training loop is above +
                        # perform the backward pass to compute gradients
                        # call the optimizer to consequently update the weights optimizer.step()
                        # reset gradients to not accumulate them optimizer.zero_grad()

    # (6) logging

# libs

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as gData
from torch_geometric.data import Dataset as gDataset
from torch_geometric.data import DataListLoader as gDataLoader
from utils import change_path, load_streamlines

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
        self.path_to_entry = {entry['path']: entry for entry in bundle_idx['AF_L']}
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
        path_trk = change_path(path_bd)
        entry = self.path_to_entry[self.all_subs[idx]]

        indices = np.concatenate([entry['hard_neg_indices'], entry['soft_neg_indices'], entry['positive_indices']])
        sample = {'points': indices}

        if self.with_gt:
            labels = np.concatenate([np.zeros(len(entry['hard_neg_indices'])+len(entry['soft_neg_indices'])), np.ones(len(entry['positive_indices']))])
            sample['gt'] = labels
        if self.transform:
            sample = self.transform(sample)
        streams, lengths = load_streamlines(path_trk,sample['points'],container='array_flat')
        if self.permute:
            streams_perm = self.permute_pts(
                np.split(streams, np.cumsum(lengths))[:-1], type=self.permute_type)
            streams = np.concatenate(streams_perm, axis=0)

        # gt_tensor = torch.from_numpy(sample['gt']).long() if self.with_gt else None
        sample['points'] = self.build_graph_sample(streams,
                    lengths,
                    sample['gt'])
        del sample['gt']
        return sample
    
    def permute_pts(self, sl_list, type='rand'):
        perm_sl_list = []
        for sl in sl_list:
            if type == 'flip':
                perm_sl_list.append(sl[::-1])
            else:
                perm_idx = torch.randperm(len(sl)).tolist()
                perm_sl_list.append(sl[perm_idx])
        return perm_sl_list

    def build_graph_sample(self, streams, lengths, gt=None):
        ### create graph structure
        lengths = torch.from_numpy(lengths).long()
        batch_vec = torch.arange(len(lengths)).repeat_interleave(lengths)
        batch_slices = torch.cat([torch.tensor([0]), lengths.cumsum(dim=0)])
        slices = batch_slices[1:-1]
        streams = torch.from_numpy(streams)
        l = streams.shape[0]
        graph_sample = GlioDefData(x=streams,
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
    
