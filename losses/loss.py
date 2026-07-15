# this script's not used. i decided to use torch.nn.NLLLoss instantiation instead.

import torch.nn as nn
import torch.nn.functional as F 

class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, pred, target):
        loss = F.nll_loss(pred, target.long())
        return loss
        
def compute_loss(cfg, logits, target, loss_dict=None):
    tot_loss = 0.
    pred = F.log_softmax(logits, dim=-1).view(-1, int(cfg['n_classes']))
    criterion = NLLLoss()          # instantiate the actual class
    loss = criterion(pred, target.long())   # calls NLLLoss.forward
    tot_loss += loss
    return tot_loss
