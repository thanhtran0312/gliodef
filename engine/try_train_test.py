# train 5 folds
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

import os
import json
import torch
from torch_geometric.loader import DataLoader as gDataLoader

import torch.nn.functional as F
from torch_geometric.data import Batch

from transforms import RndSampling
from dataset import GlioDefDataset
from verifyber import DECSeq
from loss import NLLLoss

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--folds', type=int, nargs='+', required=True)  # e.g. --folds 0 1 2
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

device = args.device

# in your for-loop over folds, use: for i in args.folds:
# plain "config" variables
n_classes = 2
n_epochs = 2
lr = 1e-3
batch_size = 2
val_freq = 5

output_dir = '/home/thanh/output'
with open(os.path.join(output_dir,'cv_folds_tum.json'),'r') as f:
    cv_folds_tum = json.load(f)
with open(os.path.join(output_dir,'cv_folds_sub.json'),'r') as f:
    cv_folds_sub = json.load(f)
with open(os.path.join(output_dir,'bundle_idx.json'),'r') as f:
    bundle_idx = json.load(f)

def collate_pyg(batch):
        return Batch.from_data_list(batch)

def train_epoch(model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0.0
        for i, sample in enumerate(loader):
            data = sample['points'].to(device)
            target = data.y.long()

            optimizer.zero_grad()
            logits = model(data)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = criterion(log_probs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f'  batch {i}/{len(loader)}: loss {loss.item():.4f}')
        
        return total_loss / (i + 1)

def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for sample in loader:
            data = sample['points'].to(device)
            target = data.y.long()
            logits = model(data)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            del data, logits, pred  # release GPU refs promptly
    torch.cuda.empty_cache()  # return freed blocks to the allocator
    return correct / total

fold_results = []
subjects_pool = [bundle_idx['AF_L'][j]['path'] for j in range(len(bundle_idx['AF_L']))]

for i in args.folds:
    testing_data = []
    testing_tum = cv_folds_tum[i]
    testing_sub = cv_folds_sub[i]

    training_data = []
    training_sub = [s for j, fold in enumerate(cv_folds_sub) if i != j for s in fold]
    training_tum = [t for j, fold in enumerate(cv_folds_tum) if i != j for t in fold]   
    for sub in training_sub:
        for tum in training_tum:
            path = f'/nilab-nexus/datasets/GLIODEF/sub-{sub}/tractography/sub-{sub}_tum-{tum}_bundle.csv'
            if path in subjects_pool:
                training_data.append(path)

    for sub in testing_sub:
        for tum in testing_tum:
            path = f'/nilab-nexus/datasets/GLIODEF/sub-{sub}/tractography/sub-{sub}_tum-{tum}_bundle.csv'
            if path in subjects_pool:
                testing_data.append(path)

    with open(os.path.join(output_dir, f'fold{i}_files.json'), 'w') as f:
        json.dump({'train': training_data, 'test': testing_data}, f)
 
    train_dataset = GlioDefDataset(training_data, bundle_idx=bundle_idx,
                                transform=RndSampling(8000, maintain_prop=False),
                                return_edges=True, with_gt=True, permute=True, permute_type='flip')
    val_dataset = GlioDefDataset(testing_data, bundle_idx=bundle_idx,
                              transform=RndSampling(8000, maintain_prop=False), return_edges=True,
                              with_gt=True, permute=True, permute_type='flip')

    train_loader = gDataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pyg)
    val_loader = gDataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg)

    model = DECSeq(input_size=3, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    criterion = NLLLoss()
    history = {'epoch': [], 'train_loss': [], 'val_acc': []}

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch}: train loss {train_loss:.4f} ')
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        if epoch % val_freq == 0:
            acc = validate(model, val_loader, device)
            print(f'  val acc: {acc:.4f}')
            history['val_acc'].append(acc)

            if acc > best_acc:
                best_acc = acc

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': acc,
                    'train_loss': train_loss,
                    'fold': i,
                }, os.path.join(output_dir, f'best_model_fold{i}.pt'))
    fold_results.append({'fold': i, 'best_val_acc': best_acc})

    with open(os.path.join(output_dir, f'history_fold{i}.json'), 'w') as f:
        json.dump(history, f)
    del model, optimizer, train_loader, val_loader, train_dataset, val_dataset
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
fold_tag = '-'.join(str(i) for i in args.folds)
with open(os.path.join(output_dir, f'cv_summary_{fold_tag}.json'), 'w') as f:
    json.dump(fold_results, f)
