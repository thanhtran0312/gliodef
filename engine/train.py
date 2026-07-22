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
import sys
import json
import torch
import time
import numpy as np
import tomli as tomllib
from pathlib import Path
from torch_geometric.loader import DataLoader as gDataLoader
import torch.nn.functional as F


project_parent = Path.cwd().parents[1]   # from training_script/data -> gliodef_script
sys.path.insert(0, str(project_parent))
from training_script.data.dataset import GlioDefDataset
from training_script.utils.transforms import RndSampling
from training_script.models.verifyber import DECSeq
from training_script.utils.transforms import RndSampling
from training_script.utils.train_utils import (
    create_tb_logger,
    dump_code,
    dump_model,
    get_lr,
    get_lr_scheduler,
    initialize_loss_dict,
    log_losses,
    update_bn_decay,
)
from training_script.utils.utils import (
    initialize_metrics,
    log_avg_metrics,
    update_metrics,
    get_metrics_inline,
)

def train_epoch(cfg, loader, model, optimizer, writer, epoch, n_iter):
        """run one epoch of training
        """
        model.train()
        num_classes = int(cfg["n_classes"])
        num_batch = cfg["num_batch"]
        ep_loss = 0.0
        ep_loss_dict = initialize_loss_dict(cfg)
        metrics = initialize_metrics()
        t0 = time.time()
        for i_batch, sample_batched in enumerate(loader):
            data = sample_batched['points']
            target = data.y   # ground truth
            data,target = data.to("cuda"), target.to("cuda")

            if not cfg["accumulation_interval"] or i_batch == 0:
                optimizer.zero_grad()
            
            # forward
            logits = model(data)

            # loss
            criterion = torch.nn.NLLLoss()
            pred =  F.log_softmax(logits, dim=-1)
            loss = criterion(pred, target)
            ep_loss += loss.item()
            # running_ep_loss = ep_loss / (i_batch + 1)
            loss.backward()

            # if (i_batch + 1) % int(cfg["accumulation_interval"]) == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            # elif not cfg["accumulation_interval"]:
            #     optimizer.step()

            optimizer.step()
            optimizer.zero_grad()
            
            # print which batch and how long did it take
            if i_batch % 10 == 0:
                elapsed = time.time() - t0
                per_batch = elapsed / 10 if i_batch > 0 else elapsed
                print(f'  batch {i_batch}/{len(loader)}: loss {loss.item():.4f}  ({per_batch:.2f}s/batch)')
                t1 = time.time()
                print(f'{t1}-{t0}s')

            # compute performance
            if cfg['task'] == 'classification':
                pred =  F.log_softmax(logits, dim=-1).view(-1, num_classes)
                pred_choice = pred.data.max(1)[1].int()
                update_metrics(metrics, pred_choice, target, task=cfg["task"])
                print(
                    "[%d: %d/%d] train loss: %f %s"
                    % (
                        epoch,
                        i_batch,
                        num_batch,
                        loss.item(),
                        get_metrics_inline(metrics, "last"),
                    )
                )
            else:
                update_metrics(metrics, logits.float(), target.float(), task=cfg["task"])
                print(
                    "[%d: %d/%d] train loss: %f %s"
                    % (
                        epoch,
                        i_batch,
                        num_batch,
                        loss.item(),
                        get_metrics_inline(metrics, "last"),
                    )
                )
            n_iter += 1
        ep_loss = ep_loss / (i_batch + 1)
        writer.add_scalar("train/epoch_loss", ep_loss, epoch)
        log_losses(ep_loss_dict, writer, epoch, i_batch + 1)
        log_avg_metrics(writer, metrics, "train", epoch)
        return ep_loss, n_iter

def validate_epoch(cfg, loader, model, writer, epoch, best_epoch, best_score):
    best = False
    num_classes = int(cfg['n_classes'])
    model.eval()
    with torch.no_grad():
        metrics_val = initialize_metrics()
        ep_loss = 0.0
        for i,sample in enumerate(loader):
            data = sample['points']
            target = data['y']
            data,target = data.to("cuda"), target.to("cuda")

            log_str = "VALIDATION [%d: %d/%d] " % (epoch, i, len(loader))
            logits = model(data)

            criterion = torch.nn.NLLLoss()
            pred = F.log_softmax(logits,dim=-1)
            loss = criterion(pred,target)
            ep_loss += loss.item()
            
            ref_metrics = "acc"
            pred = F.log_softmax(logits, dim=-1).view(-1, num_classes)
            pred_choice = pred.data.max(1)[1].int()
            update_metrics(metrics_val, pred_choice, target, task=cfg["task"])
            print(
                    "val min / max class pred %d / %d"
                    % (pred_choice.min().item(), pred_choice.max().item())
                )
            print("# class pred ", len(torch.unique(pred_choice)))
                # writer.add_scalar('val/loss', ep_loss / i, epoch)
        else:
            ref_metrics = "mse"
            update_metrics(
                    metrics_val, logits.float(), target.float(), task=cfg["task"]
                )
        log_str += "loss: %.4f " % loss.item()
        log_str += get_metrics_inline(metrics_val, type="last")
        print(log_str)
    log_avg_metrics(writer, metrics_val, "val", epoch)
    epoch_score = torch.tensor(metrics_val[ref_metrics]).mean().item()
    print("VALIDATION AVG: %s" % get_metrics_inline(metrics_val, "avg"))
    print("\n\n")

    if ref_metrics == "acc" and epoch_score > best_score:
            best_score = epoch_score
            best_epoch = epoch
            best = True
    elif ref_metrics == "mse" and epoch_score < best_score:
            best_score = epoch_score
            best_epoch = epoch
            best = True

    if cfg["save_model"]:
            dump_model(cfg, model, writer.log_dir, epoch, epoch_score, best=best)

    return best_epoch, best_score

def train(cfg, bundle_idx, training_data,testing_data):
    batch_size = int(cfg["batch_size"])
    n_epochs = int(cfg["n_epochs"])
    sample_size = int(cfg["fixed_size"])


    train_dataset = GlioDefDataset(training_data, bundle_idx=bundle_idx,
                                transform=RndSampling(sample_size, maintain_prop=False),
                                return_edges=True, with_gt=True, permute=True, permute_type='flip')
    val_dataset = GlioDefDataset(testing_data, bundle_idx=bundle_idx,
                              transform=RndSampling(sample_size, maintain_prop=False), return_edges=True,
                              with_gt=True, permute=True, permute_type='flip')


    train_loader = gDataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = gDataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)

    writer = create_tb_logger(cfg)
    dump_code(cfg, writer.log_dir)

    model = DECSeq(input_size=3, n_classes=cfg['n_classes']).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'],weight_decay=float(cfg['weight_decay']))
    lr_scheduler = get_lr_scheduler(cfg, optimizer)
    num_batch = len(train_dataset) / batch_size
    print("num of batches per epoch: %d" % num_batch)
    cfg["num_batch"] = num_batch
    
    n_iter = 0
    if cfg["task"] == "classification":
        best_pred = 0
    else:
        best_pred = np.inf
    best_epoch = 0
    current_lr = float(cfg["learning_rate"])
    initial_nll_w = cfg["nll_w"]
    
    for epoch in range(n_epochs + 1):

        # update bn decay
        if cfg["bn_decay"] and epoch != 0 and epoch % int(cfg["bn_decay_step"]) == 0:
            update_bn_decay(cfg, model, epoch)

        if cfg["nll_w_decay"] and epoch % int(cfg["nll_w_decay_step"]) == 0:
            cfg["nll_w"][0] = initial_nll_w[0] * cfg["nll_w_decay"] ** epoch

        loss, n_iter = train_epoch(
            cfg, train_loader, model, optimizer, writer, epoch, n_iter
        )

        ### validation during training
        if epoch % int(cfg["val_freq"]) == 0 and cfg["val_in_train"]:
            best_epoch, best_pred = validate_epoch(
                cfg, val_loader, model, writer, epoch, best_epoch, best_pred
            )

        # update lr
        if cfg["lr_type"] == "step" and current_lr >= float(cfg["min_lr"]):
            lr_scheduler.step()
        if cfg["lr_type"] == "plateau":
            lr_scheduler.step(loss)

        current_lr = get_lr(optimizer)
        writer.add_scalar("train/lr", current_lr, epoch)
    writer.close()

if __name__ == '__main__':
    output_dir = '/home/thanh/output'
    with open(os.path.join(output_dir,'cv_folds_tum.json'),'r') as f:
        cv_folds_tum = json.load(f)
    with open(os.path.join(output_dir,'cv_folds_sub.json'),'r') as f:
        cv_folds_sub = json.load(f)
    with open(os.path.join(output_dir,'bundle_idx.json'),'r') as f:
        bundle_idx = json.load(f)
    config_file = Path('config.toml')
    with config_file.open('rb') as fid:
        cfg = tomllib.load(fid)
        cfg = cfg["DEFAULT"]

    subjects_pool = [bundle_idx['AF_L'][j]['path'] for j in range(len(bundle_idx['AF_L']))]
    for i in range(cfg["folds"]):
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
    train(cfg, bundle_idx, training_data,testing_data)
