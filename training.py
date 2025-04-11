import os
import sys
import random

import torch
import torch.nn as nn
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from LKEDTA import LKE_DTA
from utils import *

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

datasets = ['davis','kiba']
modelings = [LKE_DTA]
random_seed = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model,device,train_loader,optimizer,epoch):

    model.train()
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output,data.y.view(-1,1).float().to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % LOG_INTERVAL == 0 and dist.get_rank() == 0:
            print(f"[Rank {dist.get_rank()}] "
                  f"Epoch: {epoch} | Batch: {batch_idx} | "
                  f"Loss: {loss.item():.6f}")
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def predicting(model,device,loader):
    model.eval()
    total_preds = []
    total_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds.append(output.cpu())
            total_labels.append(data.y.view(-1,1).cpu())
    total_preds = torch.cat(total_preds,dim=0)
    total_labels = torch.cat(total_labels,dim=0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda",local_rank)
    set_seed(random_seed)
    for dataset in datasets:
        for modeling in modelings:
            model_st = modeling.__name__
            if dist.get_rank() == 0:
                print('\nrunning on ',model_st + '_' + dataset)
            processed_data_file_train = 'processed/' + dataset + '_train.pt'
            processed_data_file_test = 'processed/' + dataset + '_test.pt'

            if (not os.path.isfile(processed_data_file_train)) or \
                    (not os.path.isfile(processed_data_file_test)):
                if dist.get_rank() == 0:
                    print('please run preparation.py to prepare data!')
                continue

            train_data = TestbedDataset(root='./',dataset=dataset + '_train')
            test_data = TestbedDataset(root='./',dataset=dataset + '_test')


            train_sampler = DistributedSampler(train_data,shuffle=True)
            test_sampler = DistributedSampler(test_data,shuffle=False)

            train_loader = DataLoader(
                train_data,
                batch_size=TRAIN_BATCH_SIZE,
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=False
            )
            test_loader = DataLoader(
                test_data,
                batch_size=TEST_BATCH_SIZE,
                sampler=test_sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=False
            )

            model = modeling().to(device)
            ddp_model = DDP(model,device_ids=[local_rank],output_device=local_rank)

            global loss_fn
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(ddp_model.parameters(),lr=LR)

            best_mse = 1000
            best_ci = 0
            best_epoch = -1

            model_file_name = 'model_' + model_st + '_' + dataset + '.models'
            result_file_name = model_st + '_' + dataset + '.csv'

            for epoch in range(NUM_EPOCHS):

                train_sampler.set_epoch(epoch)

                avg_loss = train_one_epoch(ddp_model,device,train_loader,optimizer,epoch + 1)

                if dist.get_rank() == 0:
                    G,P = predicting(ddp_model,device,test_loader)
                    ret = [rmse(G,P),
                           mse(G,P),
                           pearson(G,P),
                           spearman(G,P),
                           ci(G,P)]
                    if ret[1] < best_mse:

                        torch.save(model.state_dict(),model_file_name)
                        with open(result_file_name,'w') as f:
                            f.write(','.join(map(str,ret)))
                        best_epoch = epoch + 1
                        best_mse = ret[1]
                        best_ci = ret[-1]

                        print(
                            f"[Rank 0] Epoch={best_epoch}  rmse={ret[0]:.4f}  mse={ret[1]:.4f}  ci={ret[-1]:.4f}  --> saved")
                    else:
                        print(f"[Rank 0] Epoch={epoch + 1}  mse={ret[1]:.4f}  No improvement since epoch {best_epoch}. "
                              f"Best_mse={best_mse:.4f}  best_ci={best_ci:.4f}")

    dist.destroy_process_group()
if __name__ == "__main__":
    main()
