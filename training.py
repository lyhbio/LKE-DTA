import os
import random
from typing import List

import torch
import torch.nn as nn
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset

from LKEDTA import LKE_DTA
from utils import *
TRAIN_BATCH_SIZE = 512
VAL_BATCH_SIZE = 512
LR = 5e-4
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

datasets = ['davis']
modelings = [LKE_DTA]
random_seed = 42



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % LOG_INTERVAL == 0 and dist.get_rank() == 0:
            print(f"[Rank {dist.get_rank()}] "
                  f"Epoch: {epoch} | Batch: {batch_idx} | "
                  f"Loss: {loss.item():.6f}")
    return running_loss / max(1, len(train_loader))


def predicting_all_ranks(model, device, loader):
    model.eval()
    total_preds, total_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            if dist.get_rank() == 0:
                total_preds.append(output.cpu())
                total_labels.append(data.y.view(-1, 1).cpu())
    if dist.get_rank() == 0 and len(total_preds) > 0:
        total_preds = torch.cat(total_preds, dim=0)
        total_labels = torch.cat(total_labels, dim=0)
        return total_labels.numpy().flatten(), total_preds.numpy().flatten()
    return None, None


def load_fold_dataset(dataset_name: str, fold_id: int):
    file = f'processed/{dataset_name}_fold{fold_id}.pt'
    if not os.path.isfile(file):
        return None
    return TestbedDataset(root='./', dataset=f"{dataset_name}_fold{fold_id}")


def safe_r2m(G, P):
    try:
        return r2m_score(G, P)
    except Exception:
        return None


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(random_seed)

    for dataset in datasets:
        for modeling in modelings:
            model_st = modeling.__name__
            if dist.get_rank() == 0:
                print(f"\n==> Running 5-fold CV on {model_st}_{dataset} (predefined folds)")

            folds_metrics = []
            header = ['rmse', 'mse', 'pearson', 'mae', 'r2', 'spearman', 'ci', 'r2m']

            # ----------- Folds 0~4 -----------
            for fold_id in range(5):
                test_ds = load_fold_dataset(dataset, fold_id)
                if test_ds is None:
                    if dist.get_rank() == 0:
                        print(f"[WARN] missing processed/{dataset}_fold{fold_id}.pt, skip.")
                    continue

                train_datasets, train_fold_ids = [], []
                for other_id in range(5):
                    if other_id == fold_id:
                        continue
                    other_ds = load_fold_dataset(dataset, other_id)
                    if other_ds is not None:
                        train_datasets.append(other_ds)
                        train_fold_ids.append(other_id)

                if len(train_datasets) == 0:
                    continue
                train_ds = ConcatDataset(train_datasets)

                if dist.get_rank() == 0:
                    print(f"\n[Fold {fold_id}/5]")
                    print(f"  Test set   : fold {fold_id} ({len(test_ds)} samples)")
                    print(f"  Train set  : folds {train_fold_ids} (total {len(train_ds)} samples)")

                train_loader = DataLoader(
                    train_ds,
                    batch_size=TRAIN_BATCH_SIZE,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=True
                )
                val_loader = DataLoader(
                    test_ds,
                    batch_size=VAL_BATCH_SIZE,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=False
                )

                model = modeling().to(device)
                ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
                global loss_fn
                loss_fn = nn.MSELoss()
                optimizer = torch.optim.Adam(ddp_model.parameters(), lr=LR)

                best_mse, best_ci, best_epoch = float('inf'), -float('inf'), -1
                best_metrics = None

                model_file = f'model_{model_st}_{dataset}_fold{fold_id}.pth'
                result_file = f'{model_st}_{dataset}_fold{fold_id}.csv'

                for epoch in range(1, NUM_EPOCHS + 1):
                    _ = train_one_epoch(ddp_model, device, train_loader, optimizer, epoch)

                    G, P = predicting_all_ranks(ddp_model, device, val_loader)
                    if dist.get_rank() == 0 and G is not None:
                        ret = [rmse(G, P), mse(G, P), pearson(G, P),
                               mae(G, P), r2(G, P), spearman(G, P), ci(G, P)]
                        r2m_val = safe_r2m(G, P)

                        if ret[1] < best_mse:
                            torch.save(model.state_dict(), model_file)
                            with open(result_file, 'w') as f:
                                f.write(','.join(map(str, ret + [r2m_val if r2m_val else ""])))
                            best_epoch, best_mse, best_ci = epoch, ret[1], ret[-1]
                            best_metrics = ret + [r2m_val]
                            print(f"[Rank 0][Fold {fold_id}] Epoch={best_epoch} "
                                  f"rmse={ret[0]:.4f} mse={ret[1]:.4f} ci={ret[-1]:.4f} "
                                  f"mae={ret[3]:.4f} r2={ret[4]:.4f} pearson={ret[2]:.4f} "
                                  f"r2m={r2m_val} --> saved")
                        else:
                            print(f"[Rank 0][Fold {fold_id}] Epoch={epoch} "
                                  f"mse={ret[1]:.4f} mae={ret[3]:.4f} r2={ret[4]:.4f} "
                                  f"No improvement since {best_epoch}, "
                                  f"Best_mse={best_mse:.4f} best_ci={best_ci:.4f}")

                del model, ddp_model, optimizer, train_loader, val_loader
                torch.cuda.empty_cache()
                dist.barrier()

                if dist.get_rank() == 0 and best_metrics is not None:
                    folds_metrics.append(best_metrics)

            if dist.get_rank() == 0 and len(folds_metrics) > 0:
                arr = np.array([[*m[:-1], (m[-1] if m[-1] else np.nan)] for m in folds_metrics], dtype=float)
                mean_vec, std_vec = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0, ddof=1)

                summary_file = f'{model_st}_{dataset}_5fold_summary.csv'
                with open(summary_file, 'w') as f:
                    f.write('metric,mean,std\n')
                    for h, m, s in zip(header, mean_vec, std_vec):
                        f.write(f'{h},{m:.6f},{s:.6f}\n')

                print("\n[Rank 0] ====== 5-Fold Summary ======")
                for h, m, s in zip(header, mean_vec, std_vec):
                    print(f"{h:>8}: mean={m:.4f} std={s:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
