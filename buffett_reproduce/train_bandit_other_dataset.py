"""For Banquet Experiments"""
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from typing import Optional
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from mir_eval.separation import bss_eval_sources
from sklearn.model_selection import train_test_split

from bandit_model import MyBandSplit
# from load_data import BEATS_path, ORIG_mixture, ORIG_target # , stems
# from dataset import MusicDataset
from loss import L1SNR_Recons_Loss, L1SNRDecibelMatchLoss
from utils import _load_config
from metrics import (
    AverageMeter, cal_metrics, safe_signal_noise_ratio, MetricHandler
)

from models.types import InputType, OperationMode, SimpleishNamespace
from data.moisesdb.datamodule import (
    MoisesTestDataModule,
    MoisesValidationDataModule,
    MoisesDataModule,
    MoisesBalancedTrainDataModule,
    MoisesVDBODataModule,
)




# Init settings
wandb_use = True # False
lr = 1e-3 #1e-4
num_epochs = 200
config_path = "config/train.yml"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on device:", device)


def to_device(batch, device=device):
    batch.mixture.audio = batch.mixture.audio.to(device) # torch.Size([2, 2, 294400])
    batch.sources.target.audio = batch.sources.target.audio.to(device) # torch.Size([2, 2, 294400])
    batch.query.audio = batch.query.audio.to(device) # torch.Size([2, 2, 441000])
    return batch



if wandb_use:
    wandb.init(
        project="Query_ss",
        config={
        "learning_rate": lr,
        "architecture": "Self-Bandquet Using Other's dataset",
        "dataset": "MoisesDB",
        "epochs": num_epochs,
        }
    )

config = _load_config(config_path)
stems = config.data.train_kwargs.allowed_stems
print("Training with stems: ", stems)

datamodule = MoisesDataModule(
    data_root=config.data.data_root,
    batch_size=config.data.batch_size,
    num_workers=config.data.num_workers,
    train_kwargs=config.data.get("train_kwargs", None),
    val_kwargs=config.data.get("val_kwargs", None),
    test_kwargs=config.data.get("test_kwargs", None), # Cannot use now
    datamodule_kwargs=config.data.get("datamodule_kwargs", None),
)



# for batch in tqdm(datamodule.train_dataloader()):
#     pass

# for batch in tqdm(datamodule.val_dataloader()):
#     pass

# # for batch in tqdm(datamodule.test_dataloader()):
#     # pass




# Instantiate the enrollment model
model_kwargs = config.model.get("kwargs", {})
model = MyBandSplit(
    **model_kwargs,
    stems=stems,
).to(device)


# Optimizer & Scheduler setup
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
criterion = L1SNRDecibelMatchLoss() #L1SNR_Recons_Loss()


early_stop_counter, early_stop_thres = 0, 4
min_val_loss = 1e10

# Training loop
for epoch in tqdm(range(num_epochs)):
    
    model.train()
    train_loss = 0.0
    
    for batch in tqdm(datamodule.train_dataloader()):
        batch = InputType.from_dict(batch)
        batch = to_device(batch)
        
        optimizer.zero_grad()
        
        # Forward pass
        batch = model(batch)

        # Compute the loss
        loss = criterion(batch.estimates["target"].audio, batch.sources["target"].audio) # Y_Pred, Y_True
        train_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    
    scheduler.step()


    # Validation step
    if epoch % 5 == 0:
        model.eval()
        val_loss = 0.0
        val_metric_handler = MetricHandler(stems)
        with torch.no_grad():
            for batch in tqdm(datamodule.val_dataloader()):
                batch = InputType.from_dict(batch)
                batch = to_device(batch)
                
                # Forward pass
                batch = model(batch)

                # Compute the loss
                loss = criterion(batch.estimates["target"].audio, batch.sources["target"].audio) # Y_Pred, Y_True
                val_loss += loss.item()

                # Calculate metrics
                val_metric_handler.calculate_snr(batch.estimates["target"].audio, batch.sources["target"].audio, batch.metadata.stem)

            # Record the validation SNR
            val_snr = val_metric_handler.get_mean_median()

        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val SNR: {val_snr}")
        if wandb_use:
            wandb.log({"val_loss": val_loss})
            wandb.log(val_snr)
            # wandb.log({"val_sdr": sdr, "val_sir": sir, "val_sar": sar})
            
        # Early stop
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_thres:
                break
            
    else:
        if wandb_use:
            wandb.log({"train_loss": train_loss})

    
    
    
# # Test step after all epochs
# model.eval()
# test_loss = 0.0
# test_metric_handler = MetricHandler(stems)

# with torch.no_grad():
#     for mixture, query_emb, target, stem in test_loader:
#         mixture = mixture.to(device)
#         query_emb = query_emb.mean(axis=1).to(device)
#         target = target.to(device)

#         batch = InputType(
#             mixture= SimpleishNamespace(
#                 audio=mixture, spectrogram=None
#             ),
#             sources={
#                 f"{stem}": SimpleishNamespace(
#                     audio=target, spectrogram=None
#                 ),
#             },
#             query=SimpleishNamespace(
#                 audio=target, spectrogram=None
#             ),
#             estimates={
#                 "target": SimpleishNamespace(
#                     audio=mixture, spectrogram=None
#                 ),
#             },
#             # metadata={"sample_rate": fs}
#         )

#         # Forward pass
#         batch = model(batch)

#         # Compute the loss
#         loss = criterion(batch.estimates["target"].audio, batch.sources[f"{stem}"].audio) # Y_Pred, Y_True
#         test_loss += loss.item()

#         # Calculate metrics
#         test_metric_handler.calculate_snr(batch.estimates["target"].audio, batch.sources[f"{stem}"].audio, stem)
    
#     # Get the final result of test SNR
#     test_snr = test_metric_handler.get_mean_median()
#     print("Test snr:", test_snr)
        

# print(f"Final Test Loss: {test_loss}")

# if wandb_use: wandb.finish()