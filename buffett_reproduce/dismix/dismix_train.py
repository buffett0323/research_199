import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dismix_model import DisMixModel
from dataset import CocoChoralesTinyDataset
from dismix_loss import ELBOLoss, BarlowTwinsLoss

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
batch_size = 4


# Usage Example
data_dir = '/home/buffett/NAS_189/cocochorales_output/main_dataset/'
train_dataset = CocoChoralesTinyDataset(data_dir, split='train')
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Models and other settings
model = DisMixModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0004)

# Loss function
elbo_loss_fn = ELBOLoss(lambda_recon=1.0, lambda_kl=0.1) # For ELBO
bce_loss_fn = nn.BCEWithLogitsLoss()  # For pitch supervision
bt_loss_fn = BarlowTwinsLoss(lambda_off_diag=0.0051) # Barlow Twins


# Training
c = 0
for batch in tqdm(train_data_loader):
    print(batch["mixture"].shape, batch["query"].shape, batch["pitch_label"])
    c += 1
    if c > 10: break


