import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torch.nn.functional import gumbel_softmax
from torch.optim import Adam
from torch.distributions import RelaxedOneHotCategorical
from PIL import Image
import argparse
from Model import Encoder, Decoder, Sender, Receiver, Receiver_nn, Model
from loss_functions import compute_loss_dispatcher



# Model Hyperparameters
cuda = False
DEVICE = torch.device("cuda" if cuda else "cpu")
print(DEVICE)

batch_size = 256
img_size = (45, 45)  # (width, height)

lr = 2e-4

epochs = 0

print_step = 800

# Create the parser
parser = argparse.ArgumentParser(description='Training script for the model.')

# Add arguments
parser.add_argument('-epochs', type=int, help='Number of epochs', required=True)
parser.add_argument('-path_to_file', type=str, help='Path to the dataset file', required=True)
parser.add_argument('-loss_mode', type=int, choices=[1, 2, 3, 4], default=4, help='Loss function choice: 1=Regular, 2=Progressive, 3=Progressive Strict, 4=Progressive Strict with Containing Bias')
parser.add_argument('-sentence_length', type=int, default=5, help='Set sentence length')
parser.add_argument('-vocab_size', type=int, default=100, help='Set vocabulary size')


# Parse the arguments
args = parser.parse_args()

# Use the arguments
epochs = args.epochs
path_to_file = args.path_to_file
loss_mode = args.loss_mode
vocab_size = args.vocab_size

# Set random seeds
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Load data
import pickle
with open(path_to_file, 'rb') as handle:
    mlpimages,mlplabels = pickle.load(handle)

mlpimages = mlpimages.view(-1,3,45,45)
print(mlpimages.shape)
print(mlplabels.shape)



# Assuming mlpimages and mlplabels are already defined and are tensors
dataset = TensorDataset(mlpimages, mlplabels)

# Define the train and validation split ratios
train_ratio = 0.9
val_ratio = 0.1

# Calculate the lengths of the train and validation datasets
train_length = int(len(dataset) * train_ratio)
val_length = len(dataset) - train_length

# Split the dataset into train and validation datasets
train_dataset, val_dataset = random_split(dataset, [train_length, val_length])

# Assuming batch_size is defined
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Calculate the percentage of samples
total_samples = len(dataset)
train_samples = len(train_loader.dataset)
val_samples = len(val_loader.dataset)

train_percentage = (train_samples / total_samples) * 100
val_percentage = (val_samples / total_samples) * 100

print(f"Train dataset length: {train_samples}")
print(f"Validation dataset length: {val_samples}")
print(f"Percentage of samples in train_loader: {train_percentage:.2f}%")
print(f"Percentage of samples in val_loader: {val_percentage:.2f}%")


#Define Hyperparameters
input_dim = 3 #Encoder
hidden_dim = 50
output_dim = 3
embed_dim =50


#Create Model components
encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
decoder = Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim)
sender = Sender(vocab_size, embed_dim, hidden_dim *12 , sentence_length, 1.0, True)
receiver_nn = Receiver_nn(hidden_dim*12)
receiver = Receiver(vocab_size,embed_dim,hidden_dim*12,receiver_nn)

#Create Model
model = Model(Encoder=encoder, Decoder=decoder, Sender = sender, Receiver = receiver).to(DEVICE)

def compute_loss(x, x_hat1, ica_orders, loss_mode, sentence_length):
    if loss_mode == 1:
        return compute_loss_regular(x, x_hat1, sentence_length)
    elif loss_mode == 2:
        return compute_loss_progressive(x, x_hat1, ica_orders)
    elif loss_mode == 3:
        return compute_loss_progressive_strict(x, x_hat1, ica_orders)
    elif loss_mode == 4:
        return compute_loss_progressive_strict_containing_bias(x, x_hat1, ica_orders)
    else:
        raise ValueError("Invalid loss_mode. Must be 1,2,3, or 4.")




# # Initialize the optimizer
mse_loss = nn.MSELoss()
# optimizer = Adam(params_to_optimize, lr=lr)
optimizer = Adam(list(model.parameters()), lr=lr)


model = model.to(DEVICE)

print("Start training")
model.train()

# Lists to keep track of validation and training losses
losses = []
val_losses = []

# weights = [0.0001, 0.00002, 0.000004, 0.0000008, 0.00000016]
epoch = 0

# Your main training loop:
for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        ica_orders = None

        x = x.to(DEVICE)
        optimizer.zero_grad()

        # For Progressive-type losses, we get ICA order values. If the regular loss is chosen, ica_orders remains None because it is unsupervised
        if loss_mode in [2, 3, 4]:
            ica_orders = torch.div(y, int(1000 / sentence_length), rounding_mode='trunc')
            ica_orders -= 1


        x_hat1, message, z, receiver_op = model(x)  # x_hat1 shape: [batch_size, sentence_length, C, H, W]

        # Dispatch to the correct loss function
        recon_loss = compute_loss_dispatcher(
            x=x,
            x_hat1=x_hat1,
            ica_orders=ica_orders,
            loss_mode=loss_mode,
            sentence_length=sentence_length
        )

        recon_loss.backward()
        optimizer.step()

        if batch_idx % print_step == 0:
            losses.append(recon_loss.item())
            print("Training epoch:", epoch + 1)
            print("epoch:", epoch + 1, "  step:", batch_idx + 1, " recon_loss:", recon_loss.item())


    # Validation part
    model.eval()  # switch model to the evaluation mode
    with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
        for batch_idx, (x, _) in enumerate(val_loader):
            ica_orders = torch.div(_, 100, rounding_mode='trunc')
            ica_orders-=1
            x = x.to(DEVICE)
            if loss_mode in [2, 3, 4]:
                ica_orders = torch.div(y, int(1000 / sentence_length), rounding_mode='trunc')
                ica_orders -= 1
            else:
                ica_orders = torch.zeros_like(y)

            x_hat1, message, z, receiver_op = model(x)
            recon_loss = compute_loss_dispatcher(
                x=x,
                x_hat1=x_hat1,
                ica_orders=ica_orders,
                loss_mode=loss_mode,
                sentence_length=sentence_length
            )
            recon_loss = mse_loss(x_hat, x)

            if batch_idx % print_step == 0:
                val_losses.append(recon_loss.item())
                print("epoch:", epoch + 1, "  step:", batch_idx + 1, " recon_loss:", recon_loss.item())

    model.train()  # switch model back to the train mode

print("Finish!!")

final_epoch = epoch + 1 
checkpoint_path = f'./checkpoints_epoch_{final_epoch}_ICA_all_sent5_progressive.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': final_epoch,
    'train_losses': losses,     
    'val_losses': val_losses       
}, checkpoint_path)

print(f"Final model saved at: {checkpoint_path}")
