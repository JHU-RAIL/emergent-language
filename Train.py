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

# Parse the arguments
args = parser.parse_args()

# Use the arguments
epochs = args.epochs
path_to_file = args.path_to_file


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
sentence_length=10
vocab_size = 100
embed_dim =50


#Create Model components
encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
decoder = Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim)
sender = Sender(vocab_size, embed_dim, hidden_dim *12 , sentence_length, 1.0, True)
receiver_nn = Receiver_nn(hidden_dim*12)
receiver = Receiver(vocab_size,embed_dim,hidden_dim*12,receiver_nn)

#Create Model
model = Model(Encoder=encoder, Decoder=decoder, Sender = sender, Receiver = receiver).to(DEVICE)

#Define the containing bias loss function
def asymmetric_mse_loss(output, target, smaller_weight=1.5, larger_weight=0.5):
    """
    Asymmetric MSE loss which penalizes more heavily when the output pixel value
    is smaller than the target pixel value.
    """
    # Find where output is less than target
    smaller = (output < target).float()
    # Find where output is greater than or equal to target
    larger_or_equal = (output >= target).float()

    # Calculate the asymmetric weighted loss
    loss = (smaller_weight * smaller * (target - output) ** 2 +
            larger_weight * larger_or_equal * (output - target) ** 2)
    return loss.mean()




# # Initialize the optimizer
mse_loss = nn.MSELoss()
# optimizer = Adam(params_to_optimize, lr=lr)
optimizer = Adam(list(model.parameters()), lr=lr)


model = model.to(DEVICE)


#Run this for progressive

print("Start training VQ-VAE...")
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
        ica_orders = torch.div(_, 100, rounding_mode='trunc')
        ica_orders-=1

        x = x.to(DEVICE)
        optimizer.zero_grad()
        x_hat1, message, z, receiver_op = model(x)
        # Expand dimensions for advanced indexing
        ica_orders_expanded = ica_orders.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Gather along the second dimension using ica_orders
        x_hat = torch.gather(x_hat1, 1, ica_orders_expanded.expand(-1, -1, 3, 45, 45).to(DEVICE)).squeeze(1)

        recon_loss = mse_loss(x_hat, x)

        loss =  recon_loss

        loss.backward()

        optimizer.step()

        if batch_idx % print_step == 0:
            losses.append(recon_loss.item())
            print("Training epoch:", epoch + 1)
            print("epoch:", epoch + 1, "  step:", batch_idx + 1, " recon_loss:", recon_loss.item())
            # print("step: {}  recon_loss: {:.8f}  diversity penalty: {}  total diversity: {:.8f}  total loss: {:.8f}".format(
            #     batch_idx + 1,
            #     recon_loss.item(),
            #     [round(val, 8) for val in diversity_penalty],  # This will round each value in the list to 8 decimal places
            #     sum(diversity_penalty),
            #     loss.item()
            # ))

    # Validation part
    model.eval()  # switch model to the evaluation mode
    with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
        for batch_idx, (x, _) in enumerate(val_loader):
            ica_orders = torch.div(_, 100, rounding_mode='trunc')
            ica_orders-=1
            x = x.to(DEVICE)
            x_hat1, message, z, receiver_op = model(x)
            # Expand dimensions for advanced indexing
            ica_orders_expanded = ica_orders.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # Gather along the second dimension using ica_orders
            x_hat = torch.gather(x_hat1, 1, ica_orders_expanded.expand(-1, -1, 3, 45, 45).to(DEVICE)).squeeze(1)

            recon_loss = mse_loss(x_hat, x)



            if batch_idx % print_step == 0:
                val_losses.append(recon_loss.item())
                print("epoch:", epoch + 1, "  step:", batch_idx + 1, " recon_loss:", recon_loss.item())
                # print("step: {}  recon_loss: {:.8f}  diversity penalty: {}  total diversity: {:.8f}  total loss: {:.8f}".format(
                #     batch_idx + 1,
                #     recon_loss.item(),
                #     [round(val, 8) for val in diversity_penalty],  # This will round each value in the list to 8 decimal places
                #     sum(diversity_penalty),
                #     loss.item()
                # ))

    model.train()  # switch model back to the train mode

print("Finish!!")



#Better way to save and load:
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'losses': losses,
    'validation_losses':val_losses
}, './checkpoints_epochs_1_ICA_all_sent5_progressive.pth')