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
from Model import Encoder, Decoder, Sender, Receiver, Receiver_nn, Model


cuda = False
DEVICE = torch.device("cuda" if cuda else "cpu")
print(DEVICE)
batch_size = 256

#Define Hyperparameters
lr = 2e-4
input_dim = 3 #Encoder
hidden_dim = 50
output_dim = 3
sentence_length=10
vocab_size = 100
embed_dim =50

# Load data
print("Loading Data...")
import pickle
with open('./rgb_loader_20_40_60_80_100', 'rb') as handle:
    mlpimages,mlplabels = pickle.load(handle)

mlpimages = mlpimages.view(-1,3,45,45)
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

print("Data Loaded!!")

#Create Model components
encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
decoder = Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim)
sender = Sender(vocab_size, embed_dim, hidden_dim *12 , sentence_length, 1.0, True)
receiver_nn = Receiver_nn(hidden_dim*12)
receiver = Receiver(vocab_size,embed_dim,hidden_dim*12,receiver_nn)

#Create Model
model = Model(Encoder=encoder, Decoder=decoder, Sender = sender, Receiver = receiver).to(DEVICE)
optimizer = Adam(list(model.parameters()), lr=lr)


#Loading Model
print("Loading Model...")
checkpoint = torch.load('checkpoints_epochs_1_ICA_all_sent5_progressive.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
losses = checkpoint['losses']
val_losses = checkpoint['validation_losses']
print("Model Loaded!!")


print("Running inference on entire dataset to get messages and embeddings...")
model.train()
# losses = []
labels = []
messages = []
embeddings = []
model.Sender.training = False
for epoch in range(1):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(loader):
        x = x.to(DEVICE)
        x_hat, message,z,embedding = model(x)
        x = x.unsqueeze(1)
        x = x.expand(-1, sentence_length, -1, -1, -1)
        labels.extend(_)
        messages.extend(message)
        #Ending the loop after 1 batch
        break

model.Sender.training = True
print("Inference Done!!")



import pandas as pd, numpy as np
lookup_100 = set()
lookup_100_new = set()
lookup_80 = set()
lookup_60 = set()
lookup_40 = set()
lookup_20 = set()
# label_df_100_recovery = pd.read_csv("/content/ClusterNames.csv")
label_df_100_recovery_new = pd.read_csv("ClusterNames_100.csv")
label_df_80_recovery = pd.read_csv("ClusterNames_80.csv")
label_df_60_recovery = pd.read_csv("ClusterNames_60.csv")
label_df_40_recovery = pd.read_csv("ClusterNames_40.csv")
label_df_20_recovery = pd.read_csv("ClusterNames_20.csv")
# label_names_100 = list(label_df_100_recovery['Cluster Name 100'])
label_names_100_new = list(label_df_100_recovery_new['Cluster Name 100'])
label_names_80 = list(label_df_80_recovery['Cluster Name 100'])
label_names_60 = list(label_df_60_recovery['Cluster Name 100'])
label_names_40 = list(label_df_40_recovery['Cluster Name 100'])
label_names_20 = list(label_df_20_recovery['Cluster Name 100'])
# label_names_unique_100 = np.array([x + "_ICA100" for x in label_names_100 if x not in lookup_100 and lookup_100.add(x) is None])
label_names_unique_100_new = np.array([x + "_ICA100" for x in label_names_100_new if x not in lookup_100_new and lookup_100_new.add(x) is None])
label_names_unique_80 = np.array([x + "_ICA80" for x in label_names_80 if x not in lookup_80 and lookup_80.add(x) is None])
label_names_unique_60 = np.array([x + "_ICA60" for x in label_names_60 if x not in lookup_60 and lookup_60.add(x) is None])
label_names_unique_40 = np.array([x + "_ICA40" for x in label_names_40 if x not in lookup_40 and lookup_40.add(x) is None])
label_names_unique_20 = np.array([x + "_ICA20" for x in label_names_20 if x not in lookup_20 and lookup_20.add(x) is None])
# Put them all in one list
label_names_unique_all = [label_names_unique_20, label_names_unique_40, label_names_unique_60, label_names_unique_80, label_names_unique_100_new]


newlabels = {
    "DMN": 0, "ATTENTION": 1, "MOTOR": 2, "VISUAL": 3, "EXECUTIVE": 4,
    "SENSORY": 5, "SALIENCE": 6, "AUDITORY": 7, "COGNITIVE": 8, "BASALGANGLIA": 9,
    "LANG": 10, "CEREBELLAR": 11, "HYPOTHALAMUS": 12, "THALAMUS": 13, "OTHERS": 14
}
newnumbers = {value: key for key, value in newlabels.items()}


#Generates Symbolic Table
def generate_messages(messages,labels):
    # Assuming you have defined messages, label_names_unique, and labels arrays
    output_csv = []
    model.Sender.training = False
    import csv
    for z in range(len(labels)):
        message = np.argmax(messages[z].cpu().detach().numpy(),axis=1).astype(int)
        test_label = label_names_unique_all[(labels[z]//200) - 1][labels[z]%200]

        result = np.concatenate((message[:len(message)-1].reshape(1, -1), np.array([test_label]).reshape(1, -1)), axis=1)

        # Convert the numpy array to a list of integers and append to output_csv
        output_csv.append(result.tolist()[0])

    # Write the mixed data types to the CSV file
    with open("Symbolic_Table_sent_5_epoch_1_progressive.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(output_csv)

    model.Sender.training = True
    return output_csv

#Generate Symbolic Table
print("Generating Symbolic Table...")
messages = torch.stack(messages)[:,:,:]
output = generate_messages(messages,labels)
print("Symbolic Table Generated!!")






