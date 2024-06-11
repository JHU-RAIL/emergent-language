
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
# Define the ELSA Model

# Define the Encoder
class Encoder(nn.Module):

    # Define the layers used in the Encoder
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=(3, 3, 3, 1), stride=2):
        super(Encoder, self).__init__()

        kernel_1, kernel_2, kernel_3, kernel_4 = kernel_size

        self.strided_conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_1, stride, padding=1)
        self.strided_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, stride, padding=1)

        self.residual_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_3, padding=1)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, output_dim, kernel_4, padding=0)

        self.linear_1 = nn.Linear(hidden_dim * 12 * 12, hidden_dim * 12 * 6)
        self.linear_2 = nn.Linear(hidden_dim * 12 * 6, hidden_dim * 12)
    # Define the forward pass
    def forward(self, x):

        x = self.strided_conv_1(x)
        x = self.strided_conv_2(x)

        x = F.relu(x)
        y = self.residual_conv_1(x)
        y = y + x

        x = F.relu(y)
        y = self.residual_conv_2(x)
        y = y + x

        # Flatten the tensor
        y1 = y.view(y.size(0), -1)

        # Apply the linear layers with ReLU activation in between
        y1 = F.relu(self.linear_1(y1))
        y1 = self.linear_2(y1)
        y = y1.view(y.size(0), y.size(1), -1)
        # print(y.shape)

        return y
    
# Define the Sender
class Sender(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        temperature,
        training,
        straight_through=False,
    ):
        super(Sender, self).__init__()
        self.hidden_size = hidden_size
        assert max_len >= 1, "Cannot have a max_len below 1"
        self.max_len = max_len

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.training=training

        self.temperature = temperature

        self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=self.hidden_size)
        # print("LSTM",self.hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x):
        # print("self hidden",self.hidden_size)
        x= x.view(x.shape[0],-1)
        # print("x is",x.shape)
        prev_hidden = x[:]#UNCOMMENT FOR NEW SENDER
        # print(self.cell)
        # print("Sender_forward_x_shape",prev_hidden.shape)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        # print("Sender_forward_et_shape",e_t.shape)
        # print("Sender_forward_prevc_shape",prev_c.shape)


        sequence = []

        for step in range(self.max_len):
            h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            logits = self.hidden_to_output(h_t)
            size = logits.size()
            # print("Training is: ",training)
            if not self.training:
                # print("Turned off!")
                indexes = logits.argmax(dim=-1)
                one_hot = torch.zeros_like(logits).view(-1, size[-1])
                one_hot.scatter_(1, indexes.view(-1, 1), 1)
                one_hot = one_hot.view(*size)
                x= one_hot
            # print("h_t",torch.any(torch.isnan(h_t)))
            # print("e_t",e_t)
            # print("prev_c",prev_c)
            # print("logits",torch.isnan(logits))
            else:
                # print("Turned on!")
                x = RelaxedOneHotCategorical(logits=logits, temperature=self.temperature).rsample()


            prev_hidden = h_t
            e_t = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)

        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        sequence = torch.cat([sequence, eos], dim=1)

        return sequence
    



class Receiver_nn(nn.Module):
    def __init__(self,hidden_size):
        super(Receiver_nn, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(inplace=True)
        )
    def forward(self, signal):
        out =  self.fc_layers(signal)
        return out
    
# Define the Receiver
class Receiver(nn.Module):
    # Define the layers used in the Receiver
    def __init__(self, vocab_size, embed_dim, hidden_size,agent):
        super(Receiver, self).__init__()

        self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)

        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.agent = agent

    # Define the forward pass
    def forward(self, message=None,input=None, aux_input=None):
        outputs = []
        # print("received_message_shape",message.shape)
        emb = self.embedding(message)

        prev_hidden = None
        prev_c = None

        # to get an access to the hidden states, we have to unroll the cell ourselves
        for step in range(message.size(1)):
            e_t = emb[:, step, ...]
            h_t, prev_c = (
                self.cell(e_t, (prev_hidden, prev_c))
                if prev_hidden is not None
                else self.cell(e_t)
            )

            outputs.append(self.agent(h_t))
            prev_hidden = h_t

        outputs = torch.stack(outputs).permute(1, 0, 2)

        # print(outputs[:, -1, :].view(-1,outputs.shape[2]).shape,"lplpl")
        # print("outputs shape",outputs.shape)
        # return outputs[:, -1, :].view(-1,outputs.shape[2]) #Original line for only sendinbg last image
        return outputs #New line to send all images



# Define the Decoder
class Decoder(nn.Module):
    # Define the layers used in the Decoder
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_sizes=(1, 3, 3, 3), stride=2, ):
        #Add new parameter first k images are considered
        super(Decoder, self).__init__()

        kernel_1, kernel_2, kernel_3, kernel_4 = kernel_sizes

        self.linear_1 = nn.Linear(hidden_dim * 12, hidden_dim * 12 * 6)
        self.linear_2 = nn.Linear(hidden_dim * 12 * 6, hidden_dim * 12 * 12)

        self.residual_conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_1, padding=0)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, padding=1)

        self.strided_t_conv_1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_3, stride, padding=1, output_padding=0)
        self.strided_t_conv_2 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_4, stride, padding=1)

    # Define the forward pass
    def forward(self, x):
        batch_size, seq_len, feature_size = x.shape[0], x.shape[1], x.shape[2]
        outputs = []
        # Process each part in the second dimension independently
        for i in range(seq_len): #Change seq_len to ica_order
            x_i = x[:, i, :]
            x_i = x_i.view(batch_size, -1)
            x_i = F.relu(self.linear_1(x_i))
            x_i = self.linear_2(x_i)
            x_i = x_i.view(x_i.size(0), x_i.size(1) // 144, 12, 12)

            y_i = self.residual_conv_1(x_i)
            y_i = y_i + x_i
            x_i = F.relu(y_i)

            y_i = self.residual_conv_2(x_i)
            y_i = y_i + x_i
            y_i = F.relu(y_i)

            y_i = self.strided_t_conv_1(y_i)
            y_i = self.strided_t_conv_2(y_i)

            outputs.append(y_i.unsqueeze(1))

        # Stack the processed parts in the second dimension
        output = torch.cat(outputs, dim=1)

        return output

# Define the Model
class Model(nn.Module):
    # The model consists of the Encoder, Sender, Receiver, and Decoder models defined above
    def __init__(self, Encoder, Decoder, Sender, Receiver):
        super(Model, self).__init__()
        self.encoder = Encoder
        self.decoder = Decoder
        self.Sender = Sender
        self.Receiver = Receiver

    # Define the forward pass, we basically just pass the input through the Encoder, Sender, Receiver, and Decoder
    def forward(self, x):
        z = self.encoder(x)

        message = None
        message = self.Sender(z) #z is hidden state for sender lstm

        receiveroutput = self.Receiver(message=message)
        x_hat = self.decoder(receiveroutput)
        return x_hat, message, z,receiveroutput


