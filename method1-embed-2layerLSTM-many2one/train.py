# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:11:03 2019

@author: admin
"""

## Training, Validation, Test
import numpy as np
from SentimentRNN import SentimentRNN

train_features = np.load("train_features.npy",allow_pickle=True)
train_labels = np.load("train_labels.npy")
print("load success")


split_frac = 0.99 #split train to train set and validation set
split_idx = int(len(train_features)*split_frac)
train_x, val_x = train_features[:split_idx], train_features[split_idx:]
train_y, val_y = train_labels[:split_idx], train_labels[split_idx:]


import math
batch_size = 50
# 训练集中舍弃掉一些零头为了可以batch大一些
train_x = train_x[:math.floor(len(train_x)/batch_size)*batch_size]
train_y = train_y[:math.floor(len(train_y)/batch_size)*batch_size]
val_x = val_x[:math.floor(len(val_x)/batch_size)*batch_size]
val_y = val_y[:math.floor(len(val_y)/batch_size)*batch_size]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),  # (5062,50)
      "\nValidation set: \t{}".format(val_x.shape)) #(1266,50)

## DataLoaders and Batching
import torch
from torch.utils.data import TensorDataset, DataLoader


# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

# dataloaders

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size) #注意x和y是一起打乱的
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)


## Sentiment Network with PyTorch
# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

# Instantiate the model w/ hyperparams
import json
with open('wordDict.json') as json_file:
    vocab_to_int = json.load(json_file)

vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 400 # good 其实是相当于把7万多个不同的字嵌入到一个400维的空间
hidden_dim = 256 # LSTM里的h和c的维度应该都是这个
n_layers = 1 

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)

## training
# loss and optimization functions
lr=0.001

import torch.nn as nn
criterion = nn.BCELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

epochs = 15 # 3-4 is approx where I noticed the validation loss stop decreasing

clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    net.cuda()

# train for some number of epochs
for e in range(epochs):
    net.train()
    
    # initialize hidden state
    h = net.init_hidden(batch_size) # hidden shape (num_layers * num_directions, batch, hidden_size)
    # 即(weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

    # batch loop
    train_losses = 0
    num = 0
    for inputs, labels in train_loader:

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h]) # ?

        # zero accumulated gradients
        net.zero_grad() 

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float()) # reduction='mean'
        train_losses+=loss.item()
        num += inputs.size(0)
        
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

    print("Epoch: {}/{}...".format(e+1, epochs),
          "Loss: {:.6f}...".format(train_losses/num))
    torch.save(net, "trained_net_"+str(e)+".pkl")
    
    # Get validation loss
    net.eval()
    val_h = net.init_hidden(batch_size)
    val_losses = 0
    num = 0
    for inputs, labels in valid_loader:
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        val_h = tuple([each.data for each in val_h])
        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()
            
        output, val_h = net(inputs, val_h)
        val_loss = criterion(output.squeeze(), labels.float())
        val_losses+=val_loss.item()
        num += inputs.size(0)
        
    print("Val Loss: {:.6f}".format(val_losses/num))
