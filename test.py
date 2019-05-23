# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:46:51 2019

@author: admin
"""
import numpy as np
import torch
net = torch.load("trained_net_6.pkl")
print(net)

test_x = np.load("test_features.npy",allow_pickle=True)
print("load success")

print("Test set: \t\t{}".format(test_x.shape)) #(2712,50)

batch_size = 24

from torch.utils.data import DataLoader
test_loader = DataLoader(test_x, shuffle=False, batch_size=batch_size) 
#NOTE shuffle off!!!! keep id sequence


test_on_gpu=torch.cuda.is_available()
net.eval()
# init hidden state
h = net.init_hidden(batch_size)
test_outputs = []
# iterate over test data
for inputs in test_loader:
     # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    if(test_on_gpu):
        inputs = inputs.cuda()
    
    # get predicted outputs
    output, h = net(inputs, h)

    # convert output probabilities to predicted class (0 or 1)
    # pred = torch.round(output.squeeze())  # 0-1 rounds to the nearest integer 0 or 1
    # torch.Size([batch])
    if(test_on_gpu):
        test_outputs.extend(output.cpu().squeeze().tolist())
    else:
        test_outputs.extend(output.squeeze().tolist())
    
print(test_outputs)

test_outputs_array = np.array(test_outputs)
np.save("test_outputs.npy",test_outputs_array)


