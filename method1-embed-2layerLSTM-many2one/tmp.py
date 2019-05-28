# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:07:46 2019

@author: admin
"""

import numpy as np
train_labels = np.load("train_labels.npy")
train_features = np.load("train_features.npy")
test_features = np.load("test_features.npy")

import json
with open('wordDict.json') as json_file:
    vocab_to_int = json.load(json_file)
    
    