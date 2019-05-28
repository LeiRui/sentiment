# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:36:32 2019

@author: admin
"""
import numpy as np

test_outputs = np.load("test_outputs.npy",allow_pickle=True).tolist()
test_id = np.load("test_id.npy").tolist()
print("load success")
    
import pandas as pd
df = pd.DataFrame(zip(test_id,test_outputs), columns=["ID","Pred"])
df.to_csv('result.csv', index=False)