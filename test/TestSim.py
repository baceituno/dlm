import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import *
from src.trainers import *
from src.datatools import *

import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt

print("Setting up network...")
net = ContactNet(20, sagittal = True)
net.addFrameCVAELayers()
net.addVideoLayers()
net.addShapeVAELayers()
net.eval()
net.gamma = 10

data, vids, polygons = load_dataset_sagittal(99,99)
N_data = np.shape(data)[0]
print("parsing training data...")
inputs_1, inputs_2, inputs_img, _, labels = parse_dataVids(data)
print(np.shape(vids))

# pdb.set_trace()f
gt_outputs, _, err = net.forwardEndToEnd(torch.tensor(vids).float()[0:20,:], torch.tensor(polygons).float()[0:20,:], inputs_1.float()[0:20,:], render=True, bypass = False, x = inputs_2[0:20,:].float(), pass_soft = True)

criterion = torch.nn.MSELoss(reduction='mean')

corr_inputs_1 = inputs_1[:,:].float().view(-1,3,3,5)
# corr_inputs_1 = torch.cat((corr_inputs_1, 0.3*torch.cos(corr_inputs_1[:,:,2,:]).view(-1,3,1,5)), axis = 2)
corr_inputs_1[:,0,2,:] = 0.01*torch.sin(corr_inputs_1[:,0,2,:])
refs = corr_inputs_1[:,0,:,:].view(-1,15)
loss = criterion(100*gt_outputs.float(), 100*refs[0:20,:].float())
loss_t = loss.item()
print("GT loss at epoch ",0," = ",loss_t)
print("GT error at epoch ",0," = ",torch.sum(err.float()**2))

pdb.set_trace()