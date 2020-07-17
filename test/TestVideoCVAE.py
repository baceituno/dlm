import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import *
from src.trainers import *
from src.datatools import *

import torch
import pdb
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

import time

print("loading training data...")
# loads the training data
data, vids, pols = load_dataset(1,1) 
N_data = np.shape(data)[0]
print("parsing training data...")
inputs_1, inputs_2, inputs_img, _, labels = parse_dataVids(data)

# define network
print("Setting up network...")
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
net = ContactNet(N_data).to(device)
net.addFrameCVAELayers()
net.addVideoLayers()

net.load(name = "cnn_1_model.pt")
net.eval()

# print("training video cod. autoencoders")
# TrainVideoCVAE(net, vids, epochs = 100)
# TrainVideoJointParams(net, vids, inputs_2, epochs = 5000)
# TrainVideoDecoders(net, vids, inputs_1, inputs_img, epochs = 5000)
# net.save(name = "cnn_1_model.pt")
# TrainVideoParams(net, vids, inputs_2, epochs = 500)
# TrainVideo2V(net, vids, inputs_2, epochs = 1000)
# net.save()


criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=1e-5)

for epoch in range(200):  # loop over the dataset multiple times
    loss_t = 0
    optimizer.zero_grad()

    outputs = net.forwardVideo(torch.tensor(vids).float())
    loss = criterion(10*outputs, 10*labels.float())
    
    loss_t = loss.item()
    loss.backward()
    optimizer.step()

    print("Train loss at epoch ",epoch," = ",loss_t)

net.save(name = "cnn_1_model.pt")
net.gen_resVid(vids,'trainVid_57')