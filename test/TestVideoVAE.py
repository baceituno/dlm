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


print("loading training data...")
# loads the training data
data, vids = load_dataset(1,1) 
N_data = np.shape(data)[0]
print("parsing training data...")
inputs_1, _, inputs_img, _, labels = parse_dataVids(data)
print(np.shape(vids))

# define network
print("Setting up network...")
net = ContactNet(N_data)
net.addFrameVAELayers()
net.addVideoLayers()
net.load()
net.eval()

print("training video autoencoder")
TrainVideoVAE(net, vids, epochs = 100)

print("training video decoders")
TrainVideoDecoders(net, vids, inputs_1, inputs_img, epochs = 100)

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(100):  # loop over the dataset multiple times
    loss_t = 0
    optimizer.zero_grad()
    outputs = net.forwardVideo(torch.tensor(vids).float())
    loss = criterion(outputs, labels.float())
    loss_t = loss.item()
    loss.backward()
    optimizer.step()
    print("Train loss at epoch ",epoch," = ",loss_t)

# net.save()