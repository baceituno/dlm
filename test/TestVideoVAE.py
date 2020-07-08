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
print(np.shape(vids))

# define network
print("Setting up network...")
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
net = ContactNet(N_data).to(device)
net.addFrameVAELayers()
net.addVideoLayers()

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    net = torch.nn.DataParallel(net)
    pdb.set_trace()
net.load()
net.eval()
<<<<<<< HEAD

print("training video autoencoders")
TrainVideoDecoders(net, vids, inputs_1, inputs_img, epochs = 1000)
# VizVideoDecoders(net, vids, inputs_1, inputs_img)
net.save()
# TrainVideoParams(net, vids, inputs_2, epochs = 500)
=======


print("training video autoencoders")
TrainVideoDecoders(net, vids, inputs_1, inputs_img, epochs = 1000)
net.save()
# TrainVideoParams(net, vids, inputs_2, epochs = 100)
>>>>>>> 8d5354f7f8a2f06d7f368a181ff69af60ba58af5
# net.save()

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.0001)
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
net.gen_resVid(vids,'trainVid_57')