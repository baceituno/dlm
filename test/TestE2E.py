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
data, vids, polygons = load_dataset(0,0)
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

net.load(name = "cnn_1_model.pt")
net.eval()


losses_test, losses_val = ([], [])
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=1e-6)
for epoch in range(50):  # loop over the dataset multiple times
    loss_t = 0
    optimizer.zero_grad()

    outputs = net.forwardEndToEnd(torch.tensor(vids).float(), torch.tensor(polygons).float(),inputs_1.float())
    loss = criterion(10*outputs.float(), 10*torch.cat((inputs_1[:,:15].float(),torch.tensor(np.zeros((N_data,30))).float()), axis=1))
    
    loss_t = loss.item()
    loss.backward()
    optimizer.step()

    losses_test.append(loss_t)
    print("Train loss at epoch ",epoch," = ",loss_t)

plt.figure(1)
plt.plot(losses_test,color="b")
plt.show()

net.forwardEndToEnd(torch.tensor(vids).float(), torch.tensor(polygons).float(),inputs_1.float(), render = True)

# net.save()
# net.gen_resVid(vids,'trainVid_57')