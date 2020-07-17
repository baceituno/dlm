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


corr_inputs_1 = inputs_1[:,:15].float().view(-1,3,5)
corr_inputs_1[:,2,:] = corr_inputs_1[:,2,:]*0.03
corr_inputs_1 = corr_inputs_1.float().view(-1,15)

losses_test, losses_val = ([], [])
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=1e-6)
for epoch in range(100):  # loop over the dataset multiple times
    loss_t = 0
    optimizer.zero_grad()

    outputs = net.forwardEndToEnd(torch.tensor(vids).float(), torch.tensor(polygons).float(),inputs_1.float(), False)
    loss = criterion(100*outputs.float(), 100*corr_inputs_1[:,:15].float())
    
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