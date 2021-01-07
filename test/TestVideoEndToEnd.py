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

# define network
print("Setting up network...")
net = ContactNet(20)
net.addFrameCVAELayers()
net.addVideoLayers()
net.addShapeVAELayers()
net.eval()
net.gamma = 0.1

try:
	net.load(name = "cnn_w_model.pt")
	net.eval()
except:
	pass

optimizer = optim.Adam(net.parameters(), lr= 1e-3)
for n in []:
	data, vids, pols = load_dataset(n,n) 
	N_data = np.shape(data)[0]
	print("parsing training data...")
	inputs_1, inputs_2, inputs_img, _, labels = parse_dataVidsOld(data)

	corr_inputs_1 = inputs_1[:,:].float().view(-1,3,3,5)
	# corr_inputs_1 = torch.cat((corr_inputs_1, 0.3*torch.cos(corr_inputs_1[:,:,2,:]).view(-1,3,1,5)), axis = 2)
	corr_inputs_1[:,0,2,:] = 0.01*torch.sin(corr_inputs_1[:,0,2,:])
	inputs_1 = corr_inputs_1.float().view(N_data,-1)

	print('training CNN decoders')
	TrainVideoDecoders(net, vids, inputs_1, inputs_img, epochs = 1, n_batches = 1, optimizer = optimizer)
	TrainVideoJointParams(net, vids, inputs_2, epochs = 1, n_batches = 1, optimizer = optimizer)
	TrainVideo2V(net, vids, inputs_2, epochs = 1, optimizer = optimizer)

data1, vids1, polygons1 = load_dataset(1,1) 
N_data = np.shape(data1)[0]
print("parsing validation data...")
inputs_11, inputs_21, inputs_img1, _, labels1 = parse_dataVidsOld(data1)

gt_outputs1 = net.forwardEndToEnd(torch.tensor(vids1).float(), torch.tensor(polygons1).float(), inputs_11.float(), render=False, bypass = False, x = inputs_21.float(), pass_soft = True)
outputs = net.forwardEndToEnd(torch.tensor(vids1).float(), torch.tensor(polygons1).float(), inputs_11.float(), render=False, bypass = False)

data, vids, polygons = load_dataset(71,71)
N_data = np.shape(data)[0]
print("parsing training data...")
inputs_1, inputs_2, inputs_img, _, labels = parse_dataVidsOld(data)

gt_outputs = net.forwardEndToEnd(torch.tensor(vids).float(), torch.tensor(polygons).float(), inputs_1.float(), render=False, bypass = False, x = inputs_2.float(), pass_soft = True)
outputs = net.forwardEndToEnd(torch.tensor(vids).float(), torch.tensor(polygons).float(), inputs_1.float(), render=False, bypass = False)

render = False

losses_test = []
losses_val = []

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=1e-5)
for epoch in range(25):  # loop over the dataset multiple times
    loss_t = 0
    optimizer.zero_grad()
    outputs = net.forwardEndToEnd(torch.tensor(vids).float(), torch.tensor(polygons).float(), inputs_1.float(), render=render, bypass = False)
    loss = criterion(100*outputs.float(), 100*gt_outputs.float())
    loss_t = loss.item()
    losses_test.append(loss_t)
    print("Train loss at epoch ",epoch," = ",loss_t)
    loss.backward()
    optimizer.step()

    loss_t = 0
    outputs = net.forwardEndToEnd(torch.tensor(vids1).float(), torch.tensor(polygons1).float(), inputs_11.float(), render=render, bypass = False)
    loss = criterion(100*outputs.float(), 100*gt_outputs1.float())
    loss_t = loss.item()
    losses_val.append(loss_t)
    print("Validation loss at epoch ",epoch," = ",loss_t)

plt.figure(1)
plt.plot(losses_test,color="b")
plt.show()

plt.figure(2)
plt.plot(losses_val,color="r")
plt.show()

outputs = net.forwardEndToEnd(torch.tensor(vids).float(), torch.tensor(polygons).float(), inputs_1.float(), render=True, bypass = False)