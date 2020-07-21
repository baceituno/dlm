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
data, vids, pols = load_dataset(2,2) 
N_data = np.shape(data)[0]
print("parsing training data...")
inputs_1, inputs_2, inputs_img, _, labels = parse_dataVids(data)

print("loading test data...")
# loads the training data
data_v, vids_v, pols_v = load_dataset(57,57) 
N_data = np.shape(data)[0]
print("parsing test data...")
inputs_1_v, inputs_2_v, inputs_img_v, _, labels_v = parse_dataVids(data_v)

# define network
print("Setting up network...")
use_cuda = torch.cuda.is_available()				   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
net = ContactNet(N_data).to(device)
net.addFrameCVAELayers()
net.addVideoLayers()

# net.load(name = "cnn_2_model.pt")
# net.eval()

corr_inputs_1 = inputs_1[:,:].float().view(-1,3,3,5)
# pdb.set_trace()
# corr_inputs_1 = torch.cat((corr_inputs_1, 0.3*torch.cos(corr_inputs_1[:,:,2,:]).view(-1,3,1,5)), axis = 2)
corr_inputs_1[:,0,2,:] = 0.1*torch.sin(corr_inputs_1[:,0,2,:])
inputs_1 = corr_inputs_1.float().view(N_data,-1)

# print("training video cod. autoencoders")
TrainVideoJointParams(net, vids, inputs_2, epochs = 100)
TrainVideoDecoders(net, vids, inputs_1, inputs_img, epochs = 100)
TrainVideo2V(net, vids, inputs_2, epochs = 200)
net.save(name = "cnn_2_model.pt")


criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=1e-6)
n_batches = 20
losses_train = []
losses_test = []
for epoch in range(40):  # loop over the dataset multiple times

	for batch in range(n_batches):
		idx0 = batch*N_data//n_batches
		idx2 = (batch+1)*N_data//n_batches
		loss_t = 0
		optimizer.zero_grad()

		outputs = net.forwardVideo(torch.tensor(vids[idx0:idx2,:]).float(), inputs_1[idx0:idx2,:].float(), inputs_2[idx0:idx2,:].float())
		loss = criterion(30*outputs, 30*labels[idx0:idx2,:].float())
		
		loss_t = loss.item()
		loss.backward()
		optimizer.step()

	outputs = net.forwardVideo(torch.tensor(vids).float(), inputs_1.float(), inputs_2.float())
	loss = criterion(30*outputs, 30*labels.float())
	loss_t = loss.item()
	print("Train loss at epoch ",epoch," = ",loss_t)
	losses_train.append(loss_t)

	outputs = net.forwardVideo(torch.tensor(vids_v).float(), inputs_1_v.float(), inputs_2_v.float())
	loss = criterion(30*outputs, 30*labels_v.float())
	loss_t = loss.item()
	print("Validation loss at epoch ",epoch," = ",loss_t)
	losses_test.append(loss_t)

net.save(name = "cnn_2_model.pt")
net.gen_resVid(vids, inputs_1, inputs_2, 'trainVid_2')

fig, ax = plt.subplots()
ax.plot(losses_test, '--r', label='Test')
ax.plot(losses_train,'b', label='Train')
ax.axis('equal')
leg = ax.legend()
plt.show()