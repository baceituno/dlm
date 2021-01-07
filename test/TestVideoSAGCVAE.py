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

# define network
print("Setting up network...")
use_cuda = torch.cuda.is_available()				   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
net = ContactNet(200,sagittal=True).to(device)
net.addFrameCVAELayers()
net.addVideoLayers()
net.addShapeVAELayers()

# loads pre-trained weights
try:
	print('loading model')
	net.load(name = "cnn_y_of_model.pt")
	net.eval()
except:
	pass

###################################################
# Pre-Trains weights for the CNN + LSTM + MLP     #
###################################################

# optimizer = optim.Adam(net.parameters(), lr=1e-3)

# for i in range(1):
# 	for n in [1,2,3,4]:
# 		data, vids, pols = load_dataset_sagittal(n,n)
# 		N_data = np.shape(data)[0]
# 		print("parsing training data...")
# 		print(N_data)
# 		inputs_1, inputs_2, inputs_img, inputs_env, labels = parse_dataVids(data)

# 		corr_inputs_1 = inputs_1[:,:].float().view(-1,3,3,5)
# 		# corr_inputs_1 = torch.cat((corr_inputs_1, 0.3*torch.cos(corr_inputs_1[:,:,2,:]).view(-1,3,1,5)), axis = 2)
# 		corr_inputs_1[:,0,2,:] = 0.1*torch.sin(corr_inputs_1[:,0,2,:])
# 		inputs_1 = corr_inputs_1.float().view(N_data,-1)

# 		# parameter assess
# 		print('training CNN decoders')
# 		TrainVideoDecoders(net, vids, inputs_1, inputs_img, inputs_env, epochs = 2000, n_batches = 5, optimizer = optimizer)
# 		# TrainVideoParams(net, vids, inputs_2, epochs = 500, optimizer = optimizer, plot = True)
# 		TrainVideoJointParams(net, vids, inputs_2, epochs = 2000, n_batches = 5, optimizer = optimizer)
# 		# optimizer.param_groups[0]['lr'] *= 10
# 		TrainVideo2V(net, vids, inputs_2, epochs = 1000, optimizer = optimizer)

# 	net.save(name = "cnn_y_of_model.pt")

###################################################
# Runs the CNN + LSTM + MLP + CVX Layer inference #
###################################################

print("loading test data...")
# loads the training data
data_v, vids_v, pols_v = load_dataset_sagittal(5,5)
print("parsing test data...")
inputs_1_v, inputs_2_v, inputs_img_v, _, labels_v = parse_dataVids(data_v)

optimizer = optim.Adam(net.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss(reduction='mean')
n_batches = 2
losses_train = []
losses_test = []

data = []
vids = []
pols = []

inputs_1 = []
inputs_2 = []
inputs_img = []
labels = []

for d in [1,2,3,4]:
	print("loading training data...")
	data_, vids_, pols_ = load_dataset_sagittal(d,d) 
	print("parsing training data...")
	inputs_1_, inputs_2_, inputs_img_, _, labels_ = parse_dataVids(data_)

	data.append(data_)
	vids.append(vids_)
	pols.append(pols_)

	inputs_1.append(inputs_1_)
	inputs_2.append(inputs_2_)
	inputs_img.append(inputs_img_)
	labels.append(labels_)


for epoch in range(200):
	loss_t = 0
	for d in range(4):
		for batch in range(n_batches):
			N_data = np.shape(data[d])[0]
			idx0 = batch*N_data//n_batches
			idx2 = (batch+1)*N_data//n_batches
			loss_t = 0
			optimizer.zero_grad()

			outputs = net.forwardVideo(torch.tensor(vids[d][idx0:idx2,:]).float(), inputs_1[d][idx0:idx2,:].float(), inputs_2[d][idx0:idx2,:].float(), bypass = False)
			loss = criterion(30*outputs, 30*labels[d][idx0:idx2,:].float())
			# loss = LossObjectFrame(outputs, labels[d][idx0:idx2,:], inputs_1[d][idx0:idx2,:])
			loss_t = loss.item()
			loss.backward()
			optimizer.step()

		outputs = net.forwardVideo(torch.tensor(vids[d]).float(), inputs_1[d].float(), inputs_2[d].float(), bypass = False)
		loss = criterion(30*outputs, 30*labels[d].float())
		loss_t += loss.item()/4

	print("Train loss at epoch ",epoch," = ",loss_t)
	losses_train.append(loss_t)

	outputs = net.forwardVideo(torch.tensor(vids_v).float(), inputs_1_v.float(), inputs_2_v.float(), bypass = False)
	loss = criterion(30*outputs, 30*labels_v.float())
	loss_t = loss.item()
	print("Validation loss at epoch ",epoch," = ",loss_t)
	losses_test.append(loss_t)

	# optimizer.param_groups[0]['lr'] *= 0.95

# net.save(name = "cnn_2_model.pt")

print("loading training data...")
data, vids, pols = load_dataset_sagittal(2,2) 
N_data = np.shape(data)[0]
print("parsing training data...")
inputs_1, inputs_2, inputs_img, _, labels = parse_dataVids(data)
net.gen_resVid(vids, inputs_1, inputs_2, 'trainSagVidSHABBAT_2')

print("loading training data...")
data, vids, pols = load_dataset_sagittal(5,5) 
N_data = np.shape(data)[0]
print("parsing training data...")
inputs_1, inputs_2, inputs_img, _, labels = parse_dataVids(data)
net.gen_resVid(vids, inputs_1, inputs_2, 'trainSagVidSHABBAT_5')

fig, ax = plt.subplots()
ax.plot(losses_test, '--r', label='Test')
ax.plot(losses_train,'b', label='Train')
# ax.axis('equal')
leg = ax.legend()
plt.show()