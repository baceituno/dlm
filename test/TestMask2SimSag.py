import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import *
from src.trainers import *
from src.datatools import *

import time
import torch
import pdb
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt


print("loading training data...")
# loads the training data
data, vids, polygons = load_dataset_sagittal(71,71)
N_data = np.shape(data)[0]
print(N_data)
print("parsing test data...")
inputs_1, inputs_2, inputs_img, _, labels = parse_dataVids(data)

# define network
print("Setting up network...")
net = ContactNet(N_data, sagittal = True)
net.addShapeVAELayers()
net.addMaskDecoderLayers()
net.gamma = 0.1
# net.load(name = "cnn_w_model.pt")
# net.eval()

corr_inputs_1 = inputs_1[:,:15].float().view(-1,3,5)
corr_inputs_1[:,2,:] = torch.sin(corr_inputs_1[:,2,:])*0.01
corr_inputs_1 = corr_inputs_1.float().view(-1,15)

criterion = torch.nn.MSELoss(reduction='mean')
outputs = net.forward2Sim(inputs_1.float(),inputs_2.float(),inputs_img.float(), torch.tensor(polygons).float(), render=False, bypass = False, pass_soft = True)
loss = criterion(100*outputs.float(), 100*corr_inputs_1[:,:15].float())
loss_t = loss.item()

# print("training video autoencoders")
optimizer = optim.Adam(net.parameters(), lr=1e-3)
TrainMaskJointParams(net, inputs_1, inputs_img, inputs_2, epochs = 500, n_batches = 1, optimizer = optimizer)

net.gen_res_sim(inputs_1, inputs_2, inputs_img, torch.tensor(polygons).float(), name  = 'Sag_GT_71_', pass_ = True)
net.gen_res_sim(inputs_1, inputs_2, inputs_img, torch.tensor(polygons).float(), name  = 'Sag_MIQP_71_', pass_ = False)

print("Gound Truth loss = ",loss_t)

losses_test, losses_val = ([], [])
render = False
optimizer = optim.Adam(net.parameters(), lr=1e-6)
for epoch in range(20):
	loss_t = 0
	optimizer.zero_grad()
	print('\n\n\n\n')
	start = time.time()
	# with torch.autograd.detect_anomaly():
	outputs = net.forward2Sim(inputs_1.float(),inputs_2.float(),inputs_img.float(), torch.tensor(polygons).float(), render=render, bypass = False)

	# print(outputs.float())
	# print(inputs_1[:,:15].float())

	loss = criterion(100*outputs.float(), 100*corr_inputs_1[:,:15].float())
	loss_t = loss.item()
	losses_test.append(loss_t)

	print("Train loss at epoch ",epoch," = ",loss_t)

	end = time.time()
    

	print(end - start)
	start = time.time()
	loss.backward()
	optimizer.step()
	end = time.time()
	print(end - start)

	render = False

	
	# net.gamma *= 1.01
	# optimizer.param_groups[0]['lr'] *= 0.99

# plots training progress
plt.figure(1)
plt.plot(losses_test,color="b")
plt.show()

print(100*outputs.float())
print(100*corr_inputs_1[:,:15].float())

outputs = net.forward2Sim(inputs_1.float(),inputs_2.float(),inputs_img.float(), torch.tensor(polygons).float(), render = True, bypass = False)
net.gen_res_sim(inputs_1, inputs_2, inputs_img, torch.tensor(polygons).float(), name  = 'Sag_sim_71_', pass_ = False)