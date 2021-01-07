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

# define network
print("Setting up network...")
net = ContactNet(1)
net.addShapeVAELayers()
net.addMaskDecoderLayers()
net.gamma = 0.01

optimizer = optim.Adam(net.parameters(), lr=1e-3)

for n in [69]:
	print("loading training data...")
	# loads the training data
	data, vids, polygons = load_dataset(n,n)
	N_data = np.shape(data)[0]
	print(N_data)
	print("parsing training data...")
	inputs_1, inputs_2, inputs_img, _, _ = parse_dataVidsOld(data)
	print(np.shape(vids))

	corr_inputs_1 = inputs_1[:,:15].float().view(-1,3,5)
	corr_inputs_1[:,2,:] = torch.sin(corr_inputs_1[:,2,:])*0.01
	inputs_1[:,:15] = corr_inputs_1.float().view(-1,15)

	# print("training video autoencoders")
	TrainMaskJointParams(net, inputs_1, inputs_img, inputs_2, epochs = 2000, n_batches = 1, optimizer = optimizer)

data, vids, polygons = load_dataset(69,69)
N_data = np.shape(data)[0]
print(N_data)
print("parsing training data...")
inputs_1, inputs_2, inputs_img, _, _ = parse_dataVidsOld(data)
print(np.shape(vids))

corr_inputs_1 = inputs_1[:,:15].float().view(-1,3,5)
corr_inputs_1[:,2,:] = torch.sin(corr_inputs_1[:,2,:])*0.01
inputs_1[:,:15] = corr_inputs_1.float().view(-1,15)

losses_test, losses_val = ([], [])

criterion = torch.nn.MSELoss(reduction='mean')
render = False
optimizer = optim.Adam(net.parameters(), lr=1e-4)

outputs = net.forward2Sim(inputs_1.float(),inputs_2.float(),inputs_img.float() , torch.tensor(polygons).float(), render=False, bypass = False, pass_soft = True)
loss = criterion(100*outputs.float(), 100*outputs.float())
loss_t = loss.item()

# new error
corr_inputs_1 = outputs.clone().detach().view(-1,15)

print("MIQP Truth loss on Val. Set. = ",loss_t)

outputs = net.forward2Sim(inputs_1.float(),inputs_2.float(),inputs_img.float(), torch.tensor(polygons).float(), render=False, bypass = False, pass_soft = False)
loss = criterion(100*outputs.float(), 100*corr_inputs_1[:,:15].float())
loss_t = loss.item()

print("DNN + CVX loss on Val. Set.  = ",loss_t)

# net.gen_res_sim(inputs_1, inputs_2, inputs_img, torch.tensor(polygons).float(), name  = 'GT_99_', pass_ = True)
# net.gen_res_sim(inputs_1, inputs_2, inputs_img, torch.tensor(polygons).float(), name  = 'MIQP_99_', pass_ = False)

val_data = []
val_pols = []
val_in1 = []
val_in2 = []
val_img = []

print('packing data')
for n in [1,2,3,4]:
	data1, _, polygons1 = load_dataset(n,n)
	inputs_11, inputs_21, inputs_img1, _, _ = parse_dataVidsOld(data1)
	val_pols.append(polygons1)
	val_in1.append(inputs_11)
	val_in2.append(inputs_21)
	val_img.append(inputs_img1)

print('training')
for epoch in range(50):
	loss_v = 0
	# for n in range(1):
		# loads the training data

		# print("training video autoencoders")
		# _corr_inputs_1 = net.forward2Sim(val_in1[n].float(),val_in2[n].float(),val_img[n].float(), torch.tensor(val_pols[n]).float(), render=render, bypass = False, pass_soft = False)
		# outputs = net.forward2Sim(val_in1[n].float(),val_in2[n].float(),val_img[n].float(), torch.tensor(val_pols[n]).float(), render=render, bypass = False)
		# loss = criterion(100*outputs.float(), 100*_corr_inputs_1[:,:15].float())
		# loss_v += loss.item()/4
	
	print("Val loss at epoch ",epoch," = ",loss_v)
	losses_val.append(loss_t)

	loss_t = 0
	optimizer.zero_grad()
	print('\n\n\n\n')
	start = time.time()
	outputs = net.forward2Sim(inputs_1.float(),inputs_2.float(),inputs_img.float(), torch.tensor(polygons).float(), render=render, bypass = False)

	loss = criterion(100*outputs.float(), 100*corr_inputs_1.float())
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

	# render = False

	
	# net.gamma += (1-0.1)/30
	# optimizer.param_groups[0]['lr'] *= 0.9

print(100*outputs.float())
print(100*corr_inputs_1[:,:15].float())

outputs = net.forward2Sim(inputs_1.float(),inputs_2.float(),inputs_img.float(), torch.tensor(polygons).float(), render = True, bypass = False)

# net.gen_res_sim(inputs_1, inputs_2, inputs_img, torch.tensor(polygons).float(), name  = 'DiffSim_99_', pass_ = False)

# plots training progress
plt.figure(1)
plt.plot(losses_test,color="b")
plt.show()


plt.figure(2)
plt.plot(losses_val,color="r")
plt.show()


print(100*outputs.float())
print(100*corr_inputs_1[:,:15].float())

# pdb.set_trace()