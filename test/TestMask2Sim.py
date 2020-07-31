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
data, vids, polygons = load_dataset(69,69) 
N_data = np.shape(data)[0]
print("parsing training data...")
inputs_1, inputs_2, inputs_img, _, _ = parse_dataVids(data)
print(np.shape(vids))

# define network
print("Setting up network...")
net = ContactNet(N_data)
net.addFrameCVAELayers()
net.addVideoLayers()
net.addShapeVAELayers()
net.addDecoderLayers()
# net.load(name = "cnn_x_model.pt")
# net.eval()

corr_inputs_1 = inputs_1[:,:15].float().view(-1,3,5)
corr_inputs_1[:,2,:] = corr_inputs_1[:,2,:]*0.03
corr_inputs_1 = corr_inputs_1.float().view(-1,15)

# print("training video autoencoders")
# TrainVideoDecoders(net, vids, inputs_1, inputs_img, epochs = 10)
losses_test, losses_val = ([], [])
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=1e-6)
for epoch in range(100):
	loss_t = 0
	optimizer.zero_grad()
	print('\n\n\n\n')
	start = time.time()
	# with torch.autograd.detect_anomaly():
	outputs = net.forward2Sim(inputs_1.float(),inputs_2.float(),inputs_img.float(), torch.tensor(polygons).float(), render=False, bypass = False)
	loss = criterion(10*outputs.float(), 10*torch.cat((corr_inputs_1[:,:15].float(),torch.tensor(np.zeros((N_data,24))).float()), axis=1))
	loss_t = loss.item()
	end = time.time()
    
	print(end - start)
	start = time.time()
	loss.backward()
	optimizer.step()
	end = time.time()
	print(end - start)

	print('\n\n\n\n')
	losses_test.append(loss_t)
	print("Train loss at epoch ",epoch," = ",loss_t)

# plots training progress
plt.figure(1)
plt.plot(losses_test,color="b")
plt.show()

outputs = net.forward2Sim(inputs_1.float(),inputs_2.float(),inputs_img.float(), torch.tensor(polygons).float(), True)


# pdb.set_trace()