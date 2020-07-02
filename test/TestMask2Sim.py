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
data, vids, polygons = load_dataset(57,57) 
N_data = np.shape(data)[0]
print("parsing training data...")
inputs_1, inputs_2, inputs_img, _, _ = parse_dataVids(data)
print(np.shape(vids))

# define network
print("Setting up network...")
net = ContactNet(N_data)
# net.addFrameVAELayers()
# net.addVideoLayers()
# net.load()
# net.eval()


# print("training video autoencoders")
# TrainVideoDecoders(net, vids, inputs_1, inputs_img, epochs = 10)
losses_test, losses_val = ([], [])
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(30):
	loss_t = 0
	optimizer.zero_grad()
	print('\n\n\n\n')
	start = time.time()
	
	outputs = net.forward2Sim(inputs_1.float(),inputs_2.float(),inputs_img.float(), torch.tensor(polygons).float())
	loss = criterion(outputs.float(), torch.cat((inputs_1[:,:15].float(),torch.tensor(np.zeros((N_data,24))).float()), axis=1))
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

# pdb.set_trace()