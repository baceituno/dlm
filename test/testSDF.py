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


print("loading training data...")
# loads the training data
data, _ = load_dataset_block() 
N_data = np.shape(data)[0]
print("parsing training data...")
inputs_1, inputs_2, inputs_img, inputs_sdf, labels = parse_data(data)
print(np.shape(data))


# define network
print("Setting up network...")
net = Net(N_data)
net.eval()

N_data = np.shape(inputs_1)[0]
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.001)

print("training SDF decoder")
for epoch in range(100):  # loop over the dataset multiple times
        loss_t = 0
        # optimizer.zero_grad()
        v = net.forward_v_sdf(inputs_1.float(),inputs_2.float(),inputs_img.float())
        loss = LossShapeSDF().apply(v.float(), inputs_sdf.float(), inputs_1.float())
        # loss.backward()
        # optimizer.step()
        
        loss_t = loss.item()

        print("V SDF decoder loss at epoch ",epoch," = ",loss_t)

# plots training progress
plt.figure(1)
plt.plot(losses_test,color="b")
plt.plot(losses_val,color="r")
plt.show()