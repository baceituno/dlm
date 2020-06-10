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
data, _ = load_dataset(1,1)
N_data = np.shape(data)[0]
print("parsing training data...")
inputs_1, inputs_2, inputs_img, labels = parse_data(data)
print(np.shape(data))

print('Loading validation data...')
# validation data
data1, _ = load_dataset(5,5)
N_data1 = np.shape(data1)[0]
print("parsing validation data...")
inputs_11, inputs_21, inputs_img1, labels1 = parse_data(data1)
print(np.shape(data1))

# from PIL import Image
# Xmg = Image.fromarray(np.repeat(np.reshape(inputs_img[0,:],(50,50,1)),3,axis=2).numpy(),'RGB')
# Xmg.show()

# define network
print("Setting up network...")
net = Net(N_data)
net.load()

net.eval()

print("training autoencoder")
TrainShapeVAE(net, inputs_img.float(), epochs = 10)
net.save()

print("training decoders")
TrainDecoders(net, inputs_1, inputs_2, inputs_img, epochs = 10)
net.save()

# training set
print("training planner")
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
losses_test, losses_val = ([], [])

for epoch in range(10):  # loop over the dataset multiple times
    loss_t = 0
    optimizer.zero_grad()
    outputs = net.forward(inputs_1.float(),inputs_2.float(),inputs_img.float())
    loss = criterion(outputs, labels.float())
    loss.backward()
    optimizer.step()
    
    loss_t = loss.item()

    losses_test.append(loss_t)
    print("Train loss at epoch ",epoch," = ",loss_t)

    outputs1 = net.forward(inputs_11.float(),inputs_21.float(),inputs_img1.float())
    loss1 = criterion(outputs1, labels1.float())
    loss_t = loss1.item()

    losses_val.append(loss_t)
    print("Valid. loss at epoch ",epoch," = ",loss_t)

print('saving results...')

net.save()
net.gen_res(inputs_1,inputs_2,inputs_img,'train')
net.gen_res(inputs_11,inputs_21,inputs_img1,'res')

# plots training progress
plt.figure(1)
plt.plot(losses_test,color="b")
plt.plot(losses_val,color="r")
plt.show()