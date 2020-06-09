import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import *
from src.datatools import *
import torch
import pdb
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt


print("loading data...")
# loads training data
data, _ = load_dataset(1,4)
N_data = np.shape(data)[0]
inputs_1, inputs_2, inputs_img, labels = parse_data(data)
print(N_data)

# loads validation data
data1, _ = load_dataset(5,5)
N_data1 = np.shape(data1)[0]
inputs_11, inputs_21, inputs_img1, labels1 = parse_data(data1)


# define network
net = Net(N_data)
# net.load()
# net.eval()

print("training end-to-end")
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
losses_test, losses_val = ([], [])
for epoch in range(200):  # loop over the dataset multiple times
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

# plots training progress
plt.figure(1)
plt.plot(losses_test,color="b")
plt.plot(losses_val,color="r")
plt.show(block=False)
plt.savefig('f1.png')

# resets the network
net = Net(N_data)
# net.load()
# net.eval()

print("training autoencoder")
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(200):  # loop over the dataset multiple times
    loss_t = 0
    optimizer.zero_grad()
    outputs, mu, logvar = net.forwardShapeVAE(inputs_img.float())
    loss = LossShapeVAE(outputs, inputs_img.float(), mu, logvar)
    loss.backward()
    optimizer.step()
    
    loss_t = loss.item()

    print("Autoencoder loss at epoch ",epoch," = ",loss_t)

print("training decoders")
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(200):  # loop over the dataset multiple times
    loss_t = 0
    optimizer.zero_grad()
    dv = net.forward_v(inputs_1.float(),inputs_2.float(),inputs_img.float())
    loss = criterion(dv.float(), torch.tensor(np.zeros((N_data,40))).float())
    loss.backward()
    optimizer.step()
    
    loss_t = loss.item()

    print("Vertex decoder loss at epoch ",epoch," = ",loss_t)

    loss_t = 0
    optimizer.zero_grad()
    dv = net.forward_p(inputs_1.float(),inputs_2.float(),inputs_img.float())
    loss = criterion(dv.float(), torch.tensor(np.zeros((N_data,20))).float())
    loss.backward()
    optimizer.step()
    
    loss_t = loss.item()

    print("Guess decoder loss at epoch ",epoch," = ",loss_t)

# training set
print("training planner and refining")
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
losses_test, losses_val = ([], [])

for epoch in range(200):  # loop over the dataset multiple times
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

# plots training progress
plt.figure(2)
plt.plot(losses_test,color="b")
plt.plot(losses_val,color="r")
plt.show(block=False)
plt.savefig('f2.png')

# resets the network
net = Net(N_data)
# net.load()
# net.eval()

print("training autoencoder")
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(200):  # loop over the dataset multiple times
    loss_t = 0
    optimizer.zero_grad()
    outputs, mu, logvar = net.forward_vae(inputs_img.float())
    loss = LossShapeVAE(outputs, inputs_img.float(), mu, logvar)
    loss.backward()
    optimizer.step()
    
    loss_t = loss.item()

    print("Autoencoder loss at epoch ",epoch," = ",loss_t)

print("training decoders")
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(200):  # loop over the dataset multiple times
    loss_t = 0
    optimizer.zero_grad()
    dv = net.forward_v(inputs_1.float(),inputs_2.float(),inputs_img.float())
    loss = criterion(dv.float(), torch.tensor(np.zeros((N_data,40))).float())
    loss.backward()
    optimizer.step()
    
    loss_t = loss.item()

    print("Vertex decoder loss at epoch ",epoch," = ",loss_t)

    loss_t = 0
    optimizer.zero_grad()
    dv = net.forward_p(inputs_1.float(),inputs_2.float(),inputs_img.float())
    loss = criterion(dv.float(), torch.tensor(np.zeros((N_data,20))).float())
    loss.backward()
    optimizer.step()
    
    loss_t = loss.item()

    print("Guess decoder loss at epoch ",epoch," = ",loss_t)

# training set
print("training planner without refining")
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
losses_test, losses_val = ([], [])

for epoch in range(200):  # loop over the dataset multiple times
    loss_t = 0
    optimizer.zero_grad()
    outputs = net.forward_noenc(inputs_1.float(),inputs_2.float(),inputs_img.float())
    loss = criterion(outputs, labels.float())
    loss.backward()
    optimizer.step()
    
    loss_t = loss.item()

    losses_test.append(loss_t)
    print("Train loss at epoch ",epoch," = ",loss_t)

    outputs1 = net.forward_noenc(inputs_11.float(),inputs_21.float(),inputs_img1.float())
    loss1 = criterion(outputs1, labels1.float())
    loss_t = loss1.item()

    losses_val.append(loss_t)
    print("Valid. loss at epoch ",epoch," = ",loss_t)

# plots training progress
plt.figure(3)
plt.plot(losses_test,color="b")
plt.plot(losses_val,color="r")
plt.show(block=False)
plt.savefig('f3.png')