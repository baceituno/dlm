import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import *
import torch
import pdb
from numpy import loadtxt
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt


print("loading data...")
# loads the traijing data
data = np.array((loadtxt("../data/data_1_2f_sq.csv", delimiter=',')))
for i in range(2,6):
    new_data = np.array((loadtxt("../data/data_"+str(i)+"_2f_sq.csv", delimiter=',')))
    data = np.concatenate((data, new_data), axis=0)

vids = np.array((loadtxt("../data/vids_1_2f_sq.csv", delimiter=',')))
for i in range(2,6):
    new_data = np.array((loadtxt("../data/vids_"+str(i)+"_2f_sq.csv", delimiter=',')))
    data = np.concatenate((vids, new_data), axis=0)

# Dimensions
N_data = np.shape(data)[0]
img_dim = 2500

print(N_data)

# define network
net = Net(N_data)
net.load_state_dict(torch.load("../data/models/cnn_model.pt"))
net.eval()


# Create Tensors to hold inputs and outputs
inputs_1 = torch.tensor(data[:,:45]) # object trajectory
inputs_2 = torch.tensor(data[:,45:205]) # trajectory decoding
inputs_img = torch.tensor(data[:,205:205+img_dim]) # object shape

optimizer = optim.Adam(net.parameters(), lr=0.001)

# pdb.set_trace()

print("training autoencoder")
for epoch in range(100):  # loop over the dataset multiple times
    loss_t = 0
    optimizer.zero_grad()
    outputs, mu, logvar = net.forward_vae(inputs_img.float())
    loss = loss_fn(outputs, inputs_img.float(), mu, logvar)
    loss.backward()
    optimizer.step()
    
    loss_t = loss.item()

    print("Autoencoder loss at epoch ",epoch," = ",loss_t)

torch.save(net.state_dict(),"../data/models/cnn_model.pt")

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.001)

print("training decoders")
for epoch in range(100):  # loop over the dataset multiple times
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

torch.save(net.state_dict(),"../data/models/cnn_model.pt")

# validation data
data1 = np.array((loadtxt("../data/data_6_2f_sq.csv", delimiter=',')))
inputs_11 = torch.tensor(data1[:,:45]) # object trajectory
inputs_21 = torch.tensor(data1[:,45:205]) # trajectory decoding
inputs_img1 = torch.tensor(data1[:,205:205+img_dim]) # object shape
N_data1 = np.shape(data1)[0]

# labels
labels = torch.cat((330*torch.tensor(data[:,205+img_dim:]),torch.tensor(np.zeros((N_data,160)))), axis = 1)
labels1 = torch.cat((330*torch.tensor(data1[:,205+img_dim:]),torch.tensor(np.zeros((N_data1,160)))), axis = 1)

losses_test = []
losses_val = []

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training set
print("training planner")
for epoch in range(100):  # loop over the dataset multiple times
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

y = net.forward(torch.tensor(inputs_1).float(),torch.tensor(inputs_2).float(),torch.tensor(inputs_img).float())
loss = criterion(y, labels.float())
loss_t = loss.item()
print("Train loss = ",loss_t)

torch.save(net.state_dict(), "../data/models/cnn_model.pt")

y = y.clone().detach()/330

np.savetxt("../data/train_0_2f.csv", y.data.numpy(), delimiter=",")

y = net.forward(torch.tensor(inputs_11).float(),torch.tensor(inputs_21).float(),torch.tensor(inputs_img1).float())
loss = criterion(y, labels1.float())
loss_t = loss.item()
print("Validation loss = ",loss_t)

y = y.clone().detach()/330

np.savetxt("../data/res_0_2f.csv", y.data.numpy(), delimiter=",")

# plots training progress
plt.figure(1)
plt.plot(losses_test,color="b")
plt.plot(losses_val,color="r")
plt.show()