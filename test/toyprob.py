import torch
import pdb
from numpy import loadtxt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class Net(nn.Module):
    def __init__(self, N_data):
        super(Net, self).__init__()
        # contact-trajectory optimizer 
        self.N_c = 2
        self.T = 5
        self.dt = 0.1
        self.CTOlayer = self.setup_cto()
        self.Difflayer = self.setup_diff()

        # shape encoder
        self.encoder = torch.nn.Sequential()
        self.encoder.add_module("conv_1", torch.nn.Conv2d(1, 10, kernel_size=11))
        self.encoder.add_module("pool_1", torch.nn.MaxPool2d(kernel_size=2))
        self.encoder.add_module("relu_1", torch.nn.ReLU())
        self.encoder.add_module("conv_2", torch.nn.Conv2d(10, 20, kernel_size=5))
        self.encoder.add_module("pool_2", torch.nn.MaxPool2d(kernel_size=4))
        self.encoder.add_module("relu_2", torch.nn.ReLU())
        self.encoder.add_module("flatten", torch.nn.Flatten())
        self.encoder.add_module("fc3", torch.nn.Linear(320, 150))
        self.encoder.add_module("dropout_3", torch.nn.Dropout(0.3))
        self.encoder.add_module("relu_3", torch.nn.ReLU())
        self.encoder.add_module("fc4", torch.nn.Linear(150, 100))
        self.encoder.add_module("relu_4", torch.nn.ReLU())

        # p_r decoder
        self.p_dec = torch.nn.Sequential()
        self.p_dec.add_module("fc5", torch.nn.Linear(145, 100))
        self.p_dec.add_module("dropout_5", torch.nn.Dropout(0.3))
        self.p_dec.add_module("relu_5", torch.nn.ReLU())
        self.p_dec.add_module("fc6", torch.nn.Linear(100, 100))
        self.p_dec.add_module("relu_6", torch.nn.ReLU())
        self.p_dec.add_module("fc60", torch.nn.Linear(100, 20))

        # v decoder
        self.v_dec = torch.nn.Sequential()
        self.v_dec.add_module("fc7", torch.nn.Linear(145, 100))
        self.v_dec.add_module("dropout_7", torch.nn.Dropout(0.3))
        self.v_dec.add_module("relu_7", torch.nn.ReLU())
        self.v_dec.add_module("fc8", torch.nn.Linear(100, 100))
        self.v_dec.add_module("relu_8", torch.nn.ReLU())
        self.v_dec.add_module("fc80", torch.nn.Linear(100, 40))

    def setup_cto(self):
        # decision variables
        p = cp.Variable((2*self.N_c, self.T)) # contact location
        f = cp.Variable((2*self.N_c,self.T)) # forces
        gamma = cp.Variable((2*self.N_c,self.T)) # cone weights
        alpha1 = cp.Variable((self.N_c,self.T)) # vertex weights
        alpha2 = cp.Variable((self.N_c,self.T)) # vertex weights
        f_e = cp.Variable((4,self.T)) # external force
        gamma_e = cp.Variable((4,self.T)) # cone weights

        # input parameters
        r = cp.Parameter((3, self.T)) # trajectory
        ddr = cp.Parameter((3, self.T)) # trajectory
        p_r = cp.Parameter((2*self.N_c, self.T)) # reference contact location
        fc = cp.Parameter((4*self.N_c, self.T)) # friction cone
        p_e = cp.Parameter((4, self.T)) # external contact location
        fc_e = cp.Parameter((8, self.T)) # external friction cone
        v = cp.Parameter((4*self.N_c, self.T)) # facets for each contact

        # adds constraints
        constraints = []
        for t in range(self.T):
            # linear quasi-dynamics
            constraints.append(sum(f[:20,t]) + f_e[0,t] + f_e[1,t] == ddr[0,t])
            constraints.append(sum(f[20:,t]) + f_e[2,t] + f_e[3,t] == ddr[1,t])

            # angular dynamics with McCormick Envelopes
            tau = 0
            for c in range(self.N_c):
                u1 = (p_r[c,t]-r[0,t])
                v1 = f[self.N_c + c,t]

                u2 = (p_r[self.N_c + c,t]-r[1,t])
                v2 = f[c,t]

                tau += u1*v1 - u2*v2

            tau += (p_e[0,t]-r[0,t])*f_e[2,t] - (p_e[2,t]-r[1,t])*f_e[0,t]
            tau += (p_e[1,t]-r[0,t])*f_e[3,t] - (p_e[3,t]-r[1,t])*f_e[1,t]
            constraints.append(tau == ddr[2,t])

            # constraints contacts to their respective facets
            for c in range(self.N_c):
                constraints.append(p[c,t] == alpha1[c,t]*v[c*self.N_c,t] + alpha2[c,t]*v[c*self.N_c+2,t])
                constraints.append(p[c+self.N_c,t] == alpha1[c,t]*v[c*self.N_c+1,t] + alpha2[c,t]*v[c*self.N_c+3,t])
                constraints.append(alpha1[c,t] + alpha2[c,t] == 1)
                constraints.append(alpha1[c,t] >= 0)
                constraints.append(alpha2[c,t] >= 0)
                # if t < 4:
                    # constraints.append(p[c,t] == p_r[c,t])
                    # constraints.append(p[c+self.N_c,t] == p_r[c+self.N_c,t])

            # friction cone constraints
            for c in range(self.N_c):
                constraints.append(gamma[c,t]*fc[c*self.N_c,t] + gamma[c + self.N_c,t]*fc[c*self.N_c + 2,t] == f[c,t])
                constraints.append(gamma[c,t]*fc[c*self.N_c + 1,t] + gamma[c + self.N_c,t]*fc[c*self.N_c + 3,t] == f[self.N_c + c,t])
                constraints.append(gamma[c,t] >= 0)
                constraints.append(gamma[self.N_c + c,t] >= 0)
            
            # external friction cone constratins
            constraints.append(gamma_e[0,t]*fc_e[0,t] + gamma_e[1,t]*fc_e[2,t] == f_e[0,t])
            constraints.append(gamma_e[0,t]*fc_e[1,t] + gamma_e[1,t]*fc_e[3,t] == f_e[2,t])
            constraints.append(gamma_e[0,t] >= 0)
            constraints.append(gamma_e[1,t] >= 0)

            constraints.append(gamma_e[2,t]*fc_e[4,t] + gamma_e[3,t]*fc_e[6,t] == f_e[1,t])
            constraints.append(gamma_e[2,t]*fc_e[5,t] + gamma_e[3,t]*fc_e[7,t] == f_e[3,t])
            constraints.append(gamma_e[2,t] >= 0)
            constraints.append(gamma_e[3,t] >= 0)

        objective = cp.Minimize(cp.pnorm(f, p=1))
        problem = cp.Problem(objective, constraints)
        
        return CvxpyLayer(problem, parameters=[r, ddr, fc, p_e, fc_e, v, p_r], variables=[p, f, f_e, alpha1, alpha2, gamma, gamma_e])

    def setup_diff(self):
        # decision variables
        dr = cp.Variable((3, self.T))
        ddr = cp.Variable((3, self.T))

        # parameters
        r = cp.Parameter((3, self.T))
        
        # adds finite-diff constraints
        constraints = []
        for t in range(self.T):
            if t == 0:
                constraints.append(ddr[0,t]*(self.dt**2) == 0)
                constraints.append(ddr[1,t]*(self.dt**2) == 0)
                constraints.append(ddr[2,t]*(self.dt**2) == 0)

                constraints.append(dr[0,t] == 0)
                constraints.append(dr[1,t] == 0)
                constraints.append(dr[2,t] == 0)
            elif t == self.T-1:
                constraints.append(ddr[0,t] == 0)
                constraints.append(ddr[1,t] == 0)
                constraints.append(ddr[2,t] == 0)

                constraints.append(dr[0,t] == 0)
                constraints.append(dr[1,t] == 0)
                constraints.append(dr[2,t] == 0)
            else:
                constraints.append(ddr[0,t]*(self.dt**2) == r[0,t-1] - 2*r[0,t] + r[0,t+1])
                constraints.append(ddr[1,t]*(self.dt**2) == r[1,t-1] - 2*r[1,t] + r[1,t+1])
                constraints.append(ddr[2,t]*(self.dt**2) == r[2,t-1] - 2*r[2,t] + r[2,t+1])

                constraints.append(dr[0,t]*(self.dt) == r[0,t] - r[0,t-1])
                constraints.append(dr[1,t]*(self.dt) == r[1,t] - r[1,t-1])
                constraints.append(dr[2,t]*(self.dt) == r[2,t] - r[2,t-1])

        objective = cp.Minimize(cp.pnorm(ddr, p=2))
        problem = cp.Problem(objective, constraints)
        
        return CvxpyLayer(problem, parameters=[r], variables=[dr, ddr])

    def forward(self, xtraj, x, x_img): 
        # passes through the optimization problem
        first = True
        # shape encoding
        for i in range(np.shape(x)[0]):
            e_img = self.encoder.forward(np.reshape(x_img[i,:],(1,1,50,50)))
            # params that should be obtained from video
            r = xtraj[i,:15]
            p_e = x[i,60:80]

            # params that can be computed explicity
            dr = xtraj[i,15:30]
            ddr = xtraj[i,30:]

            # params that can be learned from above
            p_r0 = x[i,0:20]
            v0 = x[i,120:160]
            fc = x[i,20:60]
            fc_e = x[i,80:120]

            # dr, ddr = self.Difflayer(r.view(3,5))
            p_r = self.p_dec.forward(torch.cat((e_img, xtraj[i,:].view(1,45)), 1))
            v = self.v_dec.forward(torch.cat((e_img, xtraj[i,:].view(1,45)), 1))

            p, f, _ ,_, _, _, _ = self.CTOlayer(r.view(3,5), ddr.view(3,5), fc.view(8,5), p_e.view(4,5), fc_e.view(8,5), v.view(8, 5), p_r.view(4, 5))

            # autoencoding errors
            dp = p_r - p_r0
            dv = v - v0

            if first:                
                y = 100*torch.cat([p.view(1,-1), f.view(1,-1), 10*dp.view(1,-1), 10*dv.view(1,-1)], axis = 1)
                first = False
            else:
                y_1 = 100*torch.cat([p.view(1,-1), f.view(1,-1), 10*dp.view(1,-1), 10*dv.view(1,-1)], axis = 1)
                y = torch.cat((y, y_1), axis = 0)
        return y

print("loading data...")
# loads the traijing data
data = np.array((loadtxt("../data/data_1_2f_sq.csv", delimiter=',')))

# Dimensions
N_data = np.shape(data)[0]
img_dim = 2500


print(N_data)

# define network
net = Net(N_data)

# Create Tensors to hold inputs and outputs
inputs_1 = torch.tensor(data[:,:45]) # object trajectory
inputs_2 = torch.tensor(data[:,45:205]) # trajectory decoding
inputs_img = torch.tensor(data[:,205:205+img_dim]) # object shape

labels = torch.cat((100*torch.tensor(data[:,205+img_dim:]),torch.tensor(np.zeros((N_data,60)))), axis = 1)

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.002)

# pdb.set_trace()

losses_test = []
losses_val = []

# training set
print("training...")
for epoch in range(1000):  # loop over the dataset multiple times
    loss_t = 0
    optimizer.zero_grad()
    outputs = net.forward(inputs_1.float(),inputs_2.float(),inputs_img.float())
    loss = criterion(outputs, labels.float())
    loss.backward()
    optimizer.step()
    
    loss_t = loss.item()

    losses_test.append(loss_t)
    print("Train loss at epoch ",epoch," = ",loss_t)

print('saving results...')

y = net.forward(torch.tensor(inputs_1).float(),torch.tensor(inputs_2).float(),torch.tensor(inputs_img).float())
y = y.clone().detach()/100

np.savetxt("../data/res_0_2f.csv", y.data.numpy(), delimiter=",")