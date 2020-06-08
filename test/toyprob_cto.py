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
            constraints.append(sum(f[:2,t]) + f_e[0,t] + f_e[1,t] == ddr[0,t])
            constraints.append(sum(f[2:,t]) + f_e[2,t] + f_e[3,t] == ddr[1,t])

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
        
        # adds constraints
        constraints = []
        for t in range(self.T):
            if t == 0:
                constraints.append(ddr[0,t]*(self.dt**2) == r[0,t+1] - r[0,t])
                constraints.append(dr[0,t]*(2*self.dt) == 0)

                constraints.append(ddr[1,t]*(self.dt**2) == r[1,t+1] - r[1,t])
                constraints.append(dr[1,t]*(2*self.dt) == 0)

                constraints.append(ddr[2,t]*(self.dt**2) == r[2,t+1] - r[2,t])
                constraints.append(dr[2,t]*(2*self.dt) == 0)
            elif t == self.T-1:
                constraints.append(ddr[0,t]*(self.dt**2) == r[0,t-1] - r[0,t])
                constraints.append(dr[0,t]*(2*self.dt) == 0)

                constraints.append(ddr[1,t]*(self.dt**2) == r[1,t-1] - r[1,t])
                constraints.append(dr[1,t]*(2*self.dt) == 0)

                constraints.append(ddr[2,t]*(self.dt**2) == r[2,t-1] - r[2,t])
                constraints.append(dr[2,t]*(2*self.dt) == 0)
            else:
                constraints.append(ddr[0,t]*(self.dt**2) == r[0,t-1] - 2*r[0,t] + r[1,t+1])
                constraints.append(dr[0,t]*(2*self.dt) == r[0,t+1] - r[0,t-1])

                constraints.append(ddr[1,t]*(self.dt**2) == r[1,t-1] - 2*r[1,t] + r[1,t+1])
                constraints.append(dr[1,t]*(2*self.dt) == r[1,t+1] - r[1,t-1])

                constraints.append(ddr[2,t]*(self.dt**2) == r[2,t-1] - 2*r[2,t] + r[2,t+1])
                constraints.append(dr[2,t]*(2*self.dt) == r[2,t+1] - r[2,t-1])

        objective = cp.Minimize(cp.pnorm(ddr, p=2))
        problem = cp.Problem(objective, constraints)
        
        return CvxpyLayer(problem, parameters=[r], variables=[dr, ddr])

    def forward(self, xtraj, x): 
        # passes through the optimization problem
        first = True
        for i in range(np.shape(x)[0]):
            # params that should be obtained from video
            r = xtraj[i,:15]
            p_e = x[i,60:80]

            # params that can be computed explicity
            dr = xtraj[i,15:30]
            ddr = xtraj[i,30:]

            # params that can be learned from above
            p_r = x[i,0:20]
            v = x[i,120:160]
            fc = x[i,20:60]
            fc_e = x[i,80:120]

            p, f, _ ,_, _, _, _ = self.CTOlayer(r.view(3,5), ddr.view(3,5), fc.view(8,5), p_e.view(4,5), fc_e.view(8,5), v.view(8, 5), p_r.view(4, 5))

            if first:
                y = torch.cat((p.view(1,-1), f.view(1,-1)), axis = 1)
                first = False
            else:
                y_1 = torch.cat((p.view(1,-1), f.view(1,-1)), axis = 1)
                y = torch.cat((y, y_1))

        return y

print("loading data...")
# loads the traijing data
data = np.array((loadtxt("../data/data_1_2f_sq.csv", delimiter=',')))

# Dimensions
N_data = np.shape(data)[0]

# define network
net = Net(N_data)

# Create Tensors to hold inputs and outputs
inputs_1 = torch.tensor(data[:,:45])
inputs_2 = torch.tensor(data[:,45:205])
labels = torch.tensor(data[:,205:])

criterion = nn.MSELoss(reduction='mean')
# optimizer = optim.Adam(net.parameters(), lr=0.1)

# pdb.set_trace()

losses_test = []
losses_val = []


# training set
print("training...")
for epoch in range(1):  # loop over the dataset multiple times
    loss_t = 0
    # optimizer.zero_grad()
    outputs = net.forward(inputs_1.float(),inputs_2.float())
    loss = criterion(outputs, labels.float())
    # loss.backward()
    # optimizer.step()
    
    loss_t = loss.item()

    losses_test.append(loss_t)
    print("Train loss at epoch ",epoch," = ",loss_t)

print('saving results...')

y = net.forward(torch.tensor(inputs_1).float(),torch.tensor(inputs_2).float())
y = y.clone().detach()

np.savetxt("../data/res_0_2f.csv", y.data.numpy(), delimiter=",")