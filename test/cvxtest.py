import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from cvxpylayers.torch import CvxpyLayer

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		# encoder layers
		self.encoder = torch.nn.Sequential()
		self.encoder = torch.nn.Sequential()
		self.encoder.add_module("fc1", torch.nn.Linear(50, 35))
		self.encoder.add_module("relu_1", torch.nn.ReLU())
		self.encoder.add_module("dropout_1", torch.nn.Dropout())
		self.encoder.add_module("fc2", torch.nn.Linear(35, 25))
		self.encoder.add_module("relu_2", torch.nn.ReLU())
		self.encoder.add_module("fc3", torch.nn.Linear(25, 15))
		self.encoder.add_module("relu_3", torch.nn.ReLU())
		self.encoder.add_module("fc4", torch.nn.Linear(15, 9))

		# parameters
		self.N_c = 2 
		self.N_t = 5
		self.N_v = 10
		self.g = 9.8

		# sets up the problem
		self.cvxpylayer = self.setup_cvx()

	def setup_cvx(self):
		# decision variables
		f = cp.Variable((2,self.N_c,self.N_t)) # applied force
		fx = cp.Variable((2,self.N_v,self.N_t)) # external force

		A = cp.Parameter((6, 1)) # trajectory
		b = cp.Parameter((3, 1)) # vertices

		# adds constraints
		constraints = []
		for t in range(self.N_t):
			# linear quasi-dynamics
			constraints.append(sum_entries(f[1,:,t]) + sum_entries(fx[1,:,t]) == 1)
			constraints.append(sum_entries(f[2,:,t]) + sum_entries(fx[2,:,t]) - self.g == 0)

		objective = cp.Minimize(cp.pnorm(f, p=1))
		problem = cp.Problem(objective, constraints)
		
		return CvxpyLayer(problem, parameters=[A, b], variables=[x])

	def setup_cvx_real(self):
		# decision variables
		p = cp.Variable((2,self.N_c,self.N_t)) # contact location
		f = cp.Variable((2,self.N_c,self.N_t)) # applied force
		fx = cp.Variable((2,self.N_v,self.N_t)) # external force
		
		# defines weights
		alpha = cp.Variable((2,self.N_c,self.N_t)) # friction cone weight
		gamma = cp.Variable((2,self.N_v,self.N_t)) # external cone weight
		lmbda = cp.Variable((2,self.N_c,self.N_t)) # position assignment

		# inputs to layer
		r = cp.Parameter((3, self.N_t)) # trajectory
		v = cp.Parameter((3, self.N_t)) # vertices
		ddr = cp.Parameter((3, self.N_t)) # momentum
		p_ref = cp.Parameter((2,self.N_c,self.N_t)) # contact location
		fc = cp.Parameter((2,2,self.N_c,self.N_t)) # friction cone ray
		fcx = cp.Parameter((2,2,self.N_v,self.N_t)) # external cone ray
		
		# adds constraints
		constraints = []
		for t in range(self.N_t):
			# linear quasi-dynamics
			constraints.append(sum_entries(f[1,:,t]) + sum_entries(fx[1,:,t]) == ddr[1,t])
			constraints.append(sum_entries(f[2,:,t]) + sum_entries(fx[2,:,t]) - self.g == ddr[2,t])
			
			# rotational quasi-dynamics
			constraints.append(sum_entries(f[2,:,t]) + sum_entries(fx[2,:,t]) == ddr[3,t])

			# friction cone and facet assignment
			for c in range(self.N_c):
				constraints.append(f[1,c,t] == alpha[1,c,t]*fc[1,1,c,t] + alpha[2,c,t]*fc[1,2,c,t])
				constraints.append(f[2,c,t] == alpha[1,c,t]*fc[2,1,c,t] + alpha[2,c,t]*fc[2,2,c,t])  

			# friction cone assignmnt
			for v in range(self.N_v):
				constraints.append(fx[1,v,t] == gamma[1,v,t]*fcx[1,1,v,t] + gamma[2,v,t]*fcx[1,2,v,t])
				constraints.append(fx[2,v,t] == gamma[1,v,t]*fcx[2,1,v,t] + gamma[2,v,t]*fcx[2,2,v,t])
			

		objective = cp.Minimize(cp.pnorm(f, p=1))
		problem = cp.Problem(objective, constraints)
		
		return CvxpyLayer(problem, parameters=[r,v,ddr,p_ref,fc,fcx], variables=[x])

	def forward(self, x):
		# performs encoding of image
		out = self.encoder.forward(x)
		out.view(-1)
		A = out[:6]
		b = out[6:]

		# concatenates
		y, = self.cvxpylayer(A.view(6,1), b.view(3,1))

		return y

net = Net()

# solve the problem
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.001)

X = torch.randn(100,50)
y = torch.abs(torch.randn(100,2))

n_epochs = 100
lost = np.zeros((n_epochs))

for epoch in range(n_epochs):  # loop over the dataset multiple times	  
	# zero the parameter gradients

	sums = 0

	for i in range(100):
		optimizer.zero_grad()

		# batch points
		# forward + backward + optimize
		outputs = net(X[i,:])
		loss = criterion(outputs, y[i,:])
		sums += loss.item()
		loss.backward()
		optimizer.step()
	lost[epoch] = sums

	print("loss at epoch",epoch,"=",sums)
	# lost[epoch] = loss.to_numpy()
	# forward + backward + optimize
	# print statistics

plt.figure(1)
plt.plot(lost,color='k')
plt.show()