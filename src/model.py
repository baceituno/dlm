import torch
import pdb
from numpy import loadtxt
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class ContactNet(torch.nn.Module):
	def __init__(self, N_data):
		super(ContactNet, self).__init__()
		# contact-trajectory parameters 
		self.N_c = 2
		self.T = 5
		self.dt = 0.1

		# CTO layers
		self.CTOlayer = self.setupCTO()
		self.Difflayer = self.setupDiff()
		self.addShapeVAELayers()
		self.addDecoderLayers()

		self.cto_su = torch.nn.Sequential()
		self.cto_su.add_module("fcx10", torch.nn.Linear(190, 200))
		self.cto_su.add_module("relux_11", torch.nn.ReLU())
		self.cto_su.add_module("fcx12", torch.nn.Linear(200, 200))
		self.cto_su.add_module("rexlu_12", torch.nn.ReLU())
		self.cto_su.add_module("fcx120", torch.nn.Linear(200, 200))
		self.cto_su.add_module("rexlu_120", torch.nn.ReLU())
		self.cto_su.add_module("fcx1200", torch.nn.Linear(200, 200))
		self.cto_su.add_module("relux_1200", torch.nn.ReLU())
		self.cto_su.add_module("fcx12000", torch.nn.Linear(200, 200))
		self.cto_su.add_module("relux_12000", torch.nn.ReLU())
		self.cto_su.add_module("fcx12001", torch.nn.Linear(200, 40))

	def addShapeVAELayers(self):
		# shape encoder
		self.encoder = torch.nn.Sequential()
		self.encoder.add_module("conv_1", torch.nn.Conv2d(1, 10, kernel_size=11))
		self.encoder.add_module("pool_1", torch.nn.MaxPool2d(kernel_size=2))
		self.encoder.add_module("relu_1", torch.nn.ReLU())
		self.encoder.add_module("conv_2", torch.nn.Conv2d(10, 20, kernel_size=5))
		self.encoder.add_module("pool_2", torch.nn.MaxPool2d(kernel_size=4))
		self.encoder.add_module("relu_2", torch.nn.ReLU())
		self.encoder.add_module("flatten", torch.nn.Flatten())
		self.encoder.add_module("fc_enc", torch.nn.Linear(320,320))

		# layers for VAE
		self.fc1 = torch.nn.Linear(320, 100)
		self.fc2 = torch.nn.Linear(320, 100)
		self.fc3 = torch.nn.Linear(100, 320)

		# shape decoders
		self.decoder = torch.nn.Sequential()
		self.decoder.add_module("deconv_2", torch.nn.ConvTranspose2d(20, 10, kernel_size=23))
		self.decoder.add_module("derelu_2", torch.nn.ReLU())
		self.decoder.add_module("deconv_1", torch.nn.ConvTranspose2d(10, 1, kernel_size=25))
		self.decoder.add_module("derelu_1", torch.nn.Sigmoid())

	def addFrameVAELayers(self):
		# frame encoder
		self.vid_enc = torch.nn.Sequential()
		self.vid_enc.add_module("vid_conv_1", torch.nn.Conv2d(3, 30, kernel_size=5))
		self.vid_enc.add_module("vid_pool_1", torch.nn.MaxPool2d(kernel_size=2))
		self.vid_enc.add_module("vid_relu_1", torch.nn.ReLU())
		self.vid_enc.add_module("vid_conv_2", torch.nn.Conv2d(30, 60, kernel_size=5))
		self.vid_enc.add_module("vid_pool_2", torch.nn.MaxPool2d(kernel_size=2))
		self.vid_enc.add_module("vid_relu_2", torch.nn.ReLU())
		self.vid_enc.add_module("vid_conv_3", torch.nn.Conv2d(60, 120, kernel_size=5))
		self.vid_enc.add_module("vid_pool_3", torch.nn.MaxPool2d(kernel_size=2))
		self.vid_enc.add_module("vid_relu_3", torch.nn.ReLU())
		self.vid_enc.add_module("vid_flatten", torch.nn.Flatten())
		self.vid_enc.add_module("vid_fc_enc", torch.nn.Linear(480,480))
		self.vid_enc.add_module("vid_relu_4", torch.nn.ReLU())
		self.vid_enc.add_module("vid_fc_enc1", torch.nn.Linear(480,480))
		self.vid_enc.add_module("vid_relu_5", torch.nn.ReLU())
		self.vid_enc.add_module("vid_fc_enc2", torch.nn.Linear(480,480))

		# layers for frame VAE
		self.vid_fc1 = torch.nn.Linear(480, 320)
		self.vid_fc2 = torch.nn.Linear(480, 320)
		self.vid_fc3 = torch.nn.Linear(320, 480)

		self.vid_dec0 = torch.nn.Sequential()
		self.vid_dec0.add_module("vid_fc_dec", torch.nn.Linear(480,480))
		self.vid_dec0.add_module("vid_relu_dec1", torch.nn.ReLU())
		self.vid_dec0.add_module("vid_fc_dec1", torch.nn.Linear(480,480))
		self.vid_dec0.add_module("vid_relu_dec2", torch.nn.ReLU())
		self.vid_dec0.add_module("vid_fc_dec2", torch.nn.Linear(480,480))
		self.vid_dec0.add_module("vid_relu_dec3", torch.nn.ReLU())

		# frame decoders
		self.vid_dec = torch.nn.Sequential()
		self.vid_dec.add_module("vdeconv_3", torch.nn.ConvTranspose2d(120, 60, kernel_size=12))
		self.vid_dec.add_module("vderelu_3", torch.nn.ReLU())
		self.vid_dec.add_module("vdeconv_2", torch.nn.ConvTranspose2d(60, 30, kernel_size=15))
		self.vid_dec.add_module("vderelu_2", torch.nn.ReLU())
		self.vid_dec.add_module("vdeconv_1", torch.nn.ConvTranspose2d(30, 3, kernel_size=24))
		self.vid_dec.add_module("vderelu_1", torch.nn.Sigmoid())

	def addDecoderLayers(self):
		# p_r decoder
		self.p_dec = torch.nn.Sequential()
		self.p_dec.add_module("fc5", torch.nn.Linear(145, 200))
		# self.p_dec.add_module("dropout_5", torch.nn.Dropout(0.2))
		self.p_dec.add_module("relu_5", torch.nn.ReLU())
		self.p_dec.add_module("fc6", torch.nn.Linear(200, 200))
		self.p_dec.add_module("relu_6", torch.nn.ReLU())
		self.p_dec.add_module("fc61", torch.nn.Linear(200, 200))
		self.p_dec.add_module("relu_61", torch.nn.ReLU())
		self.p_dec.add_module("fc62", torch.nn.Linear(200, 200))
		self.p_dec.add_module("relu_62", torch.nn.ReLU())
		self.p_dec.add_module("fc63", torch.nn.Linear(200, 200))
		self.p_dec.add_module("relu_63", torch.nn.ReLU())
		self.p_dec.add_module("fc60", torch.nn.Linear(200, 20))

		# v decoder
		self.v_dec = torch.nn.Sequential()
		self.v_dec.add_module("fc7", torch.nn.Linear(145, 200))
		# self.v_dec.add_module("dropout_7", torch.nn.Dropout(0.2))
		self.v_dec.add_module("relu_7", torch.nn.ReLU())
		self.v_dec.add_module("fc8", torch.nn.Linear(200, 1000))
		self.v_dec.add_module("relu_70", torch.nn.ReLU())
		self.v_dec.add_module("fc80", torch.nn.Linear(1000, 1000))
		self.v_dec.add_module("relu_700", torch.nn.ReLU())
		self.v_dec.add_module("fc800", torch.nn.Linear(1000, 200))
		# self.v_dec.add_module("bn8", torch.nn.BatchNorm1d(1000))
		self.v_dec.add_module("relu_8", torch.nn.ReLU())
		self.v_dec.add_module("fc9", torch.nn.Linear(200, 200))
		self.v_dec.add_module("relu_9", torch.nn.ReLU())
		self.v_dec.add_module("fc91", torch.nn.Linear(200, 200))
		self.v_dec.add_module("relu_91", torch.nn.ReLU())
		self.v_dec.add_module("fc90", torch.nn.Linear(200, 40))

		# fc decoder
		self.fc_dec = torch.nn.Sequential()
		self.fc_dec.add_module("fc10", torch.nn.Linear(85, 200))
		# self.fc_dec.add_module("dropout_10", torch.nn.Dropout(0.2))
		# self.fc_dec.add_module("bn11", torch.nn.BatchNorm1d(num_features=200))
		self.fc_dec.add_module("relu_11", torch.nn.ReLU())
		self.fc_dec.add_module("fc12", torch.nn.Linear(200, 200))
		self.fc_dec.add_module("relu_12", torch.nn.ReLU())
		self.fc_dec.add_module("fc120", torch.nn.Linear(200, 200))
		self.fc_dec.add_module("relu_120", torch.nn.ReLU())
		self.fc_dec.add_module("fc1200", torch.nn.Linear(200, 200))
		self.fc_dec.add_module("relu_1200", torch.nn.ReLU())
		self.fc_dec.add_module("fc12000", torch.nn.Linear(200, 200))
		self.fc_dec.add_module("relu_12000", torch.nn.ReLU())
		self.fc_dec.add_module("fc12001", torch.nn.Linear(200, 40))

		# p_e decoder
		self.pe_dec = torch.nn.Sequential()
		self.pe_dec.add_module("fc13", torch.nn.Linear(145, 200))
		# self.pe_dec.add_module("dropout_13", torch.nn.Dropout(0.3))
		self.pe_dec.add_module("relu_14", torch.nn.ReLU())
		self.pe_dec.add_module("fc14", torch.nn.Linear(200, 200))
		self.pe_dec.add_module("relu_14", torch.nn.ReLU())
		self.pe_dec.add_module("fc140", torch.nn.Linear(200, 200))
		self.pe_dec.add_module("relu_140", torch.nn.ReLU())
		self.pe_dec.add_module("fc1400", torch.nn.Linear(200, 200))
		# self.pe_dec.add_module("bn14", torch.nn.BatchNorm1d(num_features=200))
		self.pe_dec.add_module("relu_1400", torch.nn.ReLU())
		self.pe_dec.add_module("fc14000", torch.nn.Linear(200, 20))

		# fc_e decoder
		self.fce_dec = torch.nn.Sequential()
		self.fce_dec.add_module("fc16", torch.nn.Linear(65, 100))
		# self.fce_dec.add_module("dropout_16", torch.nn.Dropout(0.3))
		self.fce_dec.add_module("relu_16", torch.nn.ReLU())
		self.fce_dec.add_module("fc17", torch.nn.Linear(100, 200))
		self.fce_dec.add_module("relu_17", torch.nn.ReLU())
		self.fce_dec.add_module("fc170", torch.nn.Linear(200, 200))
		self.fce_dec.add_module("relu_170", torch.nn.ReLU())
		self.fce_dec.add_module("fc1700", torch.nn.Linear(200, 200))
		self.fce_dec.add_module("relu_1700", torch.nn.ReLU())
		self.fce_dec.add_module("fc17000", torch.nn.Linear(200, 100))
		self.fce_dec.add_module("relu_17000", torch.nn.ReLU())
		self.fce_dec.add_module("fc17001", torch.nn.Linear(100, 40))

	def addVideoLayers(self):
		# decodes the pose of the object
		self.traj_dec = torch.nn.Sequential()
		self.traj_dec.add_module("fc_dc1", torch.nn.Linear(320*self.T, 1000*self.T))
		self.traj_dec.add_module("relu_dc_1", torch.nn.ReLU())
		self.traj_dec.add_module("fc_dc2", torch.nn.Linear(1000*self.T, 1000*self.T))
		self.traj_dec.add_module("relu_dc_2", torch.nn.ReLU())
		self.traj_dec.add_module("fc_dc3", torch.nn.Linear(1000*self.T, 1000*self.T))
		self.traj_dec.add_module("relu_dc_3", torch.nn.ReLU())
		self.traj_dec.add_module("fc_dc4", torch.nn.Linear(1000*self.T, 1000*self.T))
		self.traj_dec.add_module("relu_dc_4", torch.nn.ReLU())
		self.traj_dec.add_module("fc_dc5", torch.nn.Linear(1000*self.T, 200*self.T))
		self.traj_dec.add_module("relu_dc_5", torch.nn.ReLU())
		self.traj_dec.add_module("fc_dc6", torch.nn.Linear(200*self.T, 200*self.T))
		self.traj_dec.add_module("relu_dc_6", torch.nn.ReLU())
		self.traj_dec.add_module("fc_dc7", torch.nn.Linear(200*self.T, 9*self.T))

		# decodes the shape of the object
		self.shap_dec = torch.nn.Sequential()
		self.shap_dec.add_module("fc_sc1", torch.nn.Linear(320*self.T, 200*self.T))
		self.shap_dec.add_module("relu_sc_1", torch.nn.ReLU())
		self.shap_dec.add_module("fc_sc2", torch.nn.Linear(200*self.T, 200*self.T))
		self.shap_dec.add_module("relu_sc_2", torch.nn.ReLU())
		self.shap_dec.add_module("fc_sc3", torch.nn.Linear(200*self.T, 1000*self.T))
		self.shap_dec.add_module("relu_sc_3", torch.nn.ReLU())
		self.shap_dec.add_module("fc_sc4", torch.nn.Linear(1000*self.T, 100*self.T))
		self.shap_dec.add_module("relu_sc_4", torch.nn.ReLU())
		self.shap_dec.add_module("fc_sc5", torch.nn.Linear(100*self.T,100))

	def setupCTO(self):
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
			constraints.append(sum(f[:self.N_c,t]) + f_e[0,t] + f_e[1,t] == ddr[0,t])
			constraints.append(sum(f[self.N_c:,t]) + f_e[2,t] + f_e[3,t] == ddr[1,t])

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
				constraints.append(p[c,t] == alpha1[c,t]*v[c*self.N_c*2,t] + alpha2[c,t]*v[c*self.N_c*2 + 2,t])
				constraints.append(p[c+self.N_c,t] == alpha1[c,t]*v[c*self.N_c*2 + 1,t] + alpha2[c,t]*v[c*self.N_c*2 + 3,t])
				constraints.append(alpha1[c,t] + alpha2[c,t] == 1)
				constraints.append(alpha1[c,t] >= 0)
				constraints.append(alpha2[c,t] >= 0)
				# if t < 4:
					# constraints.append(p[c,t] == p_r[c,t])
					# constraints.append(p[c+self.N_c,t] == p_r[c+self.N_c,t])

			# friction cone constraints
			for c in range(self.N_c):
				constraints.append(gamma[c,t]*fc[c*self.N_c*2,t] + gamma[c + self.N_c,t]*fc[c*self.N_c*2 + 2,t] == f[c,t])
				constraints.append(gamma[c,t]*fc[c*self.N_c*2 + 1,t] + gamma[c + self.N_c,t]*fc[c*self.N_c*2 + 3,t] == f[self.N_c + c,t])
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

		objective = cp.Minimize(cp.pnorm(f, p=2))
		problem = cp.Problem(objective, constraints)
		
		return CvxpyLayer(problem, parameters=[r, ddr, fc, p_e, fc_e, v, p_r], variables=[p, f, f_e, alpha1, alpha2, gamma, gamma_e])

	def setupDiff(self):
		# decision variables
		dr = cp.Variable((3, self.T))
		ddr = cp.Variable((3, self.T))

		# parameters
		r = cp.Parameter((3, self.T))
		
		# adds finite-diff constraints
		constraints = []
		for t in range(self.T):
			if t == 0 or t == self.T-1:
				constraints.append(ddr[0,t]*(self.dt**2) == 0)
				constraints.append(ddr[1,t]*(self.dt**2) == 0)
				constraints.append(ddr[2,t]*(self.dt**2) == 0)

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
		e_img = self.forwardShapeEncoder(np.reshape(x_img,(-1,1,50,50)))

		# decodes each trajectory
		for i in range(np.shape(x)[0]):
			# params that should be obtained from video
			r = xtraj[i,:15]
			p_e0 = x[i,60:80] # external contact location
			v0 = x[i,120:160] # contact affordance

			# params that can be computed explicity
			dr = xtraj[i,15:30]
			ddr = xtraj[i,30:]

			# params that can be learned from above
			p_r0 = x[i,0:20]
			fc0 = x[i,20:60]
			fc_e0 = x[i,80:120]

			# learnes the parameters
			# dr, ddr = self.Difflayer(r.view(3,5))
			p_r = self.p_dec.forward(torch.cat((e_img[i,:].view(1,100), xtraj[i,:].view(1,45)), 1))
			v = self.v_dec.forward(torch.cat((e_img[i,:].view(1,100), xtraj[i,:].view(1,45)), 1))
			p_e = self.pe_dec.forward(torch.cat([e_img[i,:].view(1,100), xtraj[i,:].view(1,45)], 1))
			fc = self.fc_dec.forward(torch.cat([xtraj[i,:].view(1,45), v.view(1,40)], 1))
			fc_e = self.fce_dec.forward(torch.cat([p_e.view(1,20), xtraj[i,:].view(1,45)], 1))

			# p_e = p_e0
			# p_r = p_r0
			# v = v0
			# fc = fc0
			# fc_e = fc_e0

			p, f, _, _, _, _, _ = self.CTOlayer(r.view(3,5), ddr.view(3,5), fc.view(8,5), p_e.view(4,5), fc_e.view(8,5), v.view(8, 5), p_r.view(4, 5))

			# p = p_r0

			# autoencoding errors set to zero for now
			dp = (p_r0 - p_r0)
			dv = (v0 - v0)
			dfc = (fc0 - fc0)
			dpe = (p_e0 - p_e0)
			dfce = (fc_e0 - fc_e0)

			# print('\n \n \n')

			# print(ddr.view(3,5))
			# print(f)
			# print(f_e)

			# print('\n \n \n')
			
			if first:				
				y = 330*torch.cat([p.view(1,-1), f.view(1,-1), dp.view(1,-1), dv.view(1,-1), dfc.view(1,-1), dpe.view(1,-1), dfce.view(1,-1)], axis = 1)
				first = False
			else:
				y_1 = 330*torch.cat([p.view(1,-1), f.view(1,-1), dp.view(1,-1), dv.view(1,-1), dfc.view(1,-1), dpe.view(1,-1), dfce.view(1,-1)], axis = 1)
				y = torch.cat((y, y_1), axis = 0)
		return y

	def forwardVideo(self, video): 
		# passes through the optimization problem
		# video encoding		
		first_frame = True
		for t in range(self.T):
			frame = video[:,t*7500:(t+1)*7500]
			e_frame = self.forwardFrameEncoder(frame)
			if first_frame:
				e_vid = e_frame
				first_frame = False
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
		
		# extracts image encoding and object trajectory
		e_img = self.shap_dec(e_vid)
		xtraj = self.traj_dec(e_vid)

		# decodes each trajectory
		for i in range(np.shape(e_frame)[0]):
			# params that should be obtained from video
			r = xtraj[i,:15]
			# params that can be computed explicity
			dr = xtraj[i,15:30]
			ddr = xtraj[i,30:]

			# learnes the parameters
			# dr, ddr = self.Difflayer(r.view(3,5))
			p_r = self.p_dec.forward(torch.cat((e_img[i,:].view(1,100), xtraj[i,:].view(1,45)), 1))
			v = self.v_dec.forward(torch.cat((e_img[i,:].view(1,100), xtraj[i,:].view(1,45)), 1))
			p_e = self.pe_dec.forward(torch.cat([e_img[i,:].view(1,100), xtraj[i,:].view(1,45)], 1))
			fc = self.fc_dec.forward(torch.cat([xtraj[i,:].view(1,45), v.view(1,40)], 1))
			fc_e = self.fce_dec.forward(torch.cat([p_e.view(1,20), xtraj[i,:].view(1,45)], 1))

			# solves for the contact trajectory
			p, f, _, _, _, _, _ = self.CTOlayer(r.view(3,5), ddr.view(3,5), fc.view(8,5), p_e.view(4,5), fc_e.view(8,5), v.view(8, 5), p_r.view(4, 5))

			
			if first:				
				y = 330*torch.cat([p.view(1,-1), f.view(1,-1)], axis = 1)
				first = False
			else:
				y_1 = 330*torch.cat([p.view(1,-1), f.view(1,-1)], axis = 1)
				y = torch.cat((y, y_1), axis = 0)
		return y
	
	def forwardVideotoImage(self,video):
		# passes through all frames
		first_frame = True
		for t in range(self.T):
			frame = video[:,t*7500:(t+1)*7500]
			e_frame = self.forwardFrameEncoder(frame)
			if first_frame:
				e_vid = e_frame
				first_frame = False
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
		
		# extracts image encoding
		e_img = self.shap_dec(e_vid)

		# decodes image
		return self.decoder(self.fc3(e_img).view(-1,20,4,4))

	def forwardVideotoTraj(self,video):
		# passes through all frames
		first_frame = True
		for t in range(self.T):
			frame = video[:,t*7500:(t+1)*7500]
			e_frame = self.forwardFrameEncoder(frame)
			if first_frame:
				e_vid = e_frame
				first_frame = False
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
		
		# extracts object trajecotry
		return self.traj_dec(e_vid)
		
	def forward_noenc(self, xtraj, x, x_img):

		e_img = self.forwardShapeEncoder(np.reshape(x_img,(-1,1,50,50)))
		e_img = e_img.detach()
		first = True

		# decodes each trajectory
		for i in range(np.shape(x)[0]):
			# params that should be obtained from video
			r = xtraj[i,:15]
			p_e0 = x[i,60:80] # external contact location
			v0 = x[i,120:160] # contact affordance

			# params that can be computed explicity
			dr = xtraj[i,15:30]
			ddr = xtraj[i,30:]

			# params that can be learned from above
			p_r0 = x[i,0:20]
			fc0 = x[i,20:60]
			fc_e0 = x[i,80:120]

			# learnes the parameters
			# dr, ddr = self.Difflayer(r.view(3,5))
			p_r = self.p_dec.forward(torch.cat((e_img[i,:].view(1,100), xtraj[i,:].view(1,45)), 1))
			v = self.v_dec.forward(torch.cat((e_img[i,:].view(1,100), xtraj[i,:].view(1,45)), 1))
			p_e = self.pe_dec.forward(torch.cat([e_img[i,:].view(1,100), xtraj[i,:].view(1,45)], 1))
			fc = self.fc_dec.forward(torch.cat([xtraj[i,:].view(1,45), v.view(1,40)], 1))
			fc_e = self.fce_dec.forward(torch.cat([p_e.view(1,20), xtraj[i,:].view(1,45)], 1))

			# p_r = p_r0
			# p_e = p_e0
			# v = v0
			fc = fc0
			fc_e = fc_e0

			# detaches the pre-trained
			# p_r.detach()
			# v.detach()

			p, f, _ ,_, _, _, _ = self.CTOlayer(r.view(3,5), ddr.view(3,5), fc.view(8,5), p_e.view(4,5), fc_e.view(8,5), v.view(8, 5), p_r.view(4, 5))

			# autoencoding errors
			# dp = p_r - p_r0
			# dv = v - v0
			# dfc = fc - fc0
			# dpe = p_e - p_e0
			# dfce = fc_e - fc_e0

			p = p_r0.view(4,5)

			dp = (p_r0 - p_r0)
			dv = (v0 - v0)
			dfc = (fc0 - fc0)
			dpe = (p_e0 - p_e0)
			dfce = (fc_e0 - fc_e0)
			
			if first:		
				y = torch.cat([p.view(1,-1), f.view(1,-1), dp.view(1,-1), dv.view(1,-1), dfc.view(1,-1), dpe.view(1,-1), dfce.view(1,-1)], axis = 1)
				first = False
			else:
				y_1 = torch.cat([p.view(1,-1), f.view(1,-1), dp.view(1,-1), dv.view(1,-1), dfc.view(1,-1), dpe.view(1,-1), dfce.view(1,-1)], axis = 1)
				y = torch.cat((y, y_1), axis = 0)
		return y

	def forward_v(self, xtraj, x, x_img):
		# encodes shape
		e_img = self.forwardShapeEncoder(np.reshape(x_img,(-1,1,50,50)))
		first = True
		# decodes each trajectory
		for i in range(np.shape(x)[0]):
			# learnes the vertices parameters
			v = self.v_dec.forward(torch.cat((e_img[i,:].view(1,100), xtraj[i,:].view(1,45)), 1))
			v0 = x[i,120:160]
			dv = v - v0
			
			if first:				
				y = 1000*dv.view(1,-1)
				first = False
			else:
				y_1 = 1000*dv.view(1,-1)
				y = torch.cat((y, y_1), axis = 0)
		return y

	def forward_v_sdf(self, xtraj, x, x_img):
		# encodes shape
		e_img = self.forwardShapeEncoder(np.reshape(x_img,(-1,1,50,50)))
		first = True
		# decodes each trajectory
		for i in range(np.shape(x)[0]):
			# learnes the vertices parameters
			v = self.v_dec.forward(torch.cat((e_img[i,:].view(1,100), xtraj[i,:].view(1,45)), 1))
			v = x[i,120:160] # contact affordance
			if first:				
				y = v.view(1,-1)
				first = False
			else:
				y_1 = v.view(1,-1)
				y = torch.cat((y, y_1), axis = 0)
		return y

	def forward_p(self, xtraj, x, x_img):
		# encodes shape
		e_img = self.forwardShapeEncoder(np.reshape(x_img,(-1,1,50,50)))
		first = True
		# decodes each trajectory
		for i in range(np.shape(x)[0]):
			# learnes the vertices parameters
			p = self.p_dec.forward(torch.cat((e_img[i,:].view(1,100), xtraj[i,:].view(1,45)), 1))
			p_r0 = x[i,0:20]
			dp = p - p_r0
			
			if first:
				y = 1000*dp.view(1,-1)
				first = False
			else:
				y_1 = 1000*dp.view(1,-1)
				y = torch.cat((y, y_1), axis = 0)
		return y

	def forward_fc(self, xtraj, x, x_img):
		# encodes shape
		e_img = self.forwardShapeEncoder(np.reshape(x_img,(-1,1,50,50)))
		first = True
		# decodes each trajectory
		for i in range(np.shape(x)[0]):
			# learnes the vertices parameters
			v = self.v_dec.forward(torch.cat((e_img[i,:].view(1,100), xtraj[i,:].view(1,45)), 1))
			fc = self.fc_dec.forward(torch.cat([xtraj[i,:].view(1,45), v.view(1,40)], 1))
			fc0 = x[i,20:60]
			dp = fc - fc0
			
			if first:				
				y = 100*dp.view(1,-1)
				first = False
			else:
				y_1 = 100*dp.view(1,-1)
				y = torch.cat((y, y_1), axis = 0)
		return y

	def forward_fce(self, xtraj, x, x_img):
		# encodes shape
		e_img = self.forwardShapeEncoder(np.reshape(x_img,(-1,1,50,50)))
		first = True
		# decodes each trajectory
		for i in range(np.shape(x)[0]):
			# learnes the vertices parameters
			p_e = self.p_dec.forward(torch.cat((e_img[i,:].view(1,100), xtraj[i,:].view(1,45)), 1))
			fc = self.fce_dec.forward(torch.cat([p_e.view(1,20), xtraj[i,:].view(1,45)], 1))
			fc0 = x[i,80:120]
			dp = fc - fc0
			
			if first:				
				y = 100*dp.view(1,-1)
				first = False
			else:
				y_1 = 100*dp.view(1,-1)
				y = torch.cat((y, y_1), axis = 0)
		return y

	def forwardShapeVAE(self, x_img):
		# encodes
		h = self.encoder(np.reshape(x_img,(-1,1,50,50)))
		# bottlenecks
		mu, logvar = self.fc1(h), self.fc2(h)
		std = logvar.mul(0.5).exp_()
		esp = torch.randn(*mu.size())
		# reparametrizes
		z = mu + std * esp
		z = self.fc3(z)

		# decodes
		return self.decoder(z.view(-1,20,4,4)), mu, logvar

	def forwardFrameVAE(self, x_vid):
		# encodes
		h = self.vid_enc(np.reshape(x_vid,(-1,3,50,50)))

		# bottlenecks
		mu, logvar = self.vid_fc1(h), self.vid_fc2(h)
		std = logvar.mul(0.5).exp_()
		esp = torch.randn(*mu.size())
		# reparametrizes
		z = mu + std * esp
		z = self.vid_fc3(z)
		z = self.vid_dec0(z)

		# decodes
		h = self.vid_dec(z.view(-1,120,2,2))
		return h, mu, logvar

	def forwardShapeEncoder(self, x_img):
		# encodes
		h = self.encoder(np.reshape(x_img,(-1,1,50,50)))
		# bottlenecks
		mu, logvar = self.fc1(h), self.fc2(h)
		std = logvar.mul(0.5).exp_()
		esp = torch.randn(*mu.size())
		# reparametrizes
		z = mu + std * esp

		# decodes
		return z

	def forwardFrameEncoder(self, frame):
		# encodes
		h = self.vid_enc(np.reshape(frame,(-1,3,50,50)))
		
		# bottlenecks
		mu, logvar = self.vid_fc1(h), self.vid_fc2(h)
		std = logvar.mul(0.5).exp_()
		esp = torch.randn(*mu.size())

		# reparametrizes
		z = mu + std * esp

		# encodes
		return z

	def save(self, name = "cnn_model.pt"):
		torch.save(self.state_dict(),"../data/models/"+str(name))

	def load(self, name = "cnn_model.pt"):
		self.load_state_dict(torch.load("../data/models/"+str(name)),strict = False)

	def gen_res(self,inputs_1,inputs_2,inputs_img,name="res"):
		y = self.forward(torch.tensor(inputs_1).float(),torch.tensor(inputs_2).float(),torch.tensor(inputs_img).float())
		y = y.clone().detach()/330
		np.savetxt("../data/"+name+"_0_2f.csv", y.data.numpy(), delimiter=",")
