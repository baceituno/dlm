import torch
import pdb
from numpy import loadtxt
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import pygame
from lcp_physics.physics.bodies import Circle, Rect, Hull
from lcp_physics.physics.constraints import Joint, TotalConstraint
from lcp_physics.physics.constraints import FixedJoint
from lcp_physics.physics.forces import Gravity, MDP, FingerTrajectory
from lcp_physics.physics.world import World, run_world_traj, Trajectory


class ContactNet(torch.nn.Module):
	def __init__(self, N_data):
		super(ContactNet, self).__init__()
		# contact-trajectory parameters 
		self.N_c = 2
		self.T = 5
		self.dt = 0.1
		self.tol = 1
		self.rad = 10
		self.shape_dim = 150
		self.frame_dim = 300
		self.rnn_dim = 400

		self.w = 0.01

		# CTO layers
		self.CTOlayer = self.setupCTO(w_err = self.w)
		self.Difflayer = self.setupDiff(dims=4, T = 6)
		self.Difflayer3 = self.setupDiff(dims=3, T = 5)

		self.cto_su1 = torch.nn.Sequential()
		self.cto_su1.add_module("fcx10", torch.nn.Linear(190, 200))
		self.cto_su1.add_module("relux_11", torch.nn.ReLU())
		self.cto_su1.add_module("fcx12", torch.nn.Linear(200, 200))
		self.cto_su1.add_module("rexlu_12", torch.nn.ReLU())
		self.cto_su1.add_module("fcx120", torch.nn.Linear(200, 200))
		self.cto_su1.add_module("rexlu_120", torch.nn.ReLU())
		self.cto_su1.add_module("fcx1200", torch.nn.Linear(200, 200))
		self.cto_su1.add_module("relux_1200", torch.nn.ReLU())
		self.cto_su1.add_module("fcx12000", torch.nn.Linear(200, 200))
		self.cto_su1.add_module("relux_12000", torch.nn.ReLU())
		self.cto_su1.add_module("fcx12001", torch.nn.Linear(200, 20))


		self.cto_su2 = torch.nn.Sequential()
		self.cto_su2.add_module("fcx20", torch.nn.Linear(190, 200))
		self.cto_su2.add_module("relux_21", torch.nn.ReLU())
		self.cto_su2.add_module("fcx22", torch.nn.Linear(200, 200))
		self.cto_su2.add_module("rexlu_22", torch.nn.ReLU())
		self.cto_su2.add_module("fcx220", torch.nn.Linear(200, 200))
		self.cto_su2.add_module("rexlu_220", torch.nn.ReLU())
		self.cto_su2.add_module("fcx2200", torch.nn.Linear(200, 200))
		self.cto_su2.add_module("rexlu_2200", torch.nn.ReLU())
		self.cto_su2.add_module("fcx22000", torch.nn.Linear(200, 200))
		self.cto_su2.add_module("rexlu_22000", torch.nn.ReLU())
		self.cto_su2.add_module("fcx22001", torch.nn.Linear(200, 20))

		self.lstm_su = torch.nn.Sequential()
		self.lstm_su.add_module("lstm_fcx11", torch.nn.Linear(self.frame_dim*self.T, self.rnn_dim))
		self.lstm_su.add_module("lstm_relux_11", torch.nn.ReLU())
		self.lstm_su.add_module("lstm_fcx12", torch.nn.Linear(self.rnn_dim, self.rnn_dim))

	#########################	
	# Deep Learning Methods #
	#########################

	def addShapeVAELayers(self):
		# shape encoder
		self.encoder = torch.nn.Sequential()
		self.encoder.add_module("conv_1", torch.nn.Conv2d(1, 5, 4, 2, 1))
		self.encoder.add_module("bn_1", torch.nn.BatchNorm2d(5))
		self.encoder.add_module("relu_1", torch.nn.LeakyReLU(0.2))
		self.encoder.add_module("conv_2", torch.nn.Conv2d(5, 10, 4, 2, 1))
		self.encoder.add_module("bn_2", torch.nn.BatchNorm2d(10))
		self.encoder.add_module("relu_2", torch.nn.LeakyReLU(0.2))
		self.encoder.add_module("conv_3", torch.nn.Conv2d(10, 20, 4, 2, 1))
		self.encoder.add_module("bn_3", torch.nn.BatchNorm2d(20))
		self.encoder.add_module("relu_3", torch.nn.LeakyReLU(0.2))
		self.encoder.add_module("flatten", torch.nn.Flatten())
		self.encoder.add_module("linearxxx", torch.nn.Linear(720,720))

		# layers for VAE
		self.fc1 = torch.nn.Linear(720, self.shape_dim)
		self.fc2 = torch.nn.Linear(720, self.shape_dim)
		self.fc3 = torch.nn.Linear(self.shape_dim, 720)

		# shape decoders
		self.decoder = torch.nn.Sequential()
		self.decoder.add_module("deconvups_3", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.decoder.add_module("derepups_3", torch.nn.ReplicationPad2d(1))
		self.decoder.add_module("deconv_3", torch.nn.Conv2d(20, 10, 2, 1))	
		self.decoder.add_module("debn_3", torch.nn.BatchNorm2d(10, 1.e-3))
		self.decoder.add_module("derelu_3", torch.nn.LeakyReLU(0.2))
		self.decoder.add_module("deconvups_2", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.decoder.add_module("derepups_2", torch.nn.ReplicationPad2d(1))
		self.decoder.add_module("deconv_2", torch.nn.Conv2d(10, 5, 3, 1))	
		self.decoder.add_module("debn_2", torch.nn.BatchNorm2d(5, 1.e-3))
		self.decoder.add_module("derelu_2", torch.nn.LeakyReLU(0.2))
		self.decoder.add_module("deconvups_1", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.decoder.add_module("derepups_1", torch.nn.ReplicationPad2d(1))
		self.decoder.add_module("deconv_1", torch.nn.Conv2d(5, 1, 5, 1))
		self.decoder.add_module("derelu_1", torch.nn.Sigmoid())

	def addFrameVAELayers(self):
		# frame encoder
		self.vid_enc = torch.nn.Sequential()
		self.vid_enc.add_module("vid_conv_1", torch.nn.Conv2d(3, 6, 4, 2, 1))
		self.vid_enc.add_module("vid_bn_1", torch.nn.BatchNorm2d(6))
		self.vid_enc.add_module("vid_relu_1", torch.nn.LeakyReLU(0.2))
		self.vid_enc.add_module("vid_conv_2", torch.nn.Conv2d(6, 12, 4, 2, 1))
		self.vid_enc.add_module("vid_bn_2", torch.nn.BatchNorm2d(12))
		self.vid_enc.add_module("vid_relu_2", torch.nn.LeakyReLU(0.2))
		self.vid_enc.add_module("vid_conv_3", torch.nn.Conv2d(12, 24, 4, 2, 1))
		self.vid_enc.add_module("vid_bn_3", torch.nn.BatchNorm2d(24))
		self.vid_enc.add_module("vid_relu_3", torch.nn.LeakyReLU(0.2))
		self.vid_enc.add_module("vid_flatten", torch.nn.Flatten())
		self.vid_enc.add_module("vid_fc_enc", torch.nn.Linear(3456,2*self.frame_dim))
		self.vid_enc.add_module("vid_relu_6", torch.nn.LeakyReLU(0.2))

		# layers for frame VAE
		self.vid_fc1 = torch.nn.Linear(2*self.frame_dim, self.frame_dim)
		self.vid_fc2 = torch.nn.Linear(2*self.frame_dim, self.frame_dim)
		self.vid_fc3 = torch.nn.Linear(self.frame_dim, 2*self.frame_dim)

		self.vid_dec0 = torch.nn.Sequential()
		self.vid_dec0.add_module("vid_relu_dec6", torch.nn.LeakyReLU(0.2))
		self.vid_dec0.add_module("vid_fc_dec4", torch.nn.Linear(2*self.frame_dim,3456))
		self.vid_dec0.add_module("vid_relu_dec3", torch.nn.LeakyReLU(0.2))

		# frame decoders
		self.vid_dec = torch.nn.Sequential()
		self.vid_dec.add_module("vdeconvups_3", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.vid_dec.add_module("vderepups_3", torch.nn.ReplicationPad2d(1))
		self.vid_dec.add_module("vdeconv_3", torch.nn.Conv2d(24, 12, 2, 1))
		self.vid_dec.add_module("vdebn_3", torch.nn.BatchNorm2d(12, 1.e-3))
		self.vid_dec.add_module("vderelu_3", torch.nn.LeakyReLU(0.2))
		self.vid_dec.add_module("vdeconvups_2", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.vid_dec.add_module("vderepups_2", torch.nn.ReplicationPad2d(1))
		self.vid_dec.add_module("vdeconv_2", torch.nn.Conv2d(12, 6, 2, 1))	
		self.vid_dec.add_module("vdebn_2", torch.nn.BatchNorm2d(6, 1.e-3))
		self.vid_dec.add_module("vderelu_2", torch.nn.LeakyReLU(0.2))
		self.vid_dec.add_module("vdeconvups_1", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.vid_dec.add_module("vderepups_1", torch.nn.ReplicationPad2d(1))
		self.vid_dec.add_module("vdeconv_1", torch.nn.Conv2d(6, 3, 5, 1))
		self.vid_dec.add_module("vderelu_1", torch.nn.Sigmoid())

	def addFrameCVAELayers(self):
		# frame encoder
		self.vid_enc = torch.nn.Sequential()
		self.vid_enc.add_module("vid_conv_1", torch.nn.Conv2d(3, 6, 4, 2, 1))
		self.vid_enc.add_module("vid_bn_1", torch.nn.BatchNorm2d(6))
		self.vid_enc.add_module("vid_relu_1", torch.nn.LeakyReLU(0.2))
		self.vid_enc.add_module("vid_conv_2", torch.nn.Conv2d(6, 12, 4, 2, 1))
		self.vid_enc.add_module("vid_bn_2", torch.nn.BatchNorm2d(12))
		self.vid_enc.add_module("vid_relu_2", torch.nn.LeakyReLU(0.2))
		self.vid_enc.add_module("vid_conv_3", torch.nn.Conv2d(12, 24, 4, 2, 1))
		self.vid_enc.add_module("vid_bn_3", torch.nn.BatchNorm2d(24))
		self.vid_enc.add_module("vid_relu_3", torch.nn.LeakyReLU(0.2))
		self.vid_enc.add_module("vid_flatten", torch.nn.Flatten())
		self.vid_enc.add_module("vid_fc_enc", torch.nn.Linear(3456,2*self.frame_dim))
		self.vid_enc.add_module("vid_relu_6", torch.nn.LeakyReLU(0.2))


		self.cvid_enc = torch.nn.Sequential()
		self.cvid_enc.add_module("cvid_fc_dec1", torch.nn.Linear(4*self.frame_dim,2*self.frame_dim))
		self.cvid_enc.add_module("cvid_relu_dec1", torch.nn.LeakyReLU(0.2))
		self.cvid_enc.add_module("cvid_fc_dec2", torch.nn.Linear(2*self.frame_dim,2*self.frame_dim))
		self.cvid_enc.add_module("cvid_relu_dec2", torch.nn.LeakyReLU(0.2))
		self.cvid_enc.add_module("cvid_fc_dec3", torch.nn.Linear(2*self.frame_dim,2*self.frame_dim))
		self.cvid_enc.add_module("cvid_relu_dec3", torch.nn.LeakyReLU(0.2))
		self.cvid_enc.add_module("cvid_fc_dec4", torch.nn.Linear(2*self.frame_dim,2*self.frame_dim))
		self.cvid_enc.add_module("cvid_relu_dec4", torch.nn.LeakyReLU(0.2))

		# layers for frame VAE
		self.vid_fc1 = torch.nn.Linear(2*self.frame_dim, self.frame_dim)
		self.vid_fc2 = torch.nn.Linear(2*self.frame_dim, self.frame_dim)
		self.vid_fc3 = torch.nn.Linear(self.frame_dim, 2*self.frame_dim)

		self.vid_dec0 = torch.nn.Sequential()
		self.vid_dec0.add_module("vid_relu_dec6", torch.nn.LeakyReLU(0.2))
		self.vid_dec0.add_module("vid_fc_dec4", torch.nn.Linear(2*self.frame_dim,3456))
		self.vid_dec0.add_module("vid_relu_dec3", torch.nn.LeakyReLU(0.2))

		# frame decoders
		self.vid_dec = torch.nn.Sequential()
		self.vid_dec.add_module("vdeconvups_3", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.vid_dec.add_module("vderepups_3", torch.nn.ReplicationPad2d(1))
		self.vid_dec.add_module("vdeconv_3", torch.nn.Conv2d(24, 12, 2, 1))
		self.vid_dec.add_module("vdebn_3", torch.nn.BatchNorm2d(12, 1.e-3))
		self.vid_dec.add_module("vderelu_3", torch.nn.LeakyReLU(0.2))
		self.vid_dec.add_module("vdeconvups_2", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.vid_dec.add_module("vderepups_2", torch.nn.ReplicationPad2d(1))
		self.vid_dec.add_module("vdeconv_2", torch.nn.Conv2d(12, 6, 2, 1))	
		self.vid_dec.add_module("vdebn_2", torch.nn.BatchNorm2d(6, 1.e-3))
		self.vid_dec.add_module("vderelu_2", torch.nn.LeakyReLU(0.2))
		self.vid_dec.add_module("vdeconvups_1", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.vid_dec.add_module("vderepups_1", torch.nn.ReplicationPad2d(1))
		self.vid_dec.add_module("vdeconv_1", torch.nn.Conv2d(6, 3, 5, 1))
		self.vid_dec.add_module("vderelu_1", torch.nn.Sigmoid())

	def addDecoderLayers(self):

		# p_r decoder
		self.p_dec = torch.nn.Sequential()
		self.p_dec.add_module("fc5", torch.nn.Linear(self.shape_dim+45, 200))
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
		self.v_dec.add_module("fc7", torch.nn.Linear(self.shape_dim+45, 200))
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
		self.pe_dec.add_module("fc13", torch.nn.Linear(self.shape_dim+45, 200))
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
		# lstm layers
		self.videoRNN = torch.nn.LSTM(input_size=self.frame_dim, hidden_size=self.rnn_dim//2, num_layers=5, bidirectional = True, batch_first=True)
		
		# rnn fully connected layers
		self.fcRNN = torch.nn.Sequential()
		self.fcRNN.add_module("rnn_fc1", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.fcRNN.add_module("rnn_relu", torch.nn.ReLU())
		self.fcRNN.add_module("rnn_fc2", torch.nn.Linear(self.rnn_dim, self.rnn_dim))

		# decodes the pose of the object
		self.traj_dec = torch.nn.Sequential()
		self.traj_dec.add_module("traj_fc_dc1", torch.nn.Linear(self.rnn_dim*self.T, self.T*self.rnn_dim))
		self.traj_dec.add_module("traj_relu_dc_1", torch.nn.ReLU())
		self.traj_dec.add_module("traj_fc_dc2", torch.nn.Linear(self.T*self.rnn_dim, self.T*self.rnn_dim))
		self.traj_dec.add_module("traj_relu_dc_2", torch.nn.ReLU())
		self.traj_dec.add_module("traj_fc_dc3", torch.nn.Linear(self.T*self.rnn_dim, self.T*self.rnn_dim))
		self.traj_dec.add_module("traj_relu_dc_3", torch.nn.ReLU())
		self.traj_dec.add_module("traj_fc_vdc4", torch.nn.Linear(self.T*self.rnn_dim, 5*self.rnn_dim))
		self.traj_dec.add_module("traj_relu_dc_4", torch.nn.ReLU())
		self.traj_dec.add_module("traj_fc_vdc6", torch.nn.Linear(5*self.rnn_dim, self.rnn_dim))
		self.traj_dec.add_module("traj_relvu_dc_6", torch.nn.ReLU())
		self.traj_dec.add_module("traj_fc_dvc8", torch.nn.Linear(self.rnn_dim, 9*self.T))
		self.traj_dec.add_module("traj_fc_dvc9", torch.nn.Linear(9*self.T, 9*self.T))

		# decodes the shape of the object
		self.shap_dec = torch.nn.Sequential()
		self.shap_dec.add_module("fcv_sc1", torch.nn.Linear(self.T*self.rnn_dim, self.T*self.rnn_dim))
		self.shap_dec.add_module("revlu_sc_1", torch.nn.LeakyReLU(0.2))
		self.shap_dec.add_module("fcv_sc2", torch.nn.Linear(self.T*self.rnn_dim, self.T*self.rnn_dim))
		self.shap_dec.add_module("revlu_sc_2", torch.nn.ReLU())
		self.shap_dec.add_module("fc_vsc5", torch.nn.Linear(self.T*self.rnn_dim, self.T*self.rnn_dim))
		self.shap_dec.add_module("relvu_sc_5", torch.nn.ReLU())
		self.shap_dec.add_module("fcvv_sc6", torch.nn.Linear(self.T*self.rnn_dim, self.T*self.rnn_dim))
		self.shap_dec.add_module("revlu_sc_6", torch.nn.ReLU())
		self.shap_dec.add_module("fcv_sc8", torch.nn.Linear(self.T*self.rnn_dim, self.rnn_dim))
		self.shap_dec.add_module("revlu_sc_8", torch.nn.LeakyReLU(0.2))
		self.shap_dec.add_module("fcv_sc9x", torch.nn.Linear(self.rnn_dim,self.shape_dim))

		# p_r decoder
		self.vid_p_dec = torch.nn.Sequential()
		self.vid_p_dec.add_module("vfcvp5", torch.nn.Linear(self.rnn_dim + 9*self.T + self.shape_dim, self.rnn_dim))
		self.vid_p_dec.add_module("relu_vp5", torch.nn.ReLU())
		self.vid_p_dec.add_module("fcvp6", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_p_dec.add_module("relu_v6", torch.nn.ReLU())
		self.vid_p_dec.add_module("fcvp61", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_p_dec.add_module("relu_vp61", torch.nn.ReLU())
		self.vid_p_dec.add_module("vpfc62", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_p_dec.add_module("vprelu_62", torch.nn.ReLU())
		self.vid_p_dec.add_module("vpfc63", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_p_dec.add_module("vprelu_63", torch.nn.ReLU())
		self.vid_p_dec.add_module("vpfc60", torch.nn.Linear(self.rnn_dim, 20))

		# v decoder
		self.vid_v_dec = torch.nn.Sequential()
		self.vid_v_dec.add_module("vpfc7", torch.nn.Linear(self.rnn_dim + 9*self.T + self.shape_dim, self.rnn_dim))
		self.vid_v_dec.add_module("rvpelu_7", torch.nn.ReLU())
		self.vid_v_dec.add_module("vpfc8", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_v_dec.add_module("vprelu_70", torch.nn.ReLU())
		self.vid_v_dec.add_module("vpfc80", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_v_dec.add_module("vprelu_700", torch.nn.ReLU())
		self.vid_v_dec.add_module("vfc800", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_v_dec.add_module("vrelu_8", torch.nn.ReLU())
		self.vid_v_dec.add_module("vpfc91", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_v_dec.add_module("vprelu_91", torch.nn.ReLU())
		self.vid_v_dec.add_module("vpfc90c", torch.nn.Linear(self.rnn_dim, 40))

		# fc decoder
		self.vid_fc_dec = torch.nn.Sequential()
		self.vid_fc_dec.add_module("vfc10", torch.nn.Linear(self.rnn_dim + 9*self.T + self.shape_dim, self.rnn_dim))
		self.vid_fc_dec.add_module("rvelu_11", torch.nn.ReLU())
		self.vid_fc_dec.add_module("fvc12", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_fc_dec.add_module("revlu_12", torch.nn.ReLU())
		self.vid_fc_dec.add_module("fc1v20", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_fc_dec.add_module("revluv_120", torch.nn.ReLU())
		self.vid_fc_dec.add_module("fc1v2001", torch.nn.Linear(self.rnn_dim, 40))

		# p_e decoder
		self.vid_pe_dec = torch.nn.Sequential()
		self.vid_pe_dec.add_module("fcc13", torch.nn.Linear(self.rnn_dim + 9*self.T + self.shape_dim, self.rnn_dim))
		self.vid_pe_dec.add_module("rcelu_14", torch.nn.ReLU())
		self.vid_pe_dec.add_module("fcc14", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_pe_dec.add_module("reclu_14", torch.nn.ReLU())
		self.vid_pe_dec.add_module("fcc140", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_pe_dec.add_module("reclu_140", torch.nn.ReLU())
		self.vid_pe_dec.add_module("fcv14000x", torch.nn.Linear(self.rnn_dim, 20))

		# fc_e decoder
		self.vid_fce_dec = torch.nn.Sequential()
		self.vid_fce_dec.add_module("vfc16", torch.nn.Linear(self.rnn_dim + 9*self.T + self.shape_dim, self.rnn_dim))
		self.vid_fce_dec.add_module("vrelu_16", torch.nn.ReLU())
		self.vid_fce_dec.add_module("vfc17", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_fce_dec.add_module("vrelu_17", torch.nn.ReLU())
		self.vid_fce_dec.add_module("vfc170", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_fce_dec.add_module("vrelu_170", torch.nn.ReLU())
		self.vid_fce_dec.add_module("vfc17x001", torch.nn.Linear(self.rnn_dim, 40))

	#######################################	
	# Differentiable Optimization Methods #
	#######################################

	def setupCTO(self, w_err = 0.01, w_f = 1):
		# decision variables
		p = cp.Variable((2*self.N_c, self.T)) # contact location
		ddp = cp.Variable((2*self.N_c, self.T)) # contact location
		f = cp.Variable((2*self.N_c,self.T)) # forces
		gamma = cp.Variable((2*self.N_c,self.T)) # cone weights
		alpha1 = cp.Variable((self.N_c,self.T)) # vertex weights
		alpha2 = cp.Variable((self.N_c,self.T)) # vertex weights
		f_e = cp.Variable((4,self.T)) # external force
		gamma_e = cp.Variable((4,self.T)) # cone weights
		err = cp.Variable((3,self.T)) # errors
		
		# input parameters
		r = cp.Parameter((3, self.T)) # trajectory
		ddr = cp.Parameter((3, self.T)) # trajectory
		p_r = cp.Parameter((2*self.N_c, self.T)) # reference contact location
		fc = cp.Parameter((4*self.N_c, self.T)) # friction cone
		p_e = cp.Parameter((4, self.T)) # external contact location
		fc_e = cp.Parameter((8, self.T)) # external friction cone
		v = cp.Parameter((4*self.N_c, self.T)) # facets for each contacts
		# tol = cp.Parameter((1, 1)) # error tolerance

		# adds constraints
		constraints = []
		for t in range(self.T):
			# linear quasi-dynamics
			constraints.append(sum(f[:self.N_c,t]) + f_e[0,t] + f_e[1,t] == ddr[0,t] + err[0,t])
			constraints.append(sum(f[self.N_c:,t]) + f_e[2,t] + f_e[3,t] == ddr[1,t] + err[1,t])

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
			constraints.append(tau == ddr[2,t] + err[2,t])

			# constraints contacts to their respective facets
			for c in range(self.N_c):
				constraints.append(p[c,t]		   == alpha1[c,t]*v[c*4,t]	 + alpha2[c,t]*v[c*4 + 2,t])
				constraints.append(p[c+self.N_c,t] == alpha1[c,t]*v[c*4 + 1,t] + alpha2[c,t]*v[c*4 + 3,t])
				constraints.append(alpha1[c,t] + alpha2[c,t] == 1)
				constraints.append(alpha1[c,t] >= 0)
				constraints.append(alpha2[c,t] >= 0)
				# if t < 4:
					# constraints.append(p[c,t] == p_r[c,t])
					# constraints.append(p[c+self.N_c,t] == p_r[c+self.N_c,t])

			# friction cone constraints
			for c in range(self.N_c):
				constraints.append(gamma[c,t]*fc[c*4,t]	 + gamma[c + self.N_c,t]*fc[c*4 + 2,t] == f[c,t])
				constraints.append(gamma[c,t]*fc[c*4 + 1,t] + gamma[c + self.N_c,t]*fc[c*4 + 3,t] == f[self.N_c + c,t])
				constraints.append(gamma[c,t] >= 0)
				constraints.append(gamma[self.N_c + c,t] >= 0)
			
			# external friction cone constratins
			constraints.append(gamma_e[0,t]*fc_e[0,t] + gamma_e[1,t]*fc_e[2,t] == f_e[0,t])
			constraints.append(gamma_e[0,t]*fc_e[1,t] + gamma_e[1,t]*fc_e[3,t] == f_e[2,t])
			# constraints.append(gamma_e[0,t] >= 0)
			# constraints.append(gamma_e[1,t] >= 0)

			constraints.append(gamma_e[2,t]*fc_e[4,t] + gamma_e[3,t]*fc_e[6,t] == f_e[1,t])
			constraints.append(gamma_e[2,t]*fc_e[5,t] + gamma_e[3,t]*fc_e[7,t] == f_e[3,t])
			# constraints.append(gamma_e[2,t] >= 0)
			# constraints.append(gamma_e[3,t] >= 0)

		objective = cp.Minimize(w_err*cp.pnorm(err, p=2) + w_f*cp.pnorm(f, p=2))
		problem = cp.Problem(objective, constraints)
		
		return CvxpyLayer(problem, parameters=[r, ddr, fc, p_e, fc_e, v, p_r], variables=[p, f, f_e, alpha1, alpha2, gamma, gamma_e, err])

	def setupDiff(self, dims = 3, T = 5):
		# decision variables
		dr = cp.Variable((dims, T))
		ddr = cp.Variable((dims, T))

		# parameters
		r = cp.Parameter((dims, T))
		
		# adds finite-diff constraints
		constraints = []
		for t in range(T):
			for d in range(dims):
				if t == 0:
					constraints.append(ddr[d,t]*(self.dt**2) == 0)
					constraints.append(dr[d,t]*(self.dt) == r[d,t+1] - r[d,t])
				elif t == T-1:
					constraints.append(ddr[d,t]*(self.dt**2) == 0)
					constraints.append(dr[d,t]*(2*self.dt) == r[d,t] - r[d,t-1])
				else:
					constraints.append(ddr[d,t]*(self.dt**2) == r[d,t-1] - 2*r[0,t] + r[d,t+1])
					constraints.append(dr[d,t]*(self.dt) == r[d,t+1] - r[d,t])

		objective = cp.Minimize(cp.pnorm(ddr, p=2))
		problem = cp.Problem(objective, constraints)
		
		return CvxpyLayer(problem, parameters=[r], variables=[dr, ddr])

	#####################################	
	# Differentiable Simulation Methods #
	#####################################

	def forwardPlanarDiffSim(self, p, dp, ddp, polygon, xtraj0, rad, render = False):

		bodies = []
		joints = []
		restitution = 0 # no impacts in quasi-dynamics
		n_pol = int(polygon[0])
		scale = 2500

		print(xtraj0)

		xr = 500+scale*xtraj0[0]
		yr = 500-scale*xtraj0[1]

		# adds body based on triangulation

		verts = 0.01*np.array([[0.01, 0.01], [-0.01, 0.01], [-0.01, -0.01], [0.01, -0.01]])
		r0 = Hull([xr, yr], verts, restitution=0, fric_coeff=10, mass = 0.1, name="obj")
		r0.add_force(MDP(g=100))
		bodies.append(r0)

		for i in range(n_pol):
			x2 = [polygon[1+8*i], -polygon[2+8*i]]
			x1 = [polygon[3+8*i], -polygon[4+8*i]]
			x0 = [polygon[5+8*i], -polygon[6+8*i]]
			verts = scale*np.array([x0, x1, x2])
			p0 = np.array([xr + polygon[7+8*i], yr - polygon[8+8*i]])
			r1 = Hull(p0, verts, restitution=restitution, mass = 0.1, fric_coeff=1, name="obj_"+str(i))
			r1.add_force(MDP(g=100))	
			r1.add_no_contact(bodies[0])
			for j in range(i):
				r1.add_no_contact(bodies[j])
			bodies.append(r1)
			joints += [FixedJoint(r1, bodies[0])]

		pdb.set_trace()

		# Point Fingers
		traj_f = []
		for i in range(self.N_c):
			pos0 = [500+scale*p[i,0],500-scale*p[i+self.N_c,0]]
			c = Circle(pos0, 1, mass = 1, vel=(0, 0, 0), restitution=restitution, fric_coeff=1, name = "f"+str(i))
			# c.add_force(FingerTrajectory(torch.cat((0*ddp[i,:], scale*ddp[i,:], -scale*ddp[i+self.N_c,:]), axis=0).view(3,self.T+1)))
			traj = torch.cat((scale*dp[i,:],-scale*dp[i+self.N_c,:]), axis=0).view(2,self.T+1)
			traj_f.append(Trajectory(vel = traj, name = "f"+str(i)))
			if i > 0:
				c.add_no_contact(bodies[-1])
			c.add_no_contact(bodies[0])
			bodies.append(c)

		world = World(bodies, joints, dt=self.dt/10, tol = 1e-6, eps=float('inf'), post_stab = True, strict_no_penetration = False)
		screen = None
		if render:
			pygame.init()
			screen = pygame.display.set_mode((1000, 1000), pygame.DOUBLEBUF)
			screen.set_alpha(None)
		run_world_traj(world, run_time = 0.61, screen=screen, recorder=None, print_time=False, traj=traj_f)
		for t in range(self.T):
			if t > 0:
				y0 = torch.cat((world.states[t][1].view(1,1) - 500, 500 - world.states[t][2].view(1,1), scale*world.states[t][0].view(1,1)*0.03), axis = 0)/scale
				y = torch.cat((y,y0), axis = 1)
			else:
				y = torch.cat((world.states[t][1].view(1,1) - 500, 500 - world.states[t][2].view(1,1), scale*world.states[t][0].view(1,1)*0.03), axis = 0)/scale
		return y

	def forwardSagittalDiffSim(self, p, dp, ddp, polygon, xtraj0, rad, render = False):
		bodies = []
		joints = []
		restitution = 0 # no impacts in quasi-dynamics
		n_pol = int(polygon[0])
		scale = 2500

		print(xtraj0)

		xr = 500+scale*xtraj0[0]
		yr = 500-scale*xtraj0[1]

		# adds body based on triangulation
		verts = 0.01*np.array([[0.01, 0.01], [-0.01, 0.01], [-0.01, -0.01], [0.01, -0.01]])
		r0 = Hull([xr, yr], verts, restitution=0, fric_coeff=10, mass = 0.1, name="obj")
		r0.add_force(MDP(g=100))
		bodies.append(r0)

		for i in range(n_pol):
			x2 = [polygon[1+8*i], -polygon[2+8*i]]
			x1 = [polygon[3+8*i], -polygon[4+8*i]]
			x0 = [polygon[5+8*i], -polygon[6+8*i]]
			verts = scale*np.array([x0, x1, x2])
			p0 = np.array([xr + polygon[7+8*i], yr - polygon[8+8*i]])
			r1 = Hull(p0, verts, restitution=restitution, mass = 0.1, fric_coeff=1, name="obj_"+str(i))
			r1.add_force(MDP(g=100))
			r1.add_no_contact(bodies[0])
			bodies.append(r1)
			joints += [FixedJoint(r1, bodies[0])]
			for j in range(i):
				r1.add_no_contact(bodies[j])

		# Point Fingers
		traj_f = []
		for i in range(self.N_c):
			pos0 = [500+scale*p[i,0],500-scale*p[i+self.N_c,0]]
			c = Circle(pos0, 1, mass = 1, vel=(0, 0, 0), restitution=restitution, fric_coeff=0.01, name = "f"+str(i))
			bodies.append(c)
			c.add_no_contact(bodies[0])
			traj = torch.cat((scale*dp[i,:],-scale*dp[i+self.N_c,:]), axis=0).view(2,self.T+1)
			traj_f.append(Trajectory(vel = traj, name = "f"+str(i)))
		c.add_no_contact(bodies[-2])

		# Environment
		r = Rect([0, 500, 505], [900, 10], restitution=restitution, fric_coeff=1)
		bodies.append(r)
		joints.append(TotalConstraint(r))

		world = World(bodies, joints, dt=self.dt/10, tol = 1e-6, eps=1000, post_stab = True, strict_no_penetration = False)
		screen = None


		pygame.init()
		screen = pygame.display.set_mode((1000, 1000), pygame.DOUBLEBUF)
		screen.set_alpha(None)
		run_world_traj(world, run_time = 0.61, screen=screen, recorder=None, print_time=False, traj=traj_f)
		for t in range(self.T):
			if t > 0:
				y0 = torch.cat((world.states[t][1].view(1,1) - 500, 500 - world.states[t][2].view(1,1), scale*world.states[t][0].view(1,1)*1e-3), axis = 0)/scale
				y = torch.cat((y,y0), axis = 1)
			else:
				y = torch.cat((world.states[t][1].view(1,1) - 500, 500 - world.states[t][2].view(1,1), scale*world.states[t][0].view(1,1)*1e-3), axis = 0)/scale

		return y

	########################	
	# Forward Pass Methods #
	########################

	def forwardCTO(self, xtraj, x, x_img): 
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
			p_r = self.p_dec.forward(torch.cat((e_img[i,:].view(1,-1), xtraj[i,:].view(1,45)), 1))
			v = self.v_dec.forward(torch.cat((e_img[i,:].view(1,-1), xtraj[i,:].view(1,45)), 1))
			p_e = self.pe_dec.forward(torch.cat([e_img[i,:].view(1,-1), xtraj[i,:].view(1,45)], 1))
			fc = self.fc_dec.forward(torch.cat([xtraj[i,:].view(1,45), v.view(1,40)], 1))
			fc_e = self.fce_dec.forward(torch.cat([p_e.view(1,20), xtraj[i,:].view(1,45)], 1))

			# p_e = p_e0
			# p_r = p_r0
			# v = v0
			# fc = fc0
			# fc_e = fc_e0

			p, f, _, _, _, _, _, _, _ = self.CTOlayer(r.view(3,5), ddr.view(3,5), fc.view(8,5), p_e.view(4,5), fc_e.view(8,5), v.view(8, 5), p_r.view(4, 5))
			
			# pdb.set_trace()


			if first:				
				y = 33*torch.cat([p.view(1,-1), f.view(1,-1)], axis = 1)
				first = False
			else:
				y_1 = 33*torch.cat([p.view(1,-1), f.view(1,-1)], axis = 1)
				y = torch.cat((y, y_1), axis = 0)
		return y

	def forward2Sim(self, xtraj, x, x_img, polygons, render=False, bypass = True): 
		# passes through the optimization problem
		first = True
	
		# shape encoding		
		e_img = self.forwardShapeEncoder(np.reshape(x_img,(-1,1,50,50)))
		print(np.shape(e_img))
		# param decoding
		p_r = self.p_dec.forward(torch.cat((e_img.view(-1,self.shape_dim), xtraj.view(-1,45)), 1))
		v = self.v_dec.forward(torch.cat((e_img.view(-1,self.shape_dim), xtraj.view(-1,45)), 1))
		p_e = self.pe_dec.forward(torch.cat([e_img.view(-1,self.shape_dim), xtraj.view(-1,45)], 1))
		fc = self.fc_dec.forward(torch.cat([xtraj.view(-1,45), v.view(-1,40)], 1))
		fc_e = self.fce_dec.forward(torch.cat([p_e.view(-1,20), xtraj.view(-1,45)], 1))

		p_e0 = x[:,60:80] # external contact location
		v0 = x[:,120:160] # contact affordance
		p_r0 = x[:,0:20]
		fc0 = x[:,20:60]
		fc_e0 = x[:,80:120]

		# decodes each trajectory
		for i in range(np.shape(x)[0]):
			# params that should be obtained from video
			r = xtraj[i,:15]

			# learnes the parameters
			dr, ddr = self.Difflayer3(r.view(3,5))

			# solves for the contact trajectory
			if bypass:	
				p = self.cto_su1(torch.cat(((r.view(1,-1), ddr.view(1,-1), fc[i,:].view(1,-1), p_e[i,:].view(1,-1), fc_e[i,:].view(1,-1), v[i,:].view(1,-1), p_r[i,:].view(1,-1))), axis=1)).view(4,5)
				f = self.cto_su2(torch.cat(((r.view(1,-1), ddr.view(1,-1), fc[i,:].view(1,-1), p_e[i,:].view(1,-1), fc_e[i,:].view(1,-1), v[i,:].view(1,-1), p_r[i,:].view(1,-1))), axis=1)).view(4,5)
			else:
				failed = False
				while True:
					try: 
						p, f, _, _, _, _, _, err = self.CTOlayer(r.view(3,5), ddr.view(3,5), fc[i,:].view(8,5), p_e[i,:].view(4,5), fc_e[i,:].view(8,5), v[i,:].view(8, 5), p_r[i,:].view(4, 5))
						break
					except:
						failed = True
						self.w = self.w/10
						self.CTOlayer = self.setupCTO(w_err = self.w)
						print('infeasible')
						pass

				if failed:
					print('recovered from error')
					self.w = 0.01
					self.CTOlayer = self.setupCTO(w_err = self.w)

			# for t in range(5):
			# p0 = 1.2*(p[:,0]-torch.cat((r.view(3,5)[0:2,0],r.view(3,5)[0:2,0]),axis=0))+torch.cat((r.view(3,5)[0:2,0],r.view(3,5)[0:2,0]),axis=0)
			p0 = p[0:4,0]
			p = torch.cat((p0.view(-1,1), p), axis = 1)
			print(p0)
			dp, _ = self.Difflayer(p.view(4,6))
			ddp, _ = self.Difflayer(dp.view(4,6))
			
			xtraj_new = self.forwardPlanarDiffSim(p, dp.double(), ddp.double(), polygons[i,:], r.view(3,5)[:,0], 10, render)
			print(xtraj_new)
			if i == 0:				
				y = torch.cat((xtraj_new.view(1,-1), 1e-6*ddp.view(1,-1).double()), axis = 1)
			else:
				y1 = torch.cat((xtraj_new.view(1,-1), 1e-6*ddp.view(1,-1).double()), axis = 1)
				y = torch.cat((y, y1), axis = 0)
		# self.rad = self.rad/1.1
		# if self.rad < 1:
		# 	self.rad = 1
		return y

	def forwardVideo(self, video, xtraj, x, bypass = False):
		# passes through the optimization problem
		# video encoding
		for t in range(self.T):
			frame = video[:,t*3*100*100:(t+1)*3*100*100]
			e_frame = self.forwardFrameCNN(frame).view(-1,1,self.frame_dim)
			if t == 0:
				e_vid = e_frame
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
		
		# passes frames through RNN
		rnn_out, (h_n, h_c) = self.videoRNN(e_vid.view(-1,self.T,self.frame_dim))
		e_vid = self.fcRNN(rnn_out[:, -1, :])
		# pdb.set_trace()
		# e_vid = self.lstm_su(e_vid.view(-1,self.T*self.frame_dim))
		# extracts image encoding and object trajectory
		
		e_img = self.shap_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))
		xtraj = self.traj_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))

		embedding = torch.cat((e_vid, e_img, xtraj), axis = 1)

		p_r = self.vid_p_dec.forward(embedding)
		v = self.vid_v_dec.forward(embedding)
		p_e = self.vid_pe_dec.forward(embedding)
		fc = self.vid_fc_dec.forward(embedding)
		fc_e = self.vid_fce_dec.forward(embedding)

		p_e0 = x[:,60:80] # external contact location
		v0 = x[:,120:160] # contact affordance
		p_r0 = x[:,0:20]	
		fc0 = x[:,20:60]
		fc_e0 = x[:,80:120]

		# self.w = 0.01
		# self.CTOlayer = self.setupCTO(w_err = self.w)

		# decodes each trajectory
		# print('going through cvx')
		for i in range(np.shape(e_frame)[0]):
			# params that should be obtained from video
			r = xtraj[i,:15]
			# params that can be computed explicity
			dr = xtraj[i,15:30]
			ddr = xtraj[i,30:]

			# learnes the parameters
			# dr, ddr = self.Difflayer3(r.view(3,5))

			# solves for the contact trajectory
			if bypass:
				p = self.cto_su1(torch.cat(((r.view(1,-1), ddr.view(1,-1), fc[i,:].view(1,-1), p_e[i,:].view(1,-1), fc_e[i,:].view(1,-1), v[i,:].view(1,-1), p_r[i,:].view(1,-1))), axis=1)).view(4,5)
				f = self.cto_su2(torch.cat(((r.view(1,-1), ddr.view(1,-1), fc[i,:].view(1,-1), p_e[i,:].view(1,-1), fc_e[i,:].view(1,-1), v[i,:].view(1,-1), p_r[i,:].view(1,-1))), axis=1)).view(4,5)
				err = f
			else:
				failed = False
				while True:
					try: 
						p, f, _, _, _, _, _, err = self.CTOlayer(r.view(3,5), ddr.view(3,5), fc[i,:].view(8,5), p_e[i,:].view(4,5), fc_e[i,:].view(8,5), v[i,:].view(8, 5), p_r[i,:].view(4, 5))
						break
					except:
						failed = True
						self.w = self.w/10
						self.CTOlayer = self.setupCTO(w_err = self.w)
						print('infeasible')
						pass

				if failed:
					print('recovered from error')

			# self.w = 0.01

			if i == 0:				
				y = torch.cat([p.view(1,-1), 1e-6*torch.max(err).view(1,-1)], axis = 1)
			else:
				y = torch.cat((y, torch.cat([p.view(1,-1), 1e-6*torch.max(err).view(1,-1)], axis = 1)), axis = 0)
		# self.tol = self.tol/1.1
		return y

	def forwardEndToEnd(self, video, polygons, xtraj, render = False, bypass = False):
		# passes through the optimization problem
		# video encoding
		for t in range(self.T):
			frame = video[:,t*3*100*100:(t+1)*3*100*100]
			e_frame = self.forwardFrameCNN(frame).view(-1,1,self.frame_dim)
			if t == 0:
				e_vid = e_frame
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
		
		# passes frames through RNN
		# rnn_out, (h_n, h_c) = self.videoRNN(e_vid.view(-1,self.T,self.frame_dim))
		# e_vid = self.fcRNN(rnn_out[:, -1, :]).detach()

		e_vid = self.lstm_su(e_vid.view(-1,self.T*self.frame_dim))
		# print('decoding')
			
		print('decoding')
		# extracts image encoding and object trajectory
		# e_img = self.shap_dec(e_vid)
		# xtraj = self.traj_dec(e_vid)
		p_r = self.vid_p_dec.forward(e_vid)
		v = self.vid_v_dec.forward(e_vid)
		p_e = self.vid_pe_dec.forward(e_vid)
		fc = self.vid_fc_dec.forward(e_vid)
		fc_e = self.vid_fce_dec.forward(e_vid)
		print('going through cvx + sim')
		# decodes each trajectory
		for i in range(np.shape(e_vid)[0]):
			# params that should be obtained from video
			r = xtraj[i,:15]
			# params that can be computed explicity
			# dr = xtraj[i,15:30]
			# ddr = xtraj[i,30:]

			# learnes the parameters
			dr, ddr = self.Difflayer3(r.view(3,5))
			# print(ddr)
			# print(v)
			# solves for the contact trajectory
			
			if bypass:
				pass
				p = cto_su1(torch.cat(((r.view(1,-1), ddr.view(1,-1), fc[i,:].view(1,-1), p_e[i,:].view(1,-1), fc_e[i,:].view(1,-1), v[i,:].view(1,-1), p_r[i,:].view(1,-1))), axis=1)).view(4,5)
				f = cto_su2(torch.cat(((r.view(1,-1), ddr.view(1,-1), fc[i,:].view(1,-1), p_e[i,:].view(1,-1), fc_e[i,:].view(1,-1), v[i,:].view(1,-1), p_r[i,:].view(1,-1))), axis=1)).view(4,5)
			else:
				failed = False
				while True:
					try: 
						p, f, _, _, _, _, _, err = self.CTOlayer(r.view(3,5), ddr.view(3,5), fc[i,:].view(8,5), p_e[i,:].view(4,5), fc_e[i,:].view(8,5), v[i,:].view(8, 5), p_r[i,:].view(4, 5))
						break
					except:
						failed = True
						self.w = self.w/10
						self.CTOlayer = self.setupCTO(w_err = self.w)
						print('infeasible')
						pass

				if failed:
					print('recovered from error')
					self.w = 0.01
					self.CTOlayer = self.setupCTO(w_err = self.w)
			
			# normalizes to avoid penetrration
			d1 = (p[0:2,0]-r.view(3,5)[0:2,0])
			d2 = (p[2:4,0]-r.view(3,5)[0:2,0])

			p1 = d1/(torch.norm(d1) + 1e-6)
			p2 = d2/(torch.norm(d2) + 1e-6)

			# p0 = 500*torch.cat((p1,p2), axis = 0) + torch.cat((r.view(3,5)[0:2,0],r.view(3,5)[0:2,0]), axis = 0)

			p0 = p[0:4,0]
			
			p = torch.cat((p0.view(-1,1), p), axis = 1)

			dp, ddp = self.Difflayer(p.view(4,6))
			
			xtraj_new = self.forwardPlanarDiffSim(p, dp.double(), ddp.double(), polygons[i,:], r.view(3,5)[:,0], 10, render)
			
			print(xtraj_new)
			print(r.view(3,5))
			print(err)

			if i == 0:				
				y = torch.cat([xtraj_new.view(1,-1).float()], axis = 1)
			else:
				y = torch.cat((y, torch.cat([xtraj_new.view(1,-1).float()], axis = 1)), axis = 0)

		return y

	#####################	
	# Parameter Methods #
	#####################

	def forwardVideotoImage(self,video):
		for t in range(self.T):
			frame = video[:,t*3*100*100:(t+1)*3*100*100]
			e_frame = self.forwardFrameCNN(frame).detach().view(-1,1,self.frame_dim)
			if t == 0:
				e_vid = e_frame
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
		
		# passes frames through RNN
		rnn_out, (h_n, h_c) = self.videoRNN(e_vid.view(-1,self.T,self.frame_dim))
		e_vid = self.fcRNN(rnn_out[:, -1, :])
		
		# extracts image encoding
		e_img = self.shap_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))

		# decodes image
		return self.decoder(self.fc3(e_img).view(-1,20,6,6))

	def forwardVideotoTraj(self,video):
		# passes through all frames
		for t in range(self.T):
			frame = video[:,t*3*100*100:(t+1)*3*100*100]
			e_frame = self.forwardFrameCNN(frame).detach().view(-1,1,self.frame_dim)
			if t == 0:
				e_vid = e_frame
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
		
		# passes frames through RNN
		rnn_out, (h_n, h_c) = self.videoRNN(e_vid.view(-1,self.T,self.frame_dim))
		
		# extracts object trajecotry
		return self.traj_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))

	#####################
	# Assesment methods #
	#####################

	def forwardVideoToParams(self, video, x):
		# passes through the optimization problem
		# video encoding
		for t in range(self.T):
			frame = video[:,t*3*100*100:(t+1)*3*100*100]
			e_frame = self.forwardFrameCNN(frame).view(-1,1,self.frame_dim)
			if t == 0:
				e_vid = e_frame
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
		
		# passes frames through RNN
		rnn_out, (h_n, h_c) = self.videoRNN(e_vid.view(-1,self.T,self.frame_dim))
		e_vid = self.fcRNN(rnn_out[:, -1, :])
	
		e_img = self.shap_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))
		xtraj = self.traj_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))

		embedding = torch.cat((e_vid, e_img, xtraj), axis = 1)

		# extracts the parameters
		p_r = self.vid_p_dec.forward(embedding)
		v = self.vid_v_dec.forward(embedding)
		p_e = self.vid_pe_dec.forward(embedding)
		fc = self.vid_fc_dec.forward(embedding)
		fc_e = self.vid_fce_dec.forward(embedding)

		# params that can be learned from above
		p_r0 = x[:,0:20]
		v0 = x[:,120:160]
		p_e0 = x[:,60:80]
		fc0 = x[:,20:60]
		fc_e0 = x[:,80:120]

		return p_r-p_r0, v-v0, p_e-p_e0, fc-fc0, fc_e-fc_e0

	###################################	
	# Variational AutoEncoder Methods #
	###################################

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
		return self.decoder(z.view(-1,20,6,6)), mu, logvar

	def forwardFrameVAE(self, x_vid):
		# encodes
		# print(np.shape(x_vid))
		h = self.vid_enc(np.reshape(x_vid,(-1,3,100,100)))
		# print(np.shape(h))
		# bottlenecks
		mu, logvar = self.vid_fc1(h), self.vid_fc2(h)
		std = logvar.mul(0.5).exp_()
		esp = torch.randn(*mu.size())
		# reparametrizes
		z = mu + std * esp
		z = self.vid_fc3(z)
		z = self.vid_dec0(z)
		# print(np.shape(z))
		# decodes
		h = self.vid_dec(z.view(-1,24,12,12))
		# print(np.shape(h))
		return h, mu, logvar


	def forwardFrameCVAE(self, x_vid, x_vid0):
		# encodes

		h1 = self.vid_enc(np.reshape(x_vid,(-1,3,100,100)))
		h0 = self.vid_enc(np.reshape(x_vid0,(-1,3,100,100)))

		h = self.cvid_enc(torch.cat((h0, h1), axis = 1))
		# print(np.shape(h))
		# bottlenecks

		mu, logvar = self.vid_fc1(h), self.vid_fc2(h)
		std = logvar.mul(0.5).exp_()
		esp = torch.randn(*mu.size())
		# reparametrizes
		z = mu + std * esp
		
		# decdes
		z1 = self.vid_fc3(z)
		z1 = self.vid_dec0(z1)
		
		h1 = self.vid_dec(z1.view(-1,24,12,12))
		# print(np.shape(h))
		return h1, mu, logvar


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
		h = self.vid_enc(np.reshape(frame,(-1,3,100,100)))
		
		# bottlenecks
		mu, logvar = self.vid_fc1(h), self.vid_fc2(h)
		std = logvar.mul(0.5).exp_()
		esp = torch.randn(*mu.size())

		# reparametrizes
		z = mu + std * esp

		# encodes
		# print(np.shape(z))
		return m

	def forwardFrameCNN(self, frame):
		# encodes
		h = self.vid_enc(np.reshape(frame,(-1,3,100,100)))

		# returns
		return self.vid_fc1(h)


	###################	
	# Utility Methods #
	###################

	def save(self, name = "cnn_model.pt"):
		torch.save(self.state_dict(),"../data/models/"+str(name))

	def load(self, name = "cnn_model.pt"):
		self.load_state_dict(torch.load("../data/models/"+str(name)),strict = False)

	def gen_res(self,inputs_1,inputs_2,inputs_img,name="res"):
		y = self.forward(torch.tensor(inputs_1).float(),torch.tensor(inputs_2).float(),torch.tensor(inputs_img).float())
		y = y.clone().detach()
		np.savetxt("../data/"+name+"_0_2f.csv", y.data.numpy(), delimiter=",")

	def gen_resVid(self,vids, x, xtraj, name="res"):
		y = self.forwardVideo(torch.tensor(vids).float(), x.float(), xtraj.float())
		y = y.clone().detach()
		np.savetxt("../data/"+name+"_0_2f.csv", y.data.numpy(), delimiter=",")
