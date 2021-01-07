import torch
import pdb
from numpy import loadtxt
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.siren import *

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import pygame
from lcp_physics.physics.bodies import Circle, Rect, Hull
from lcp_physics.physics.constraints import Joint, TotalConstraint
from lcp_physics.physics.constraints import FixedJoint
from lcp_physics.physics.forces import Gravity, MDP, FingerTrajectory
from lcp_physics.physics.world import World, run_world, Trajectory
from src.FinDiffSim import *

class ContactNet(torch.nn.Module):
	def __init__(self, N_data = 0, sagittal = False):
		super(ContactNet, self).__init__()
		# contact-trajectory parameters 
		self.N_c = 2
		self.T = 5
		self.dt = 0.1
		self.tol = 1
		self.rad = 10
		self.shape_dim = 50
		self.frame_dim = 75
		self.rnn_dim = 150
		self.sagittal = sagittal
		self.gamma = 0.001

		if self.sagittal:
			self.sim = FinDiffSagSim
		else:
			self.sim = FinDiffSim

		self.w = 0.01

		# CTO layers
		if sagittal:
			self.CTOlayer = self.setupCTOSag(w_err = self.w)
		else:
			self.CTOlayer = self.setupCTO(w_err = self.w)

		self.Difflayer = self.setupDiff(dims=4, T = 5)
		self.Difflayer3 = self.setupDiff(dims=3, T = 5)

		self.sin_layer = Siren(self.rnn_dim*self.T,self.rnn_dim*self.T,self.rnn_dim*self.T,self.rnn_dim*self.T)

		self.cto_su1 = torch.nn.Sequential()
		self.cto_su1.add_module("surr_fcx10", torch.nn.Linear(20, 20))

		self.cto_su2 = torch.nn.Sequential()
		self.cto_su2.add_module("surr_fcx220", torch.nn.Linear(20, 20))

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
		self.encoder.add_module("enc_conv_1", torch.nn.Conv2d(1, 5, 4, 2, 1))
		self.encoder.add_module("enc_bn_1", torch.nn.BatchNorm2d(5))
		self.encoder.add_module("enc_relu_1", torch.nn.ReLU())
		self.encoder.add_module("enc_drop_1", torch.nn.Dropout2d(0.1))
		self.encoder.add_module("enc_conv_2", torch.nn.Conv2d(5, 10, 4, 2, 1))
		self.encoder.add_module("enc_bn_2", torch.nn.BatchNorm2d(10))
		self.encoder.add_module("enc_relu_2", torch.nn.ReLU())
		self.encoder.add_module("enc_conv_3", torch.nn.Conv2d(10, 20, 4, 2, 1))
		self.encoder.add_module("enc_bn_3", torch.nn.BatchNorm2d(20))
		self.encoder.add_module("enc_relu_3", torch.nn.ReLU())
		self.encoder.add_module("enc_flatten", torch.nn.Flatten())
		self.encoder.add_module("enc_linearxxx", torch.nn.Linear(720,720))

		# layers for VAE
		self.fc1 = torch.nn.Linear(720, self.shape_dim)
		self.fc2 = torch.nn.Linear(720, self.shape_dim)
		self.fc3 = torch.nn.Linear(self.shape_dim, 720)
		self.fc4 = torch.nn.Linear(self.shape_dim, 720)

		# shape decoders
		self.decoder = torch.nn.Sequential()
		self.decoder.add_module("deconvups_3", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.decoder.add_module("derepups_3", torch.nn.ReplicationPad2d(1))
		self.decoder.add_module("deconv_3", torch.nn.Conv2d(20, 10, 2, 1))	
		self.decoder.add_module("debn_3", torch.nn.BatchNorm2d(10, 1.e-3))
		self.decoder.add_module("derelu_3", torch.nn.ReLU())
		self.decoder.add_module("deconvups_2", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.decoder.add_module("derepups_2", torch.nn.ReplicationPad2d(1))
		self.decoder.add_module("deconv_2", torch.nn.Conv2d(10, 5, 3, 1))	
		self.decoder.add_module("debn_2", torch.nn.BatchNorm2d(5, 1.e-3))
		self.decoder.add_module("derelu_2", torch.nn.ReLU())
		self.decoder.add_module("deconvups_1", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.decoder.add_module("derepups_1", torch.nn.ReplicationPad2d(1))
		self.decoder.add_module("deconv_1", torch.nn.Conv2d(5, 1, 5, 1))
		self.decoder.add_module("derelu_1", torch.nn.Sigmoid())

		# environment decoders
		self.edecoder = torch.nn.Sequential()
		self.edecoder.add_module("edeconvups_3", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.edecoder.add_module("ederepups_3", torch.nn.ReplicationPad2d(1))
		self.edecoder.add_module("edeconv_3", torch.nn.Conv2d(20, 10, 2, 1))	
		self.edecoder.add_module("edebn_3", torch.nn.BatchNorm2d(10, 1.e-3))
		self.edecoder.add_module("ederelu_3", torch.nn.ReLU())
		self.edecoder.add_module("edeconvups_2", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.edecoder.add_module("ederepups_2", torch.nn.ReplicationPad2d(1))
		self.edecoder.add_module("edeconv_2", torch.nn.Conv2d(10, 5, 3, 1))	
		self.edecoder.add_module("edebn_2", torch.nn.BatchNorm2d(5, 1.e-3))
		self.edecoder.add_module("ederelu_2", torch.nn.ReLU())
		self.edecoder.add_module("edeconvups_1", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.edecoder.add_module("ederepups_1", torch.nn.ReplicationPad2d(1))
		self.edecoder.add_module("edeconv_1", torch.nn.Conv2d(5, 1, 5, 1))
		self.edecoder.add_module("ederelu_1", torch.nn.Sigmoid())

	def addFrameCVAELayers(self):
		# frame encoder
		self.vid_enc = torch.nn.Sequential()
		self.vid_enc.add_module("vid_conv_1", torch.nn.Conv2d(3, 6, 4, 2, 1))
		self.vid_enc.add_module("vid_bn_1", torch.nn.BatchNorm2d(6))
		self.vid_enc.add_module("vid_relu_1", torch.nn.ReLU())
		self.vid_enc.add_module("vid_drop_1", torch.nn.Dropout2d(0.2))
		self.vid_enc.add_module("vid_conv_2", torch.nn.Conv2d(6, 12, 4, 2, 1))
		self.vid_enc.add_module("vid_bn_2", torch.nn.BatchNorm2d(12))
		self.vid_enc.add_module("vid_relu_2", torch.nn.ReLU())
		self.vid_enc.add_module("vid_drop_2", torch.nn.Dropout2d(0.1))
		self.vid_enc.add_module("vid_conv_3", torch.nn.Conv2d(12, 24, 4, 2, 1))
		self.vid_enc.add_module("vid_bn_3", torch.nn.BatchNorm2d(24))
		self.vid_enc.add_module("vid_relu_3", torch.nn.ReLU())
		self.vid_enc.add_module("vid_flatten", torch.nn.Flatten())
		self.vid_enc.add_module("vid_fc_enc", torch.nn.Linear(3456,2*self.frame_dim))
		self.vid_enc.add_module("vid_bn_4", torch.nn.BatchNorm1d(2*self.frame_dim))
		self.vid_enc.add_module("vid_relu_6", torch.nn.ReLU())


		self.cvid_enc = torch.nn.Sequential()
		self.cvid_enc.add_module("cvid_fc_dec1", torch.nn.Linear(4*self.frame_dim,2*self.frame_dim))
		self.cvid_enc.add_module("cvid_relu_dec1", torch.nn.ReLU())
		self.cvid_enc.add_module("cvid_fc_dec_out", torch.nn.Linear(2*self.frame_dim,2*self.frame_dim))

		# layers for frame VAE
		self.shape_to_shapes = torch.nn.Sequential()
		self.shape_to_shapes.add_module("sts_fc1", torch.nn.Linear(self.shape_dim,7*self.shape_dim))
		self.shape_to_shapes.add_module("sts_bn1", torch.nn.BatchNorm1d(7*self.shape_dim))
		self.shape_to_shapes.add_module("sts_elu1", torch.nn.ReLU())
		# self.shape_to_shapes.add_module("sts_drop_1", torch.nn.Dropout(0.2))
		self.shape_to_shapes.add_module("sts_fc2", torch.nn.Linear(7*self.shape_dim,7*self.shape_dim))

		# layers for frame VAE
		self.vid_fc1 = torch.nn.Linear(2*self.frame_dim, self.frame_dim)
		self.vid_fc2 = torch.nn.Linear(2*self.frame_dim, self.frame_dim)
		self.vid_fc3 = torch.nn.Linear(self.frame_dim, 2*self.frame_dim)

		self.vid_dec0 = torch.nn.Sequential()
		self.vid_dec0.add_module("vid_relu_dec6", torch.nn.ReLU())
		self.vid_dec0.add_module("vid_fc_dec4", torch.nn.Linear(2*self.frame_dim,3456))
		self.vid_dec0.add_module("vid_relu_dec3", torch.nn.ReLU())

		# frame decoders
		self.vid_dec = torch.nn.Sequential()
		self.vid_dec.add_module("vdeconvups_3", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.vid_dec.add_module("vderepups_3", torch.nn.ReplicationPad2d(1))
		self.vid_dec.add_module("vdeconv_3", torch.nn.Conv2d(24, 12, 2, 1))
		self.vid_dec.add_module("vdebn_3", torch.nn.BatchNorm2d(12, 1.e-3))
		self.vid_dec.add_module("vderelu_3", torch.nn.ReLU())
		self.vid_dec.add_module("vdeconvups_2", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.vid_dec.add_module("vderepups_2", torch.nn.ReplicationPad2d(1))
		self.vid_dec.add_module("vdeconv_2", torch.nn.Conv2d(12, 6, 2, 1))	
		self.vid_dec.add_module("vdebn_2", torch.nn.BatchNorm2d(6, 1.e-3))
		self.vid_dec.add_module("vderelu_2", torch.nn.ReLU())
		self.vid_dec.add_module("vdeconvups_1", torch.nn.UpsamplingNearest2d(scale_factor=2))
		self.vid_dec.add_module("vderepups_1", torch.nn.ReplicationPad2d(1))
		self.vid_dec.add_module("vdeconv_1", torch.nn.Conv2d(6, 3, 5, 1))
		self.vid_dec.add_module("vderelu_1", torch.nn.Sigmoid())

	def addMaskDecoderLayers(self):
		# p_r decoder
		self.p_dec = torch.nn.Sequential()
		self.p_dec.add_module("fc5_p", torch.nn.Linear(self.shape_dim+45, 200))
		self.p_dec.add_module("dropout_5", torch.nn.Dropout(0.2))
		self.p_dec.add_module("relu_5_p", torch.nn.ReLU())
		self.p_dec.add_module("fc6", torch.nn.Linear(200, 200))
		self.p_dec.add_module("relu_6", torch.nn.ReLU())
		self.p_dec.add_module("fc61", torch.nn.Linear(200, 200))
		self.p_dec.add_module("relu_61", torch.nn.ReLU())
		self.p_dec.add_module("fcouy_p", torch.nn.Linear(200, 20))

		# v decoder
		self.v_dec = torch.nn.Sequential()
		self.v_dec.add_module("fc7_v", torch.nn.Linear(self.shape_dim+45, 200))
		self.v_dec.add_module("dropout_7", torch.nn.Dropout(0.2))
		self.v_dec.add_module("relu_7_v", torch.nn.ReLU())
		self.v_dec.add_module("fc8", torch.nn.Linear(200, 200))
		self.v_dec.add_module("relu_70", torch.nn.ReLU())
		self.v_dec.add_module("fc800", torch.nn.Linear(200, 200))
		self.v_dec.add_module("relu_91", torch.nn.ReLU())
		self.v_dec.add_module("fcout_v", torch.nn.Linear(200, 40))

		# fc decoder
		self.fc_dec = torch.nn.Sequential()
		self.fc_dec.add_module("fc10_fc", torch.nn.Linear(85, 200))
		self.fc_dec.add_module("fc_drop1", torch.nn.Dropout(0.2))
		self.fc_dec.add_module("relu_11_fc", torch.nn.ReLU())
		self.fc_dec.add_module("fc12", torch.nn.Linear(200, 200))
		self.fc_dec.add_module("relu_12", torch.nn.ReLU())
		self.fc_dec.add_module("fc120", torch.nn.Linear(200, 200))
		self.fc_dec.add_module("relu_120", torch.nn.ReLU())
		self.fc_dec.add_module("fc12001_fc", torch.nn.Linear(200, 40))

		# p_e decoder
		self.pe_dec = torch.nn.Sequential()
		self.pe_dec.add_module("fc13_pe", torch.nn.Linear(self.shape_dim+45, 200))
		self.pe_dec.add_module("dropout_13", torch.nn.Dropout(0.2))
		self.pe_dec.add_module("relu_14_pe", torch.nn.ReLU())
		self.pe_dec.add_module("fc14", torch.nn.Linear(200, 200))
		self.pe_dec.add_module("relu_14", torch.nn.ReLU())
		self.pe_dec.add_module("fc140", torch.nn.Linear(200, 200))
		self.pe_dec.add_module("relu_140", torch.nn.ReLU())
		self.pe_dec.add_module("fc14000_pe", torch.nn.Linear(200, 20))

		# fc_e decoder
		self.fce_dec = torch.nn.Sequential()
		self.fce_dec.add_module("fc16_fce", torch.nn.Linear(65, 200))
		self.fce_dec.add_module("dropout_16", torch.nn.Dropout(0.2))
		self.fce_dec.add_module("relu_16_fce", torch.nn.ReLU())
		self.fce_dec.add_module("fc17", torch.nn.Linear(200, 200))
		self.fce_dec.add_module("relu_17", torch.nn.ReLU())
		self.fce_dec.add_module("fc170", torch.nn.Linear(200, 200))
		self.fce_dec.add_module("relu_170", torch.nn.ReLU())
		self.fce_dec.add_module("fc17001_fce", torch.nn.Linear(200, 40))

	def addVideoLayers(self):
		# lstm layers
		self.videoRNN = torch.nn.LSTM(input_size=self.frame_dim, hidden_size=self.rnn_dim//2, num_layers=5, bidirectional = True, batch_first=True)
		
		# rnn fully connected layers
		self.fcRNN = torch.nn.Sequential()
		self.fcRNN.add_module("rnn_fc1", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.fcRNN.add_module("rnn_relu", torch.nn.ReLU())
		self.fcRNN.add_module("rnn_fc2", torch.nn.Linear(self.rnn_dim, self.rnn_dim))

		self.re_embed = torch.nn.Sequential()
		self.re_embed.add_module("reemb_fc1", torch.nn.Linear(self.rnn_dim + 9*self.T + self.shape_dim + 1, self.rnn_dim + 9*self.T + self.shape_dim + 1))
		self.re_embed.add_module("reemb_batch_norm1", torch.nn.BatchNorm1d(self.rnn_dim + 9*self.T + self.shape_dim + 1))
		self.re_embed.add_module("reemb_relu15", torch.nn.ReLU())
		self.re_embed.add_module("reemb_fc_out", torch.nn.Linear(self.rnn_dim + 9*self.T + self.shape_dim + 1, self.rnn_dim + 9*self.T + self.shape_dim))

		# decodes the pose of the object
		self.traj_dec = torch.nn.Sequential()
		self.traj_dec.add_module("traj_fc_dc1", torch.nn.Linear(self.T*self.rnn_dim, self.rnn_dim))
		self.traj_dec.add_module("traj_bn1", torch.nn.BatchNorm1d(self.rnn_dim))
		self.traj_dec.add_module("traj_relu_dc_1", torch.nn.ReLU())
		self.traj_dec.add_module("traj_drop_1", torch.nn.Dropout(0.2))
		self.traj_dec.add_module("traj_fc_dc2", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.traj_dec.add_module("traj_bn2", torch.nn.BatchNorm1d(self.rnn_dim))
		self.traj_dec.add_module("traj_relu_dc_2", torch.nn.ReLU())
		self.traj_dec.add_module("traj_fc_dc3", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.traj_dec.add_module("traj_bn3", torch.nn.BatchNorm1d(self.rnn_dim))
		self.traj_dec.add_module("traj_relu_dc_3", torch.nn.ReLU())
		self.traj_dec.add_module("traj_fc_dcout", torch.nn.Linear(self.rnn_dim, 9*self.T))

		# decodes the shape of the object
		self.shap_dec = torch.nn.Sequential()
		self.shap_dec.add_module("shap_fc_sc1", torch.nn.Linear(self.T*self.rnn_dim, self.rnn_dim))
		self.shap_dec.add_module("shap_bn1", torch.nn.BatchNorm1d(self.rnn_dim))
		self.shap_dec.add_module("shap_relu_sc_1", torch.nn.ReLU())
		self.shap_dec.add_module("shap_drop_1", torch.nn.Dropout(0.2))
		self.shap_dec.add_module("shap_fc_sc2", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.shap_dec.add_module("shap_bn2", torch.nn.BatchNorm1d(self.rnn_dim))
		self.shap_dec.add_module("shap_relu_sc_2", torch.nn.ReLU())
		self.shap_dec.add_module("shap_fc_sc3", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.shap_dec.add_module("shap_bn3", torch.nn.BatchNorm1d(self.rnn_dim))
		self.shap_dec.add_module("shap_relu_sc_3", torch.nn.ReLU())
		self.shap_dec.add_module("shap_fc_out", torch.nn.Linear(self.rnn_dim, self.shape_dim))

		# decodes the shape of the object
		self.env_dec = torch.nn.Sequential()
		self.env_dec.add_module("env_fc_sc1", torch.nn.Linear(self.T*self.rnn_dim, self.rnn_dim))
		self.env_dec.add_module("env_bn1", torch.nn.BatchNorm1d(self.rnn_dim))
		self.env_dec.add_module("env_relu_sc_1", torch.nn.ReLU())
		self.env_dec.add_module("env_drop_1", torch.nn.Dropout(0.2))
		self.env_dec.add_module("env_fc_sc2", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.env_dec.add_module("env_bn2", torch.nn.BatchNorm1d(self.rnn_dim))
		self.env_dec.add_module("env_relu_sc_2", torch.nn.ReLU())
		self.env_dec.add_module("env_fc_sc3", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.env_dec.add_module("env_bn3", torch.nn.BatchNorm1d(self.rnn_dim))
		self.env_dec.add_module("env_relu_sc_3", torch.nn.ReLU())
		self.env_dec.add_module("env_fc_out", torch.nn.Linear(self.rnn_dim,self.shape_dim))

		# p_r decoder
		self.vid_p_dec = torch.nn.Sequential()
		self.vid_p_dec.add_module("p_fc1", torch.nn.Linear(self.rnn_dim + 9*self.T + self.shape_dim, self.rnn_dim))
		self.vid_p_dec.add_module("p_bn1", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_p_dec.add_module("p_relu1", torch.nn.ReLU())
		self.vid_p_dec.add_module("p_drop_1", torch.nn.Dropout(0.2))
		self.vid_p_dec.add_module("p_fc2", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_p_dec.add_module("p_bn2", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_p_dec.add_module("p_relu2", torch.nn.ReLU())
		self.vid_p_dec.add_module("p_fc3", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_p_dec.add_module("p_bn3", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_p_dec.add_module("p_relu3", torch.nn.ReLU())
		self.vid_p_dec.add_module("p_fc_out", torch.nn.Linear(self.rnn_dim, 20))

		# v decoder
		self.vid_v_dec = torch.nn.Sequential()
		self.vid_v_dec.add_module("v_fc1", torch.nn.Linear(self.rnn_dim + 9*self.T + self.shape_dim, self.rnn_dim))
		self.vid_v_dec.add_module("v_bn1", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_v_dec.add_module("v_relu1", torch.nn.ReLU())
		self.vid_v_dec.add_module("v_drop_1", torch.nn.Dropout(0.2))
		self.vid_v_dec.add_module("v_fc2", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_v_dec.add_module("v_bn2", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_v_dec.add_module("v_relu2", torch.nn.ReLU())
		self.vid_v_dec.add_module("v_fc3", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_v_dec.add_module("v_bn3", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_v_dec.add_module("v_relu3", torch.nn.ReLU())
		self.vid_v_dec.add_module("v_fc_out", torch.nn.Linear(self.rnn_dim, 40))

		# fc decoder
		self.vid_fc_dec = torch.nn.Sequential()
		self.vid_fc_dec.add_module("fc_fc1", torch.nn.Linear(self.rnn_dim + 9*self.T + self.shape_dim, self.rnn_dim))
		self.vid_fc_dec.add_module("fc_bn1", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_fc_dec.add_module("fc_relu_1", torch.nn.ReLU())
		self.vid_fc_dec.add_module("fc_drop_1", torch.nn.Dropout(0.2))
		self.vid_fc_dec.add_module("fc_fc2", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_fc_dec.add_module("fc_bn2", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_fc_dec.add_module("fc_relu_2", torch.nn.ReLU())
		self.vid_fc_dec.add_module("fc_fc3", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_fc_dec.add_module("fc_bn3", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_fc_dec.add_module("fc_relu_3", torch.nn.ReLU())
		self.vid_fc_dec.add_module("fc_fc_out", torch.nn.Linear(self.rnn_dim, 40))

		# p_e decoder
		self.vid_pe_dec = torch.nn.Sequential()
		self.vid_pe_dec.add_module("pe_fc1", torch.nn.Linear(self.rnn_dim + 9*self.T + self.shape_dim, self.rnn_dim))
		self.vid_pe_dec.add_module("pe_bn1", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_pe_dec.add_module("pe_relu1", torch.nn.ReLU())
		self.vid_pe_dec.add_module("pe_drop_1", torch.nn.Dropout(0.2))
		self.vid_pe_dec.add_module("pe_fc2", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_pe_dec.add_module("pe_bn2", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_pe_dec.add_module("pe_relu2", torch.nn.ReLU())
		self.vid_pe_dec.add_module("pe_fc3", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_pe_dec.add_module("pe_bn3", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_pe_dec.add_module("pe_relu3", torch.nn.ReLU())
		self.vid_pe_dec.add_module("pe_fc_out", torch.nn.Linear(self.rnn_dim, 20))

		# fc_e decoder
		self.vid_fce_dec = torch.nn.Sequential()
		self.vid_fce_dec.add_module("fce_fc1", torch.nn.Linear(self.rnn_dim + 9*self.T + self.shape_dim, self.rnn_dim))
		self.vid_fce_dec.add_module("fce_bn1", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_fce_dec.add_module("fce_relu1", torch.nn.ReLU())
		self.vid_fce_dec.add_module("fce_drop_1", torch.nn.Dropout(0.2))
		self.vid_fce_dec.add_module("fce_fc2", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_fce_dec.add_module("fce_bn2", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_fce_dec.add_module("fce_relu2", torch.nn.ReLU())
		self.vid_fce_dec.add_module("fce_fc3", torch.nn.Linear(self.rnn_dim, self.rnn_dim))
		self.vid_fce_dec.add_module("fce_bn3", torch.nn.BatchNorm1d(self.rnn_dim))
		self.vid_fce_dec.add_module("fce_relu3", torch.nn.ReLU())
		self.vid_fce_dec.add_module("fce_fc_out", torch.nn.Linear(self.rnn_dim, 40))

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

			# friction cone constraints
			for c in range(self.N_c):
				constraints.append(gamma[c,t]*fc[c*4,t]	 + gamma[c + self.N_c,t]*fc[c*4 + 2,t] == f[c,t])
				constraints.append(gamma[c,t]*fc[c*4 + 1,t] + gamma[c + self.N_c,t]*fc[c*4 + 3,t] == f[self.N_c + c,t])
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

		objective = cp.Minimize(w_err*cp.pnorm(err, p=2) + w_f*cp.pnorm(f, p=2) + w_err*cp.pnorm(p - p_r, p=2))
		problem = cp.Problem(objective, constraints)
		
		return CvxpyLayer(problem, parameters=[r, ddr, fc, p_e, fc_e, v, p_r], variables=[p, f, f_e, alpha1, alpha2, gamma, gamma_e, err])

	def setupCTOSag(self, w_err = 0.01, w_f = 1):
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
			constraints.append(sum(f[self.N_c:,t]) + f_e[2,t] + f_e[3,t] - 9.8 == ddr[1,t] + err[1,t])

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
			constraints.append(gamma_e[0,t] >= 0)
			constraints.append(gamma_e[1,t] >= 0)

			constraints.append(gamma_e[2,t]*fc_e[4,t] + gamma_e[3,t]*fc_e[6,t] == f_e[1,t])
			constraints.append(gamma_e[2,t]*fc_e[5,t] + gamma_e[3,t]*fc_e[7,t] == f_e[3,t])
			constraints.append(gamma_e[2,t] >= 0)
			constraints.append(gamma_e[3,t] >= 0)

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

	########################	
	# Forward Pass Methods #
	########################

	def forwardCTO(self, xtraj, x, x_img, pass_soft = False): 
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
			if pass_soft:
				v = v0

			# fc = fc0
			# fc_e = fc_e0

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
			
			# pdb.set_trace()


			if first:				
				y = torch.cat([p.view(1,-1), f.view(1,-1)], axis = 1)
				first = False
			else:
				y_1 = torch.cat([p.view(1,-1), f.view(1,-1)], axis = 1)
				y = torch.cat((y, y_1), axis = 0)
		return y

	def forward2Sim(self, xtraj, x, x_img, polygons, render=False, bypass = False, pass_soft = False): 
		# passes through the optimization problem
		first = True
	
		# shape encoding		
		e_img = self.forwardShapeEncoder(np.reshape(x_img[:,0:2500],(-1,1,50,50)))
		
		# # param decoding
		p_r = self.p_dec.forward(torch.cat((e_img.view(-1,self.shape_dim), xtraj.view(-1,45)), 1))
		v = self.v_dec.forward(torch.cat((e_img.view(-1,self.shape_dim), xtraj.view(-1,45)), 1))
		p_e = self.pe_dec.forward(torch.cat([e_img.view(-1,self.shape_dim), xtraj.view(-1,45)], 1))
		fc = self.fc_dec.forward(torch.cat([xtraj.view(-1,45), v.view(-1,40)], 1))
		fc_e = self.fce_dec.forward(torch.cat([p_e.view(-1,20), xtraj.view(-1,45)], 1))

		# p_e = x[:,60:80] # external contact location
		if pass_soft:
			v = x[:,120:160] # contact affordance
		# p_r = x[:,0:20]
		# fc = x[:,20:60]
		# fc_e = x[:,80:120]

		# decodes each trajectory
		for i in range(np.shape(x)[0]):
			# params that should be obtained from video
			r = xtraj[i,:15]
			# dr = xtraj[i,15:30]
			# ddr = xtraj[i,30:]

			# learnes the parameters
			dr, ddr = self.Difflayer3(r.view(3,5))

			# solves for the contact trajectory
			if bypass:	
				p = self.cto_su1(p_r[i,:].view(1,-1)).view(4,5)
				f = self.cto_su2(p_r[i,:].view(1,-1)).view(4,5)
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
			
			p = torch.clamp(p, -0.2, 0.2)

			solve = self.sim()
			xtraj_new = solve.apply(p.view(1,-1).double(), polygons[i,:].double(), r.view(3,5)[:,0].double(), render, self.gamma)
			dp, ddp = self.Difflayer3(xtraj_new.view(3,5))

			if i == 0:				
				y = xtraj_new.view(1,-1)
			else:
				y1 = xtraj_new.view(1,-1)
				y = torch.cat((y, y1), axis = 0)
		return y

	def forwardVideo(self, video, xtraj, x, bypass = False, pass_soft = False):
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
		# xtraj = self.traj_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))

		embedding = torch.cat((e_vid, e_img, xtraj), axis = 1)

		if self.sagittal:
			embedding = self.re_embed(torch.cat( (embedding, torch.tensor(np.ones((np.shape(video)[0],1))).float()) , axis = 1))
		else:
			embedding = self.re_embed(torch.cat((embedding,0*torch.tensor(np.ones((np.shape(video)[0],1))).float()), axis = 1))

		p_r = self.vid_p_dec.forward(embedding)
		v = self.vid_v_dec.forward(embedding)
		fc = self.vid_fc_dec.forward(embedding)
		p_e = self.vid_pe_dec.forward(embedding)
		fc_e = self.vid_fce_dec.forward(embedding)

		# v = x[:,120:160]

		if pass_soft:
			p_r = x[:,0:20]
			v = x[:,120:160]
			p_e = x[:,60:80]
			fc = x[:,20:60]
			fc_e = x[:,80:120]

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

			# learns the parameters
			# dr, ddr = self.Difflayer3(r.view(3,5))

			# solves for the contact trajectory
			if bypass:
				p = self.cto_su1(p_r[i,:].view(1,-1)).view(4,5)
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
				y = p.view(1,-1)
			else:
				y = torch.cat((y, p.view(1,-1)), axis = 0)
		# self.tol = self.tol/1.1
		return y

	def forwardVideotoCVX(self, video, xtraj, x, bypass = False, pass_soft = False):
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
		# xtraj = self.traj_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))

		embedding = torch.cat((e_vid, e_img, xtraj), axis = 1)

		if self.sagittal:
			embedding = self.re_embed(torch.cat( (embedding, torch.tensor(np.ones((np.shape(video)[0],1))).float()) , axis = 1))
		else:
			embedding = self.re_embed(torch.cat((embedding,0*torch.tensor(np.ones((np.shape(video)[0],1))).float()), axis = 1))

		p_r = self.vid_p_dec.forward(embedding)
		v = self.vid_v_dec.forward(embedding)
		fc = self.vid_fc_dec.forward(embedding)
		p_e = self.vid_pe_dec.forward(embedding)
		fc_e = self.vid_fce_dec.forward(embedding)

		# v = x[:,120:160]

		if pass_soft:
			p_r = x[:,0:20]
			v = x[:,120:160]
			p_e = x[:,60:80]
			fc = x[:,20:60]
			fc_e = x[:,80:120]

		# self.w = 0.01
		# self.CTOlayer = self.setupCTO(w_err = self.w)

		bypass = False

		# decodes each trajectory
		# print('going through cvx')
		for i in range(np.shape(e_frame)[0]):
			# params that should be obtained from video
			r = xtraj[i,:15]
			# params that can be computed explicity
			dr = xtraj[i,15:30]
			ddr = xtraj[i,30:]

			# learns the parameters
			# dr, ddr = self.Difflayer3(r.view(3,5))

			# solves for the contact trajectory
			if bypass:
				p = self.cto_su1(p_r[i,:].view(1,-1)).view(4,5)
			else:
				failed = False
				while True:
					try:
						p, f, _, _, _, _, _, err = self.CTOlayer(r.view(3,5), ddr.view(3,5), fc[i,:].view(8,5), p_e[i,:].view(4,5), fc_e[i,:].view(8,5), v[i,:].view(8, 5), p_r[i,:].view(4, 5))
						loss = self.w*torch.abs(err).sum() + torch.norm(f) + self.w*torch.abs(p - p_r[i,:].view(4,5)).sum()
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
				y = loss.view(1,-1)
			else:
				y = torch.cat((y, loss.view(1,-1)), axis = 0)
		# self.tol = self.tol/1.1
		return y

	def forwardEndToEnd(self, video, polygons, xtraj, render = False, bypass = False, pass_soft = False, x = []):
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
		# pdb.set_tracse()
		# e_vid = self.lstm_su(e_vid.view(-1,self.T*self.frame_dim))
		# extracts image encoding and object trajectory
		
		e_img = self.shap_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))
		# xtraj = self.traj_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))

		embedding = torch.cat((e_vid, e_img, xtraj), axis = 1)

		if self.sagittal:
			embedding = self.re_embed(torch.cat( (embedding, torch.tensor(np.ones((np.shape(video)[0],1))).float() ), axis = 1))
		else:
			embedding = self.re_embed(torch.cat((embedding,0*torch.tensor(np.ones((np.shape(video)[0],1))).float()), axis = 1))

		p_r = self.vid_p_dec.forward(embedding)
		v = self.vid_v_dec.forward(embedding)
		fc = self.vid_fc_dec.forward(embedding)
		p_e = self.vid_pe_dec.forward(embedding)
		fc_e = self.vid_fce_dec.forward(embedding)

		if pass_soft:
			# import pdb
			# pdb.set_trace()
			p_r = x[:,0:20]
			v = x[:,120:160]
			p_e = x[:,60:80]
			fc = x[:,20:60]
			fc_e = x[:,80:120]

		print('going through cvx + sim')
		# decodes each trajectory
		for i in range(np.shape(e_vid)[0]):
			# params that should be obtained from video
			r = xtraj[i,:15]
			# params that can be computed explicity
			dr = xtraj[i,15:30]
			ddr = xtraj[i,30:]

			# learnes the parameters
			# dr, ddr = self.Difflayer3(r.view(3,5))
			# print(ddr)
			# print(v)
			# solves for the contact trajectory
			
			if bypass:
				p = self.cto_su1(p_r[i,:].view(1,-1)).view(4,5)
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

			solve = self.sim()
			# print(r.view(3,5))
			xtraj_new = solve.apply(p.view(1,-1).double(), polygons[i,:].double(), r.view(3,5)[:,0].double(), render, self.gamma)

			# pdb.set_trace()

			if i == 0:				
				y = xtraj_new[:,:15].view(1,-1).float()
			else:
				y = torch.cat((y, xtraj_new[:,:15].view(1,-1).float()), axis = 0)

			if i == 0:				
				p_o = xtraj_new[:,15:].view(1,-1).float()
			else:
				p_o = torch.cat((p_o, xtraj_new[:,15:].view(1,-1).float()), axis = 0)

			if i == 0:				
				p_err = xtraj_new[:,15:].view(1,-1).float() - p.view(1,-1)
			else:
				p_err = torch.cat((p_err, xtraj_new[:,15:].view(1,-1).float() - p.view(1,-1)), axis = 0)

		return y, p_o, p_err

	#####################	
	# Parameter Methods #
	#####################

	def forwardVideotoImage(self,video):
		for t in range(self.T):
			frame = video[:,t*3*100*100:(t+1)*3*100*100]
			e_frame = self.forwardFrameCNN(frame).view(-1,1,self.frame_dim)
			if t == 0:
				e_vid = e_frame
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
		
		# passes frames through RNN
		rnn_out, (_, _) = self.videoRNN(e_vid.view(-1,self.T,self.frame_dim))
		
		# extracts image encoding
		e_img = self.shap_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))
		# decodes image
		return self.decoder(self.fc3(e_img).view(-1,20,6,6))

	def forwardVideotoEnv(self,video):
		for t in range(self.T):
			frame = video[:,t*3*100*100:(t+1)*3*100*100]
			e_frame = self.forwardFrameCNN(frame).view(-1,1,self.frame_dim)
			if t == 0:
				e_vid = e_frame
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
		
		# passes frames through RNN
		rnn_out, (_, _) = self.videoRNN(e_vid.view(-1,self.T,self.frame_dim))
		
		# extracts image encoding
		e_img = self.env_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))
		# decodes image
		return self.edecoder(self.fc4(e_img).view(-1,20,6,6)).view(-1,2500)

	def forwardVideotoShapes(self,video):
		for t in range(self.T):
			frame = video[:,t*3*100*100:(t+1)*3*100*100]
			e_frame = self.forwardFrameCNN(frame).view(-1,1,self.frame_dim)
			if t == 0:
				e_vid = e_frame
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
		
		# passes frames through RNN
		rnn_out, (_, _) = self.videoRNN(e_vid.view(-1,self.T,self.frame_dim))
		
		# extracts image encoding
		e_img = self.shap_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))
		e_shape = self.shape_to_shapes(e_img)
		for i in range(7):
			if i == 0:
				dec = self.decoder(self.fc3(e_shape[-1,i*self.shape_dim:(i+1)*self.shape_dim]).view(-1,20,6,6)).view(-1,2500)
			else:
				dec = y = torch.cat((dec, self.decoder(self.fc3(e_shape[-1,i*self.shape_dim:(i+1)*self.shape_dim]).view(-1,20,6,6)).view(-1,2500)), axis = 1)
		# decodes image
		return dec

	def forwardVideotoTraj(self,video):
		# passes through all frames
		for t in range(self.T):
			frame = video[:,t*3*100*100:(t+1)*3*100*100]
			e_frame = self.forwardFrameCNN(frame).view(-1,1,self.frame_dim)
			if t == 0:
				e_vid = e_frame
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
		
		# passes frames through RNN
		rnn_out, (_, _) = self.videoRNN(e_vid.view(-1,self.T,self.frame_dim))
		
		# extracts object trajectory
		xtraj = self.traj_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))
		return xtraj

	#####################
	# Assesment methods #
	#####################

	def forwardVideoToParams(self, video, x, xtraj):
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
		# xtraj_2 = self.traj_dec(rnn_out.contiguous().view(-1,self.T*self.rnn_dim))

		# pdb.set_trace()

		embedding = torch.cat((e_vid, e_img, xtraj), axis = 1)

		if self.sagittal:
			embedding = self.re_embed(torch.cat((embedding,torch.tensor(np.ones((np.shape(video)[0],1))).float()), axis = 1))
		else:
			embedding = self.re_embed(torch.cat((embedding,0*torch.tensor(np.ones((np.shape(video)[0],1))).float()), axis = 1))

		# extracts the parameters
		p_r = self.vid_p_dec.forward(embedding)
		v = self.vid_v_dec.forward(embedding)
		fc = self.vid_fc_dec.forward(embedding)
		p_e = self.vid_pe_dec.forward(embedding)
		fc_e = self.vid_fce_dec.forward(embedding)

		# params that can be learned from above
		p_r0 = x[:,0:20]
		v0 = x[:,120:160]
		p_e0 = x[:,60:80]
		fc0 = x[:,20:60]
		fc_e0 = x[:,80:120]

		return p_r-p_r0, v-v0, p_e-p_e0, fc-fc0, fc_e-fc_e0

	def forwardMaskToParams(self, xtraj, x, x_img): 
		# passes through the optimization problem
		first = True
	
		# shape encoding		
		e_img = self.forwardShapeEncoder(np.reshape(x_img[:,0:2500],(-1,1,50,50)))
		
		# # param decoding
		p_r = self.p_dec.forward(torch.cat((e_img.view(-1,self.shape_dim), xtraj.view(-1,45)), 1))
		v = self.v_dec.forward(torch.cat((e_img.view(-1,self.shape_dim), xtraj.view(-1,45)), 1))
		p_e = self.pe_dec.forward(torch.cat([e_img.view(-1,self.shape_dim), xtraj.view(-1,45)], 1))
		fc = self.fc_dec.forward(torch.cat([xtraj.view(-1,45), v.view(-1,40)], 1))
		fc_e = self.fce_dec.forward(torch.cat([p_e.view(-1,20), xtraj.view(-1,45)], 1))

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
		return mu

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
		return mu

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
		y = self.forwardCTO(torch.tensor(inputs_1).float(),torch.tensor(inputs_2).float(),torch.tensor(inputs_img).float())
		y = y.clone().detach()
		np.savetxt("../data/"+name+"_0_2f.csv", y.data.numpy(), delimiter=",")

	def gen_res_sim(self,inputs_1,inputs_2,inputs_img,polygons,name="res",pass_=False):

		y = self.forwardCTO(torch.tensor(inputs_1).float(),torch.tensor(inputs_2).float(),torch.tensor(inputs_img).float(), pass_soft = pass_)
		y = y.clone().detach()

		r = self.forward2Sim(torch.tensor(inputs_1).float(),torch.tensor(inputs_2).float(),torch.tensor(inputs_img).float(),torch.tensor(polygons).float(), pass_soft = pass_, bypass = False)
		r = r.clone().detach()

		np.savetxt("../data/sim_p_"+name+"_0_2f.csv", y.data.numpy(), delimiter=",")
		np.savetxt("../data/sim_r_"+name+"_0_2f.csv", r.data.numpy(), delimiter=",")

	def genResults(self,vids,polygons,inputs_1,inputs_2,name="res", bypass = False, pass_=False):
		r, p, err = self.forwardEndToEnd(torch.tensor(vids).float(), torch.tensor(polygons).float(), inputs_1.float(), render=False, bypass = bypass, pass_soft = pass_, x = inputs_2.float())
		r = r.clone().detach()

		# pdb.set_trace()

		np.savetxt("../data/sim_p_"+name+"_0_2f.csv", p.data.numpy(), delimiter=",")
		np.savetxt("../data/sim_r_"+name+"_0_2f.csv", r.data.numpy(), delimiter=",")

	def gen_resVid(self, vids, x, xtraj, name="res"):
		y = self.forwardVideo(torch.tensor(vids).float(), x.float(), xtraj.float(), bypass = False)
		y = y.clone().detach()
		np.savetxt("../data/"+name+"_0_2f.csv", y.data.numpy(), delimiter=",")

	def gen_resVidSim(self, vids, x, xtraj, name="res"):
		y = self.forwardVideo(torch.tensor(vids).float(), x.float(), xtraj.float(), bypass = False)
		y = y.clone().detach()
		np.savetxt("../data/"+name+"_0_2f.csv", y.data.numpy(), delimiter=",")
