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
from lcp_physics.physics.forces import Gravity, MDP
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
		self.shape_dim = 360
		self.frame_dim = 1500

		# CTO layers
		self.CTOlayer = self.setupCTO()
		self.Difflayer = self.setupDiff(dims=4, T = 6)
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
		self.encoder.add_module("fc_enc", torch.nn.Linear(720,720))

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
		self.vid_enc.add_module("vid_fc_enc", torch.nn.Linear(3456,3456))
		self.vid_enc.add_module("vid_relu_4", torch.nn.LeakyReLU(0.2))

		# layers for frame VAE
		self.vid_fc1 = torch.nn.Linear(3456, self.frame_dim)
		self.vid_fc2 = torch.nn.Linear(3456, self.frame_dim)
		self.vid_fc3 = torch.nn.Linear(self.frame_dim, 3456)

		self.vid_dec0 = torch.nn.Sequential()
		self.vid_dec0.add_module("vid_relu_dec4", torch.nn.LeakyReLU(0.2))
		self.vid_dec0.add_module("vid_fc_dec4", torch.nn.Linear(3456,3456))
		self.vid_dec0.add_module("vid_relu_dec5", torch.nn.LeakyReLU(0.2))

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
		# decodes the pose of the object
		self.traj_dec = torch.nn.Sequential()
		self.traj_dec.add_module("vpfc_dc1", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.traj_dec.add_module("vprelu_dc_1", torch.nn.ReLU())
		self.traj_dec.add_module("vpfc_dc2", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.traj_dec.add_module("vprelu_dc_2", torch.nn.ReLU())
		self.traj_dec.add_module("vfc_dc3", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.traj_dec.add_module("vprelu_dc_3", torch.nn.ReLU())
		self.traj_dec.add_module("vfc_dc4", torch.nn.Linear(self.frame_dim*self.T, 600*self.T))
		self.traj_dec.add_module("vprelu_dc_4", torch.nn.ReLU())
		self.traj_dec.add_module("vpfc_dc5", torch.nn.Linear(600*self.T, 200*self.T))
		self.traj_dec.add_module("vrelu_dc_5", torch.nn.ReLU())
		self.traj_dec.add_module("vfc_dc6", torch.nn.Linear(200*self.T, 200*self.T))
		self.traj_dec.add_module("vrelu_dc_6", torch.nn.ReLU())
		self.traj_dec.add_module("vfc_dc6", torch.nn.Linear(200*self.T, 200*self.T))
		self.traj_dec.add_module("vrelu_dc_6", torch.nn.ReLU())
		self.traj_dec.add_module("vfc_dc7", torch.nn.Linear(200*self.T, 9*self.T))
		self.traj_dec.add_module("vfc_dc8", torch.nn.Linear(9*self.T, 9*self.T))

		# decodes the shape of the object
		self.shap_dec = torch.nn.Sequential()
		self.shap_dec.add_module("fc_sc1", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.shap_dec.add_module("relu_sc_1", torch.nn.LeakyReLU(0.2))
		self.shap_dec.add_module("fc_sc2", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.shap_dec.add_module("relu_sc_2", torch.nn.LeakyReLU(0.2))
		self.shap_dec.add_module("fc_sc3", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.shap_dec.add_module("relu_sc_3", torch.nn.LeakyReLU(0.2))
		self.shap_dec.add_module("fc_sc4", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.shap_dec.add_module("relu_sc_4", torch.nn.LeakyReLU(0.2))
		self.shap_dec.add_module("fc_sc5", torch.nn.Linear(self.frame_dim*self.T, 500*self.T))
		self.shap_dec.add_module("relu_sc_5", torch.nn.LeakyReLU(0.2))
		self.shap_dec.add_module("fc_sc6", torch.nn.Linear(500*self.T, 500*self.T))
		self.shap_dec.add_module("relu_sc_6", torch.nn.LeakyReLU(0.2))
		self.shap_dec.add_module("fc_sc7", torch.nn.Linear(500*self.T, 500*self.T))
		self.shap_dec.add_module("relu_sc_7", torch.nn.LeakyReLU(0.2))
		self.shap_dec.add_module("fc_sc8", torch.nn.Linear(500*self.T, 100*self.T))
		self.shap_dec.add_module("relu_sc_8", torch.nn.LeakyReLU(0.2))
		self.shap_dec.add_module("fc_sc9x", torch.nn.Linear(100*self.T,360))
		# p_r decoder
		self.vid_p_dec = torch.nn.Sequential()
		self.vid_p_dec.add_module("fcvp5", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		# self.p_dec.add_module("dropout_5", torch.nn.Dropout(0.2))
		self.vid_p_dec.add_module("relu_vp5", torch.nn.ReLU())
		self.vid_p_dec.add_module("fcvp6", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_p_dec.add_module("relu_v6", torch.nn.ReLU())
		self.vid_p_dec.add_module("fcvp61", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_p_dec.add_module("relu_vp61", torch.nn.ReLU())
		self.vid_p_dec.add_module("fcvp611", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_p_dec.add_module("vprelu_611", torch.nn.ReLU())
		self.vid_p_dec.add_module("vpfc6112", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_p_dec.add_module("vprelu_6112", torch.nn.ReLU())
		self.vid_p_dec.add_module("vpfc612", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_p_dec.add_module("vprelu_612", torch.nn.ReLU())
		self.vid_p_dec.add_module("vpfc62", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim))
		self.vid_p_dec.add_module("vprelu_62", torch.nn.ReLU())
		self.vid_p_dec.add_module("vpfc63", torch.nn.Linear(self.frame_dim, 200))
		self.vid_p_dec.add_module("vprelu_63", torch.nn.ReLU())
		self.vid_p_dec.add_module("vpfc60", torch.nn.Linear(200, 20))

		# v decoder
		self.vid_v_dec = torch.nn.Sequential()
		self.vid_v_dec.add_module("vpfc7", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_v_dec.add_module("rvpelu_7", torch.nn.ReLU())
		self.vid_v_dec.add_module("vpfc8", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_v_dec.add_module("vprelu_70", torch.nn.ReLU())
		self.vid_v_dec.add_module("vpfc80", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_v_dec.add_module("vprelu_700", torch.nn.ReLU())
		self.vid_v_dec.add_module("vfc800", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_v_dec.add_module("vrelu_8", torch.nn.ReLU())
		self.vid_v_dec.add_module("vpfc9", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_v_dec.add_module("vprelu_9", torch.nn.ReLU())
		self.vid_v_dec.add_module("vpfc90", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_v_dec.add_module("vprelu_90", torch.nn.ReLU())
		self.vid_v_dec.add_module("vpfc901", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_v_dec.add_module("vprelu_901", torch.nn.ReLU())
		self.vid_v_dec.add_module("vpfc902", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_v_dec.add_module("vprelu_902", torch.nn.ReLU())
		self.vid_v_dec.add_module("vpfc91", torch.nn.Linear(self.frame_dim*self.T, 200))
		self.vid_v_dec.add_module("vprelu_91", torch.nn.ReLU())
		self.vid_v_dec.add_module("vpfc90c", torch.nn.Linear(200, 40))

		# fc decoder
		self.vid_fc_dec = torch.nn.Sequential()
		self.vid_fc_dec.add_module("vfc10", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		# self.fc_dec.add_module("dropout_10", torch.nn.Dropout(0.2))
		# self.fc_dec.add_module("bn11", torch.nn.BatchNorm1d(num_features=200))
		self.vid_fc_dec.add_module("rvelu_11", torch.nn.LeakyReLU(0.2))
		self.vid_fc_dec.add_module("fvc12", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_fc_dec.add_module("revlu_12", torch.nn.ReLU())
		self.vid_fc_dec.add_module("fc1v20", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_fc_dec.add_module("revluv_120", torch.nn.LeakyReLU(0.2))
		self.vid_fc_dec.add_module("fcv1200", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_fc_dec.add_module("revlu_1200", torch.nn.ReLU())
		self.vid_fc_dec.add_module("fcv1201", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_fc_dec.add_module("revlu_1201", torch.nn.LeakyReLU(0.2))
		self.vid_fc_dec.add_module("fcv1202", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_fc_dec.add_module("revvlu_1202", torch.nn.ReLU())
		self.vid_fc_dec.add_module("fcv1203", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_fc_dec.add_module("revvlu_1203", torch.nn.LeakyReLU(0.2))
		self.vid_fc_dec.add_module("fcv1204", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_fc_dec.add_module("revvlu_1204", torch.nn.ReLU())
		self.vid_fc_dec.add_module("fcv1205", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_fc_dec.add_module("revvlu_1205", torch.nn.LeakyReLU(0.2))
		self.vid_fc_dec.add_module("fc12000", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim))
		self.vid_fc_dec.add_module("revvlu_12000", torch.nn.ReLU())
		self.vid_fc_dec.add_module("fc1v2001", torch.nn.Linear(self.frame_dim, 40))

		# p_e decoder
		self.vid_pe_dec = torch.nn.Sequential()
		self.vid_pe_dec.add_module("fcc13", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		# self.pe_dec.add_module("dropout_13", torch.nn.Dropout(0.3))
		self.vid_pe_dec.add_module("rcelu_14", torch.nn.ReLU())
		self.vid_pe_dec.add_module("fcc14", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_pe_dec.add_module("reclu_14", torch.nn.ReLU())
		self.vid_pe_dec.add_module("fcc140", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_pe_dec.add_module("reclu_140", torch.nn.ReLU())
		self.vid_pe_dec.add_module("fcv1401", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_pe_dec.add_module("revlu_1401", torch.nn.ReLU())
		self.vid_pe_dec.add_module("fcv1402", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_pe_dec.add_module("revlu_1402", torch.nn.ReLU())
		self.vid_pe_dec.add_module("fc1400", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim))
		# self.pe_dec.add_module("bn1vv4", torch.nn.BatchNorm1d(num_features=200))
		self.vid_pe_dec.add_module("revlu_1400", torch.nn.ReLU())
		self.vid_pe_dec.add_module("fcv14000x", torch.nn.Linear(self.frame_dim, 20))

		# fc_e decoder
		self.vid_fce_dec = torch.nn.Sequential()
		self.vid_fce_dec.add_module("vfc16", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		# self.fce_dec.add_module("dropout_16", torch.nn.Dropout(0.3))
		self.vid_fce_dec.add_module("vrelu_16", torch.nn.ReLU())
		self.vid_fce_dec.add_module("vfc17", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_fce_dec.add_module("vrelu_17", torch.nn.ReLU())
		self.vid_fce_dec.add_module("vfc170", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_fce_dec.add_module("vrelu_170", torch.nn.ReLU())
		self.vid_fce_dec.add_module("vfc1700", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_fce_dec.add_module("rvelu_1700", torch.nn.ReLU())
		self.vid_fce_dec.add_module("vfc17001", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_fce_dec.add_module("vrelu_17001", torch.nn.ReLU())
		self.vid_fce_dec.add_module("vfc17002", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim*self.T))
		self.vid_fce_dec.add_module("vrelu_17002", torch.nn.ReLU())
		self.vid_fce_dec.add_module("vfc17000", torch.nn.Linear(self.frame_dim*self.T, self.frame_dim))
		self.vid_fce_dec.add_module("vrelu_17000", torch.nn.ReLU())
		self.vid_fce_dec.add_module("vfc17x001", torch.nn.Linear(self.frame_dim, 40))

	#######################################	
	# Differentiable Optimization Methods #
	#######################################

	def setupCTO(self):
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
			for c in range(self.N_c):
				if t == 0:
					constraints.append(ddp[c,t]*self.dt**2 == p[c,t] - 2*p[c,t] + p[c,t+1])
					constraints.append(ddp[c+self.N_c,t]*self.dt**2 == p[c+self.N_c,t] - 2*p[c+self.N_c,t] + p[c+self.N_c,t+1])
				elif t == self.T-1:
					constraints.append(ddp[c,t]*self.dt**2 == p[c,t-1] - 2*p[c,t] + p[c,t])
					constraints.append(ddp[c+self.N_c,t]*self.dt**2 == p[c+self.N_c,t-1] - 2*p[c+self.N_c,t] + p[c+self.N_c,t])
				else:
					constraints.append(ddp[c,t]*self.dt**2 == p[c,t-1] - 2*p[c,t] + p[c,t+1])
					constraints.append(ddp[c+self.N_c,t]*self.dt**2 == p[c+self.N_c,t-1] - 2*p[c+self.N_c,t] + p[c+self.N_c,t+1])

				# constraints.append(ddp[c,t] <= 50)
				# constraints.append(ddp[c+self.N_c,t] <= 50)

				# constraints.append(ddp[c,t] >= -50)
				# constraints.append(ddp[c+self.N_c,t] >= -50)

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
				constraints.append(p[c,t]          == alpha1[c,t]*v[c*4,t]     + alpha2[c,t]*v[c*4 + 2,t])
				constraints.append(p[c+self.N_c,t] == alpha1[c,t]*v[c*4 + 1,t] + alpha2[c,t]*v[c*4 + 3,t])
				constraints.append(alpha1[c,t] + alpha2[c,t] == 1)
				constraints.append(alpha1[c,t] >= 0)
				constraints.append(alpha2[c,t] >= 0)
				# if t < 4:
					# constraints.append(p[c,t] == p_r[c,t])
					# constraints.append(p[c+self.N_c,t] == p_r[c+self.N_c,t])

			# friction cone constraints
			for c in range(self.N_c):
				constraints.append(gamma[c,t]*fc[c*4,t]     + gamma[c + self.N_c,t]*fc[c*4 + 2,t] == f[c,t])
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

		objective = cp.Minimize(cp.pnorm(err, p=2) + cp.pnorm(f, p=2))
		problem = cp.Problem(objective, constraints)
		
		return CvxpyLayer(problem, parameters=[r, ddr, fc, p_e, fc_e, v, p_r], variables=[p, f, f_e, alpha1, alpha2, gamma, gamma_e, err, ddp])

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

	def forwardPlanarDiffSim(self, p, dp, polygon, xtraj0, rad):

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
		r0 = Hull([xr, yr], verts, restitution=0, fric_coeff=1, mass = 0.5/(n_pol+1), name="obj")
		r0.add_force(MDP(g=10000))
		bodies.append(r0)

		for i in range(n_pol):
			x2 = [polygon[1+8*i], -polygon[2+8*i]]
			x1 = [polygon[3+8*i], -polygon[4+8*i]]
			x0 = [polygon[5+8*i], -polygon[6+8*i]]
			verts = scale*np.array([x0, x1, x2])
			p0 = np.array([xr + polygon[7+8*i], yr - polygon[8+8*i]])
			r1 = Hull(p0, verts, restitution=restitution, mass = 0.5/(n_pol+1), fric_coeff=1, name="obj_"+str(i))
			r1.add_force(MDP(g=10000))
			# r1.add_force(Gravity(g=100))
			bodies.append(r1)
			joints += [FixedJoint(r1, bodies[0])]
			r1.add_no_contact(bodies[0])

		# Point Fingers
		traj_f = []
		for i in range(self.N_c):
			pos0 = [500+scale*p[i,0],500-scale*p[i+self.N_c,0]]
			c = Circle(pos0, 1.5, mass = 1, vel=(0, 0, 0), restitution=restitution, fric_coeff=1, name = "f"+str(i))
			bodies.append(c)
			traj = torch.cat((scale*dp[i,:],-scale*dp[i+self.N_c,:]), axis=0).view(2,self.T+1)
			traj_f.append(Trajectory(vel = traj, name = "f"+str(i)))

		world = World(bodies, joints, dt=self.dt/5, tol = 1e-6, eps=0.1, strict_no_penetration = False)
		screen = None
		pygame.init()
		screen = pygame.display.set_mode((1000, 1000), pygame.DOUBLEBUF)
		screen.set_alpha(None)
		run_world_traj(world, run_time = 0.61, screen=screen, recorder=None, print_time=False, traj=traj_f)
		print('\n\n\n\n\n\n')
		print(world.states)
		print(len(world.states))
		print('\n\n\n\n\n\n')
		for t in range(self.T):
			if t > 0:
				y0 = torch.cat((world.states[t][1].view(1,1) - 500, 500 - world.states[t][2].view(1,1), scale*world.states[t][0].view(1,1)), axis = 0)/scale
				y = torch.cat((y,y0), axis = 1)
			else:
				y = torch.cat((world.states[t][1].view(1,1) - 500, 500 - world.states[t][2].view(1,1), scale*world.states[t][0].view(1,1)), axis = 0)/scale
		return y

	def forwardSagittalDiffSim(self, p, dp, polygon, xtraj0):
		bodies = []
		joints = []
		restitution = 0.00 # no impacts in quasi-dynamics
		fric_coeff = 0.01
		n_pol = int(polygon[0])

		xr = 500+250*xtraj0[0]
		yr = 500-250*xtraj0[1]

		# adds body based on triangulation
		r0 = Hull([xr, yr], [[1, 1], [-1, 1], [-1, -1], [1, -1]],
		         restitution=0.00, fric_coeff=0.00, mass = 0.01, name="obj")
		bodies.append(r0)

		for i in range(n_pol):
			x2 = [polygon[1+8*i], -polygon[2+8*i]]
			x1 = [polygon[3+8*i], -polygon[4+8*i]]
			x0 = [polygon[5+8*i], -polygon[6+8*i]]
			verts = 250*np.array([x0, x1, x2])
			print(verts)
			p0 = np.array([xr + polygon[7+8*i], yr - polygon[8+8*i]])
			r1 = Hull(p0, verts, restitution=restitution, mass = 0.0001, fric_coeff=1, name="obj_"+str(i))
			r1.add_force(Gravity(g=100))
			bodies.append(r1)
			joints += [FixedJoint(r1, r0)]
			r1.add_no_contact(r0)
			r0 = r1

		# Point Fingers
		traj_f = []
		for i in range(self.N_c):
			c = Circle([500+250*p[c*self.T],500-250*p[(c+self.N_c)*self.T]], 5, mass = 100000000, vel=(0, 0, 0), restitution=restitution,
			            fric_coeff=1, name = "f"+str(i))
			bodies.append(c)
			traj = torch.cat((250*dp[c*self.T:(c+1)*self.T],-250*dp[(c+self.N_c)*self.T:(c+self.N_c+1)*self.T]),axis=0)
			traj_f.append(Trajectory(vel = traj, name = "f"+str(i)))

		# Environment
		r = Rect([0, 500, 505], [900, 10],
		         restitution=restitution, fric_coeff=1)
		bodies.append(r)
		joints.append(TotalConstraint(r))

		world = World(bodies, joints, dt=self.dt)
		run_world_traj(world, run_time=self.dt*self.T, screen=None, recorder=None, traj=traj_f)
		return torch.cat([world.states[1,:]/250 - 0.2 , 0.2-world.states[2,:]/250 , world.states[0,:]], axis = 0)

	########################	
	# Forward Pass Methods #
	########################

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

	def forward2Sim(self, xtraj, x, x_img, polygons): 
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
			# for t in range(5):
			p0 = 1.1*(p[:,0]-torch.cat((r.view(3,5)[0:2,0],r.view(3,5)[0:2,0]),axis=0))+torch.cat((r.view(3,5)[0:2,0],r.view(3,5)[0:2,0]),axis=0)
			p = torch.cat((p0.view(-1,1), p), axis = 1)

			print(np.shape(p))

			dp, ddp = self.Difflayer(p.view(4,6))
			
			# print(p)
			# print(dp)
			# pdb.set_trace()
			self.rad = 10
			xtraj_new = self.forwardPlanarDiffSim(p, dp.double(), polygons[i,:], r.view(3,5)[:,0], self.rad)
			print(xtraj_new)
			print(r.view(3,5))
			if first:				
				y = torch.cat((xtraj_new.view(1,-1), .1*ddp.view(1,-1).double()), axis = 1)
				first = False
			else:
				y1 = torch.cat((xtraj_new.view(1,-1), .1*ddp.view(1,-1).double()), axis = 1)
				y = torch.cat((y, y1), axis = 0)
		# self.rad = self.rad/1.1
		# if self.rad < 1:
		# 	self.rad = 1
		return y

	def forwardVideo(self, video):
		# passes through the optimization problem
		# video encoding
		for t in range(self.T):
			frame = video[:,t*3*100*100:(t+1)*3*100*100]
			e_frame = self.forwardFrameEncoder(frame).detach()
			if t == 0:
				e_vid = e_frame
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
			
		print('decoding')
		# extracts image encoding and object trajectory
		# e_img = self.shap_dec(e_vid)
		xtraj = self.traj_dec(e_vid).detach()
		p_r = self.vid_p_dec.forward(e_vid)
		v = self.vid_v_dec.forward(e_vid)
		p_e = self.vid_pe_dec.forward(e_vid)
		fc = self.vid_fc_dec.forward(e_vid)
		fc_e = self.vid_fce_dec.forward(e_vid)
		print('going through cvx')
		# decodes each trajectory
		first = True
		for i in range(np.shape(e_frame)[0]):
			# params that should be obtained from video
			r = xtraj[i,:15]
			# params that can be computed explicity
			dr = xtraj[i,15:30]
			ddr = xtraj[i,30:]

			# learnes the parameters
			# dr, ddr = self.Difflayer(r.view(3,5))

			# solves for the contact trajectory
			p, f, _, _, _, _, _, err, _ = self.CTOlayer(r.view(3,5), ddr.view(3,5), fc[i,:].view(8,5), p_e[i,:].view(4,5), fc_e[i,:].view(8,5), v[i,:].view(8, 5), p_r[i,:].view(4, 5))

			if first:				
				y = torch.cat([p.view(1,-1), f.view(1,-1), 0*torch.max(err).view(1,-1)], axis = 1)
				first = False
			else:
				y_1 = torch.cat([p.view(1,-1), f.view(1,-1), 0*torch.max(err).view(1,-1)], axis = 1)
				y = torch.cat((y, y_1), axis = 0)
		# self.tol = self.tol/1.1
		return y

	#####################	
	# Parameter Methods #
	#####################

	def forwardVideotoImage(self,video):
		# passes through all frames
		first_frame = True
		for t in range(self.T):
			frame = video[:,t*3*100*100:(t+1)*3*100*100]
			e_frame = self.forwardFrameEncoder(frame)
			if first_frame:
				e_vid = e_frame
				first_frame = False
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
		
		# extracts image encoding
		e_img = self.shap_dec(e_vid)

		# decodes image
		return self.decoder(self.fc3(e_img).view(-1,20,6,6))

	def forwardVideotoTraj(self,video):
		# passes through all frames
		first_frame = True
		for t in range(self.T):
			frame = video[:,t*3*100*100:(t+1)*3*100*100]
			e_frame = self.forwardFrameEncoder(frame)
			if first_frame:
				e_vid = e_frame
				first_frame = False
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)
		
		# extracts object trajecotry
		return self.traj_dec(e_vid)

	#####################
	# Assesment methods #
	#####################

	def forwardVideoToParrams(self, video, x):
		# passes through the optimization problem
		# video encoding
		for t in range(self.T):
			frame = video[:,t*3*100*100:(t+1)*3*100*100]
			e_frame = self.forwardFrameEncoder(frame).clone().detach()
			if t == 0:
				e_vid = e_frame
			else:
				e_vid = torch.cat((e_vid,e_frame), axis=1)

		# extracts the parameters
		p_r = self.vid_p_dec.forward(e_vid)
		v = self.vid_v_dec.forward(e_vid)
		p_e = self.vid_pe_dec.forward(e_vid)
		fc = self.vid_fc_dec.forward(e_vid)
		fc_e = self.vid_fce_dec.forward(e_vid)

		# params that can be learned from above
		p_r0 = x[:,0:20]
		v0 = x[:,120:160]
		p_e0 = x[:,60:80]
		fc0 = x[:,20:60]
		fc_e0 = x[:,80:120]

		return p_r-p_r0, v-v0, p_e-p_e0, fc-fc0, fc_e-fc_e0
	
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
			p_r = self.p_dec.forward(torch.cat((e_img[i,:].view(1,-1), xtraj[i,:].view(1,45)), 1))
			v = self.v_dec.forward(torch.cat((e_img[i,:].view(1,-1), xtraj[i,:].view(1,45)), 1))
			p_e = self.pe_dec.forward(torch.cat([e_img[i,:].view(1,-1), xtraj[i,:].view(1,45)], 1))
			fc = self.fc_dec.forward(torch.cat([xtraj[i,:].view(1,45), v.view(1,40)], 1))
			fc_e = self.fce_dec.forward(torch.cat([p_e.view(1,20), xtraj[i,:].view(1,45)], 1))

			# p_r = p_r0
			# p_e = p_e0
			# v = v0
			# fc = fc0
			# fc_e = fc_e0

			# detaches the pre-trained
			# p_r.detach()
			# v.detach()

			p, f, _ ,_, _, _, _ = self.CTOlayer(r.view(3,5), ddr.view(3,5), fc.view(8,5), p_e.view(4,5), fc_e.view(8,5), v.view(8, 5), p_r.view(4, 5))

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
			v = self.v_dec.forward(torch.cat((e_img[i,:].view(1,-1), xtraj[i,:].view(1,45)), 1))
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
			v = self.v_dec.forward(torch.cat((e_img[i,:].view(1,-1), xtraj[i,:].view(1,45)), 1))
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
			p = self.p_dec.forward(torch.cat((e_img[i,:].view(1,-1), xtraj[i,:].view(1,45)), 1))
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
			v = self.v_dec.forward(torch.cat((e_img[i,:].view(1,-1), xtraj[i,:].view(1,45)), 1))
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
			p_e = self.p_dec.forward(torch.cat((e_img[i,:].view(1,-1), xtraj[i,:].view(1,45)), 1))
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
		return z

	###################	
	# Utility Methods #
	###################

	def save(self, name = "cnn_model.pt"):
		torch.save(self.state_dict(),"../data/models/"+str(name))

	def load(self, name = "cnn_model.pt"):
		self.load_state_dict(torch.load("../data/models/"+str(name)),strict = False)

	def gen_res(self,inputs_1,inputs_2,inputs_img,name="res"):
		y = self.forward(torch.tensor(inputs_1).float(),torch.tensor(inputs_2).float(),torch.tensor(inputs_img).float())
		y = y.clone().detach()/330
		np.savetxt("../data/"+name+"_0_2f.csv", y.data.numpy(), delimiter=",")

	def gen_resVid(self,vids,name="res"):
		y = self.forwardVideo(torch.tensor(vids).float())
		y = y.clone().detach()/33
		np.savetxt("../data/"+name+"_0_2f.csv", y.data.numpy(), delimiter=",")
