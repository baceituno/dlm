import numpy as np
from numpy import loadtxt
import torch

def load_dataset(start = 1, end = 2):
	data = np.array((loadtxt("../data/data_" + str(start) +  "_2f_sq.csv", delimiter=',')))
	for i in range(start+1,end+1):
	    new_data = np.array((loadtxt("../data/data_"+str(i)+"_2f_sq.csv", delimiter=',')))
	    data = np.concatenate((data, new_data), axis=0)

	vids = np.array((loadtxt("../data/vids_" + str(start) +  "_2f_sq.csv", delimiter=',')))
	for i in range(start+1,end+1):
	    new_data = np.array((loadtxt("../data/vids_"+str(i)+"_2f_sq.csv", delimiter=',')))
	    vids = np.concatenate((vids, new_data), axis=0)

	polygons = np.array((loadtxt("../data/polygons_" + str(start) +  "_2f_sq.csv", delimiter=',')))
	for i in range(start+1,end+1):
	    new_data = np.array((loadtxt("../data/polygons_"+str(i)+"_2f_sq.csv", delimiter=',')))
	    polygons = np.concatenate((polygons, new_data), axis=0)

	return data, vids, polygons

def load_dataset_sagittal(start = 1, end = 2):
	data = np.array((loadtxt("../data/data_sag_" + str(start) +  "_2f_sq.csv", delimiter=',')))
	for i in range(start+1,end+1):
	    new_data = np.array((loadtxt("../data/data_sag_"+str(i)+"_2f_sq.csv", delimiter=',')))
	    data = np.concatenate((data, new_data), axis=0)

	vids = np.array((loadtxt("../data/vids_sag_" + str(start) +  "_2f_sq.csv", delimiter=',')))
	for i in range(start+1,end+1):
	    new_data = np.array((loadtxt("../data/vids_sag_"+str(i)+"_2f_sq.csv", delimiter=',')))
	    vids = np.concatenate((vids, new_data), axis=0)

	polygons = np.array((loadtxt("../data/polygons_sag_" + str(start) + "_2f_sq.csv", delimiter=',')))
	for i in range(start+1,end+1):
	    new_data = np.array((loadtxt("../data/polygons_sag_"+str(i)+"_2f_sq.csv", delimiter=',')))
	    polygons = np.concatenate((polygons, new_data), axis=0)

	return data, vids, polygons

def parse_data(data, img_dim = 2500, extra_zeros = 161):
	inputs_1 = torch.tensor(data[:,:45]) # object trajectory
	inputs_2 = torch.tensor(data[:,45:205]) # trajectory decoding
	inputs_img = torch.tensor(data[:,205:205+img_dim*7]) # object shape
	SDF = torch.tensor(data[:,205+img_dim*7:205+8*img_dim]) # object sdf
	N_data = np.shape(data)[0]
	inputs_2[:,0:20] = torch.tensor(data[:,205+8*img_dim:225+8*img_dim])
	labels = torch.cat((torch.tensor(data[:,205+8*img_dim:]),torch.tensor(np.zeros((N_data,extra_zeros)))), axis = 1)

	return inputs_1, inputs_2, inputs_img, SDF, labels

def parse_dataVids(data, img_dim = 2500, extra_zeros = 1):
	inputs_1 = torch.tensor(data[:,:45]) # object trajectory
	inputs_2 = torch.tensor(data[:,45:205]) # trajectory decoding
	inputs_img = torch.tensor(data[:,205:205+7*img_dim]) # object shape
	inputs_env = torch.tensor(data[:,205+7*img_dim:205+8*img_dim]) # env mask
	N_data = np.shape(data)[0]
	inputs_2[:,0:20] = torch.tensor(data[:,205+8*img_dim:225+8*img_dim])
	labels = torch.cat((torch.tensor(data[:,205+8*img_dim:225+8*img_dim]),torch.tensor(np.zeros((N_data,extra_zeros)))), axis = 1)
	
	return inputs_1, inputs_2, inputs_img, inputs_env, labels

def parse_dataVidsOld(data, img_dim = 2500, extra_zeros = 1):
	inputs_1 = torch.tensor(data[:,:45]) # object trajectory
	inputs_2 = torch.tensor(data[:,45:205]) # trajectory decoding
	inputs_img = torch.tensor(data[:,205:205+img_dim]) # object shape
	N_data = np.shape(data)[0]
	inputs_2[:,0:20] = torch.tensor(data[:,205+2*img_dim:225+2*img_dim])
	labels = torch.cat((torch.tensor(data[:,205+2*img_dim:225+2*img_dim]),torch.tensor(np.zeros((N_data,extra_zeros)))), axis = 1)
	
	return inputs_1, inputs_2, inputs_img, labels, labels