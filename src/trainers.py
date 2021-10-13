import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt

def TrainVideoJointParams(net, vids, x, xtraj, epochs = 100, n_batches = 20, optimizer = None):
	N_data = np.shape(vids)[0]
	criterion = torch.nn.MSELoss(reduction='mean')
	losses_test1 = []
	n_data = np.shape(vids)[0]

	for epoch in range(epochs):
		for batch in range(n_batches):
			idx0 = batch*n_data//n_batches
			idx2 = (batch+1)*n_data//n_batches
		
			loss_t = 0
			optimizer.zero_grad()
			pr, v, p_e, fc, fc_e = net.forwardVideoToParams(torch.tensor(vids[idx0:idx2,:]).float(), torch.tensor(x[idx0:idx2,:]).float(), xtraj = xtraj[idx0:idx2,:])
			loss = criterion(100*pr, torch.tensor(np.zeros((N_data//n_batches, 20))).float())
			loss += criterion(100*v, torch.tensor(np.zeros((N_data//n_batches, 40))).float())
			loss += criterion(100*p_e, torch.tensor(np.zeros((N_data//n_batches, 20))).float())
			loss += criterion(10*fc, torch.tensor(np.zeros((N_data//n_batches, 40))).float())
			loss += criterion(10*fc_e, torch.tensor(np.zeros((N_data//n_batches, 40))).float())
			loss_t = loss.item()
			loss.backward()
			optimizer.step()

		pr, v, p_e, fc, fc_e = net.forwardVideoToParams(torch.tensor(vids).float(), torch.tensor(x).float(), xtraj = xtraj)
		loss = criterion(100*pr, torch.tensor(np.zeros((N_data, 20))).float())
		loss += criterion(100*v, torch.tensor(np.zeros((N_data, 40))).float())
		loss += criterion(100*p_e, torch.tensor(np.zeros((N_data, 20))).float())
		loss += criterion(10*fc, torch.tensor(np.zeros((N_data, 40))).float())
		loss += criterion(10*fc_e, torch.tensor(np.zeros((N_data, 40))).float())
		loss_t = loss.item()

		print("params Reconstruction loss at epoch ",epoch," = ",loss_t)

	dpr = criterion(100*pr, torch.tensor(np.zeros((N_data, 20))).float()).item()
	dv = criterion(100*v, torch.tensor(np.zeros((N_data, 40))).float()).item()
	dpe = criterion(100*p_e, torch.tensor(np.zeros((N_data, 20))).float()).item()
	dfc = criterion(10*fc, torch.tensor(np.zeros((N_data, 40))).float()).item()
	dfce = criterion(10*fc_e, torch.tensor(np.zeros((N_data, 40))).float()).item()

	return loss_t, dpr, dv, dpe, dfc, dfce

def TrainMaskJointParams(net, xtraj, x_img, x, epochs = 100, n_batches = 20, optimizer = None):
	N_data = np.shape(xtraj)[0]
	criterion = torch.nn.MSELoss(reduction='mean')
	losses_test1 = []
	n_data = np.shape(xtraj)[0]

	for epoch in range(epochs):
		for batch in range(n_batches):
			idx0 = batch*n_data//n_batches
			idx2 = (batch+1)*n_data//n_batches
		
			loss_t = 0
			optimizer.zero_grad()
			pr, v, p_e, fc, fc_e = net.forwardMaskToParams(torch.tensor(xtraj[idx0:idx2,:]).float(), torch.tensor(x[idx0:idx2,:]).float(), torch.tensor(x_img[idx0:idx2,:]).float())
			loss = criterion(100*pr, torch.tensor(np.zeros((N_data//n_batches, 20))).float())
			loss += criterion(100*v, torch.tensor(np.zeros((N_data//n_batches, 40))).float())
			loss += criterion(100*p_e, torch.tensor(np.zeros((N_data//n_batches, 20))).float())
			loss += criterion(10*fc, torch.tensor(np.zeros((N_data//n_batches, 40))).float())
			loss += criterion(10*fc_e, torch.tensor(np.zeros((N_data//n_batches, 40))).float())
			loss_t = loss.item()
			losses_test1.append(loss_t)
			loss.backward()
			optimizer.step()

		pr, v, p_e, fc, fc_e = net.forwardMaskToParams(torch.tensor(xtraj).float(), torch.tensor(x).float(), torch.tensor(x_img).float())
		loss = criterion(100*pr, torch.tensor(np.zeros((N_data, 20))).float())
		loss += criterion(100*v, torch.tensor(np.zeros((N_data, 40))).float())
		loss += criterion(100*p_e, torch.tensor(np.zeros((N_data, 20))).float())
		loss += criterion(10*fc, torch.tensor(np.zeros((N_data, 40))).float())
		loss += criterion(10*fc_e, torch.tensor(np.zeros((N_data, 40))).float())
		loss_t = loss.item()

		print("params Reconstruction loss at epoch ",epoch," = ",loss_t)


def TrainVideo2V(net, vids, x, epochs = 100, optimizer = None):
	N_data = np.shape(vids)[0]
	criterion = torch.nn.MSELoss(reduction='mean')

	for epoch in range(epochs):
		loss_t = 0
		optimizer.zero_grad()
		_, v, _, _, _ = net.forwardVideoToParams(torch.tensor(vids).float(), torch.tensor(x).float())
		loss = criterion(100*v, torch.tensor(np.zeros((N_data, 40))).float())
		# loss = LossObjectFrame(outputs, p[:,0:20], inputs_1.float())
		loss_t = loss.item()
		loss.backward()
		optimizer.step()

		print("v Reconstruction loss at epoch ",epoch," = ",loss_t)

def TrainVideo2P_NN(net, vids, x, inputs_1, p, epochs = 100, optimizer = None):
	N_data = np.shape(vids)[0]
	criterion = torch.nn.MSELoss(reduction='mean')
	labels = torch.tensor(net.forwardVideo(torch.tensor(vids).float(), inputs_1.float(), x.float(), bypass = False, pass_soft = True).clone().detach().numpy())

	for epoch in range(epochs):
		loss_t = 0
		optimizer.zero_grad()
		outputs = net.forwardVideo(torch.tensor(vids).float(), inputs_1.float(), x.float(), bypass = True, pass_soft = False)
		loss = criterion(100*outputs, 100*labels.float())
		# loss = LossObjectFrame(outputs, p[:,0:20], inputs_1.float())
		loss_t = loss.item()
		loss.backward()
		optimizer.step()

		print("p Reconstruction loss at epoch ",epoch," = ",loss_t)

	return loss_t


def Evaluate2P(net, vids, x, inputs_1, p, bypass = False, pass_soft = False):
	N_data = np.shape(vids)[0]
	criterion = torch.nn.MSELoss(reduction='mean')
	labels = torch.tensor(net.forwardVideo(torch.tensor(vids).float(), inputs_1.float(), x.float(), bypass = False, pass_soft = True).clone().detach().numpy())

	outputs = net.forwardVideo(torch.tensor(vids).float(), inputs_1.float(), x.float(), bypass = bypass, pass_soft = pass_soft)
	loss = criterion(100*outputs, 100*labels.float())
	loss_t = loss.item()
	print("p Reconstruction loss = ",loss_t)

	return loss_t

def TrainVideo2P_CVX(net, vids, x, inputs_1, p = [], epochs = 100, optimizer = None):
	N_data = np.shape(vids)[0]
	criterion = torch.nn.MSELoss(reduction='mean')
	labels = torch.tensor(net.forwardVideo(torch.tensor(vids).float(), inputs_1.float(), x.float(), bypass = False, pass_soft = True).clone().detach().numpy())

	for epoch in range(epochs):
		loss_t = 0
		optimizer.zero_grad()
		outputs = net.forwardVideo(torch.tensor(vids).float(), inputs_1.float(), x.float(), bypass = False, pass_soft = False)
		loss = criterion(100*outputs, 100*labels.float())
		# loss = LossObjectFrame(outputs, p[:,0:20], inputs_1.float())
		loss_t = loss.item()
		loss.backward()
		optimizer.step()

		print("p Reconstruction loss at epoch ",epoch," = ",loss_t)

	return loss_t

def TrainVideoCVXStructure(net, vids, x, inputs_1, p = [], epochs = 100, optimizer = None):
	N_data = np.shape(vids)[0]
	criterion = torch.nn.MSELoss(reduction='mean')
	labels = torch.tensor(net.forwardVideo(torch.tensor(vids).float(), inputs_1.float(), x.float(), bypass = False, pass_soft = True).clone().detach().numpy())
	
	for epoch in range(epochs):
		loss_t = 0
		optimizer.zero_grad()
		outputs = net.forwardVideotoCVX(torch.tensor(vids).float(), inputs_1.float(), x.float(), bypass = False, pass_soft = False)
		# pr, v, p_e, fc, fc_e = net.forwardVideoToParams(torch.tensor(vids).float(), torch.tensor(x).float(), xtraj = inputs_1.float())
		
		loss = criterion(100*outputs, 0.0*labels.float()[:,0])
		# loss += criterion(100*pr, torch.tensor(np.zeros((N_data, 20))).float())
		# loss += criterion(100*v, torch.tensor(np.zeros((N_data, 40))).float())
		# loss += criterion(100*p_e, torch.tensor(np.zeros((N_data, 20))).float())
		# loss += criterion(10*fc, torch.tensor(np.zeros((N_data, 40))).float())
		# loss += criterion(10*fc_e, torch.tensor(np.zeros((N_data, 40))).float())

		loss_t = loss.item()
		loss.backward()
		optimizer.step()

		print("CVX loss at epoch ",epoch," = ",loss_t)

	return loss_t

def TrainVideoParams(net, vids, x, epochs = 100, plot = False, optimizer = None):
	N_data = np.shape(vids)[0]
	criterion = torch.nn.MSELoss(reduction='mean')
	losses_test1 = []

	for epoch in range(epochs):
		loss_t = 0
		optimizer.zero_grad()
		pr, _, _, _, _ = net.forwardVideoToParams(torch.tensor(vids).float(), torch.tensor(x).float())
		loss = criterion(100*pr, torch.tensor(np.zeros((N_data, 20))).float())
		loss_t = loss.item()
		losses_test1.append(loss_t)
		loss.backward()
		optimizer.step()

		print("p_r Reconstruction loss at epoch ",epoch," = ",loss_t)

	losses_test2 = []

	for epoch in range(epochs):
		loss_t = 0
		optimizer.zero_grad()
		_, v, _, _, _ = net.forwardVideoToParams(torch.tensor(vids).float(), torch.tensor(x).float())
		loss = criterion(100*v, torch.tensor(np.zeros((N_data, 40))).float())
		loss_t = loss.item()
		losses_test2.append(loss_t)
		loss.backward()
		optimizer.step()

		print("v Reconstruction loss at epoch ",epoch," = ",loss_t)

	losses_test3 = []

	for epoch in range(epochs):
		loss_t = 0
		optimizer.zero_grad()
		_, _, p_e, _, _ = net.forwardVideoToParams(torch.tensor(vids).float(), torch.tensor(x).float())
		loss = criterion(100*p_e, torch.tensor(np.zeros((N_data, 20))).float())
		loss_t = loss.item()
		losses_test3.append(loss_t)
		loss.backward()
		optimizer.step()

		print("p_e Reconstruction loss at epoch ",epoch," = ",loss_t)

	losses_test4 = []

	for epoch in range(epochs):
		loss_t = 0
		optimizer.zero_grad()
		_, _, _, fc, _ = net.forwardVideoToParams(torch.tensor(vids).float(), torch.tensor(x).float())
		loss = criterion(10*fc, torch.tensor(np.zeros((N_data, 40))).float())
		loss_t = loss.item()
		losses_test4.append(loss_t)
		loss.backward()
		optimizer.step()

		print("fc Reconstruction loss at epoch ",epoch," = ",loss_t)

	losses_test5 = []

	for epoch in range(epochs):
		loss_t = 0
		optimizer.zero_grad()
		_, _, _, _, fc_e = net.forwardVideoToParams(torch.tensor(vids).float(), torch.tensor(x).float())
		loss = criterion(10*fc_e, torch.tensor(np.zeros((N_data, 40))).float())
		loss_t = loss.item()
		losses_test5.append(loss_t)
		loss.backward()
		optimizer.step()

		print("fc_e Reconstruction loss at epoch ",epoch," = ",loss_t)

	if plot:
		fig,a =  plt.subplots(2,2)
		a[0][0].plot(losses_test1)
		a[0][0].set_title('p_r loss')

		a[0][1].plot(losses_test2)
		a[0][1].set_title('v loss')

		a[1][0].plot(losses_test3)
		a[1][0].set_title('p_e loss')

		a[1][1].plot(losses_test4)
		a[1][1].set_title('fc loss')
		plt.show()

