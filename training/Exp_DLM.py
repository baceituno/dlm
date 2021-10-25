import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import *
from src.trainers import *
from src.datatools import *

import torch
import pdb
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

# define network
print("Setting up network...")
net = ContactNet(20, sagittal = True)
net.addFrameCVAELayers()
net.addVideoLayers()
net.addShapeVAELayers()
net.eval()
net.gamma = 0.1

try:
	net.load(name = "cnn_w_sag_model.pt")
	net.eval()
except:
	pass

data11, vids11, polygons11 = load_dataset_sagittal(1,1)
print("parsing training data...")
inputs_111, inputs_211, inputs_img11, _, labels11 = parse_dataVids(data11)
print(np.shape(vids11))

print("loading data...")
data, vids, polygons = load_dataset_sagittal(42,42)
N_data = np.shape(data)[0]
print("parsing training data...")
inputs_1, inputs_2, inputs_img, _, labels = parse_dataVids(data)
print(np.shape(vids))

corr_inputs_1 = inputs_1[:,:].float().view(-1,3,3,5)
# corr_inputs_1 = torch.cat((corr_inputs_1, 0.3*torch.cos(corr_inputs_1[:,:,2,:]).view(-1,3,1,5)), axis = 2)
corr_inputs_1[:,0,2,:] = 0.03*torch.sin(corr_inputs_1[:,0,2,:])
refs = corr_inputs_1[:,0,:,:].view(-1,15)

corr_inputs_1 = inputs_111[:,:].float().view(-1,3,3,5)
# corr_inputs_1 = torch.cat((corr_inputs_1, 0.3*torch.cos(corr_inputs_1[:,:,2,:]).view(-1,3,1,5)), axis = 2)
corr_inputs_1[:,0,2,:] = 0.03*torch.sin(corr_inputs_1[:,0,2,:])
refs11 = corr_inputs_1[:,0,:,:].view(-1,15)

render = False

passes = 1

refs11_p = refs11.view(-1,3,5)[:,0:2,:]
refs11_th = refs11.view(-1,3,5)[:,2:,:]
refs_p = refs.view(-1,3,5)[:,0:2,:]
refs_th = refs.view(-1,3,5)[:,2:,:]

losses_test_p = []
losses_train_p = []

losses_test_th = []
losses_train_th = []

losses_test = []
losses_train = []
err_train = []
err_test = []

import time

losses_2p = []
losses_2p_test = []

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=1e-5)
for epoch in range(3):  # loop over the dataset multiple times
    start = time.time()
    outputs, p, err = net.forwardEndToEnd(torch.tensor(vids11).float(), torch.tensor(polygons11).float(), inputs_111.float(), render=render, bypass = False)
    
    outputs_p = outputs.view(-1,3,5)[:,0:2,:]
    outputs_th = outputs.view(-1,3,5)[:,2:,:]

    loss_p = criterion(100*outputs_p.float(), 100*refs11_p.float())    
    loss_th = criterion(100*outputs_th.float(), 100*refs11_th.float())    

    losses_train_p.append(loss_p.item())
    losses_train_th.append(loss_th.item())


    loss = criterion(100*outputs.float(), 100*refs11.float())
    loss_t = loss.item()
    err_train.append(torch.sum(err.float()**2))
    if epoch == 0:
        losses_train.append(loss_t)
    else:
        if loss_t < losses_train[-1]:
            losses_train.append(loss_t)
        else:
            losses_train.append(loss_t)
            # losses_train.append(losses_train[-1])
    print("Train loss at epoch ",epoch," = ",loss_t)
    print("fwd pass: ")
    end = time.time()
    print(end - start)

    outputs, p, err = net.forwardEndToEnd(torch.tensor(vids).float(), torch.tensor(polygons).float(), inputs_1.float(), render=render, bypass = False)
    
    outputs_p = outputs.view(-1,3,5)[:,0:2,:]
    outputs_th = outputs.view(-1,3,5)[:,2:,:]

    loss_p = criterion(100*outputs_p.float(), 100*refs_p.float())    
    loss_th = criterion(100*outputs_th.float(), 100*refs_th.float())    

    losses_test_p.append(loss_p.item())
    losses_test_th.append(loss_th.item())

    loss = criterion(100*outputs.float(), 100*refs)
    loss_t = loss.item()
    err_test.append(torch.sum(err.float()**2))
    losses_test.append(loss_t)

    print("Test loss at epoch ",epoch," = ",loss_t)

    loss_t = 0

    for i in range(passes):
        optimizer.zero_grad()
        outputs11, p11, err11 = net.forwardEndToEnd(torch.tensor(vids11).float(), torch.tensor(polygons11).float(), inputs_111.float(), render=render, bypass = False)
        loss_1 = criterion(100*outputs11.float(), 100*refs11.float())
        # loss_1 += torch.sum((err11.float())**2)
        start = time.time()
        loss_1.backward()
        optimizer.step()
        print("backward pass: ")
        end = time.time()
        print(end - start)

    loss2p = Evaluate2P(net, vids11, inputs_211, inputs_111, labels11)
    losses_2p.append(loss2p)

    loss2p = Evaluate2P(net, vids, inputs_2, inputs_1, labels)
    losses_2p_test.append(loss2p)

net.save(name = "cnn_z3_sag_model.pt")

print('dsdws')

net.gamma = 1
net.load(name = "cnn_z3_sag_model.pt")
net.genResults(vids,polygons,inputs_1,inputs_2,name="full4_sag",bypass = False, pass_=False)
net.genResults(vids,polygons,inputs_1,inputs_2,name="GT4_sag",bypass = False, pass_=True)
net.load(name = "cnn_w_sag_model.pt")
net.genResults(vids,polygons,inputs_1,inputs_2,name="mdr4_sag",bypass = False, pass_=False)
net.load(name = "cnn_y_sag_model.pt")
net.genResults(vids,polygons,inputs_1,inputs_2,name="cvx4_sag",bypass = False, pass_=False)
net.load(name = "cnn_nn_sag_model.pt")
net.genResults(vids,polygons,inputs_1,inputs_2,name="nn4_sag",bypass = True, pass_=False)

pdb.set_trace()

import matplotlib as mpl
mpl.use('tkagg')

plt.figure(1)
plt.plot(losses_test,color="b")
plt.plot(losses_train,color="r")
plt.title('object traj loss: DDM')
plt.show()

plt.figure(2)
plt.plot(losses_test_p,color="b")
plt.plot(losses_train_p,color="r")
plt.title('object p_traj loss: DDM')
plt.show()

plt.figure(3)
plt.plot(losses_test_th,color="b")
plt.plot(losses_train_th,color="r")
plt.title('object th_traj loss: DDM')
plt.show()

plt.figure(4)
plt.plot(err_train,color="r")
plt.plot(err_test,color="b")
plt.title('err_p')
plt.show()

np.savetxt("../experiments_sag/ablations/l2p_full3_train.csv", np.array(losses_2p), delimiter=",")
np.savetxt("../experiments_sag/ablations/l2p_full3_curve.csv", np.array(losses_2p_test), delimiter=",")

np.savetxt("../experiments_sag/ablations/exp_full3_curve.csv", np.array(losses_test), delimiter=",")
np.savetxt("../experiments_sag/ablations/exp_full3_train.csv", np.array(losses_train), delimiter=",")

np.savetxt("../experiments_sag/ablations/test_full3_curve_p.csv", np.array(losses_test_p), delimiter=",")
np.savetxt("../experiments_sag/ablations/train_full3_curve_p.csv", np.array(losses_train_p), delimiter=",")

np.savetxt("../experiments_sag/ablations/test_full3_curve_th.csv", np.array(losses_test_th), delimiter=",")
np.savetxt("../experiments_sag/ablations/train_full3_curve_th.csv", np.array(losses_train_th), delimiter=",")

outputs = net.forwardEndToEnd(torch.tensor(vids).float(), torch.tensor(polygons).float(), inputs_1.float(), render=True, bypass = True)