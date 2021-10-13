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
net.gamma = 1

data11, vids11, polygons11 = load_dataset_sagittal(1,1)
print("parsing training data...")
inputs_111, inputs_211, inputs_img11, _, labels11 = parse_dataVids(data11)
print(np.shape(vids11))

data, vids, polygons = load_dataset_sagittal(80,80)
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

refs11_p = refs11.view(np.shape(refs11)[0],3,5)[:,0:2,:]
refs11_th = refs11.view(np.shape(refs11)[0],3,5)[:,2:,:]
refs_p = refs.view(np.shape(refs)[0],3,5)[:,0:2,:]
refs_th = refs.view(np.shape(refs)[0],3,5)[:,2:,:]

gt_outputs, _, err = net.forwardEndToEnd(torch.tensor(vids).float(), torch.tensor(polygons).float(), inputs_1.float(), render=False, bypass = False, x = inputs_2.float(), pass_soft = True)

criterion = torch.nn.MSELoss(reduction='mean')
loss = criterion(100*gt_outputs.float(), 100*refs.float())
loss_t = loss.item()
print("GT test loss at epoch ",0," = ",loss_t)
print("GT test error at epoch ",0," = ",torch.sum(err.float()**2).item())

gt_outputs, _, err = net.forwardEndToEnd(torch.tensor(vids11).float(), torch.tensor(polygons11).float(), inputs_111.float(), render=False, bypass = False, x = inputs_211.float(), pass_soft = True)

criterion = torch.nn.MSELoss(reduction='mean')
loss = criterion(100*gt_outputs.float(), 100*refs11.float())
loss_t = loss.item()
print("GT train loss at epoch ",0," = ",loss_t)
print("GT train error at epoch ",0," = ",torch.sum(err.float()**2).item())

n_gr = 0
n_corr = 0
n_cra = 0
n_we = 0

for i in range(40):
    error_i = criterion(100*gt_outputs[i:i+1].float(), 100*refs11[i:i+1].float()).item()
    if error_i < 0.5:
        n_gr = n_gr + 1

    if error_i < 3.0:
        n_corr = n_corr + 1

    if error_i > 5.0:
        n_we = n_we + 1

    if error_i > 10.0:
        n_cra = n_cra + 1

print('From ', np.shape(refs11)[0],' simulations, these many were great:',100*n_gr/np.shape(refs11)[0], '%')
print('From ', np.shape(refs11)[0],' simulations, these many were reasonable:',100*n_corr/np.shape(refs11)[0], '%')
print('From ', np.shape(refs11)[0],' simulations, these many were weird:',100*n_we/np.shape(refs11)[0], '%')
print('From ', np.shape(refs11)[0],' simulations, these many were crazy:',100*n_cra/np.shape(refs11)[0], '%')

# pdb.set_trace()

render = False

losses_2p_test = []

losses_test = []
losses_train = []

losses_test_p = []
losses_train_p = []

losses_test_th = []
losses_train_th = []

err_test = []
err_train = []

losses_NN = []

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=1e-4)
for epoch in range(10):  # loop over the dataset multiple times
    outputs,_,err = net.forwardEndToEnd(torch.tensor(vids11).float(), torch.tensor(polygons11).float(), inputs_111.float(), render=render, bypass = True)
    
    outputs_p = outputs.view(-1,3,5)[:,0:2,:]
    outputs_th = outputs.view(-1,3,5)[:,2:,:]

    loss_p = criterion(100*outputs_p.float(), 100*refs11_p.float())    
    loss_th = criterion(100*outputs_th.float(), 100*refs11_th.float())    

    losses_train_p.append(loss_p.item())
    losses_train_th.append(loss_th.item())

    loss = criterion(100*outputs.float(), 100*refs11.float())
    loss_t = loss.item()
    if epoch == 0:
        losses_train.append(loss_t)
    else:
        if loss_t < losses_train[-1]:
            losses_train.append(loss_t)
        else:
            losses_train.append(loss_t)
            # losses_train.append(losses_train[-1])

    err_train.append(torch.sum(err.float()**2))
    print("Train loss at epoch ",epoch," = ",loss_t)
    print("Train error at epoch ",epoch," = ",torch.sum(err.float()**2))

    outputs,_,err = net.forwardEndToEnd(torch.tensor(vids).float(), torch.tensor(polygons).float(), inputs_1.float(), render=render, bypass = True)
    
    outputs_p = outputs.view(-1,3,5)[:,0:2,:]
    outputs_th = outputs.view(-1,3,5)[:,2:,:]
    

    loss_p = criterion(100*outputs_p.float(), 100*refs_p.float())    
    loss_th = criterion(100*outputs_th.float(), 100*refs_th.float())    

    losses_test_p.append(loss_p.item())
    losses_test_th.append(loss_th.item())

    loss = criterion(100*outputs.float(), 100*refs)
    loss_t = loss.item()
    if epoch == 0:
        losses_test.append(loss_t)
    else:
        if loss_t < losses_test[-1]:
            losses_test.append(loss_t)
        else:
            losses_test.append(loss_t)
            # losses_test.append(losses_test[-1])
    err_test.append(torch.sum(err.float()**2))
    print("Test loss at epoch ",epoch," = ",loss_t)
    print("Train error at epoch ",epoch," = ",torch.sum(err.float()**2))

    loss_t = 0
    loss_nn = TrainVideo2P_NN(net, vids11, inputs_211, inputs_111, labels11, epochs = 50, optimizer = optimizer)
    losses_NN.append(loss_nn)

    loss2p = Evaluate2P(net, vids, inputs_2, inputs_1, labels)
    losses_2p_test.append(loss2p)

plt.figure(1)
plt.plot(losses_test,color="b")
plt.plot(losses_train,color="r")
plt.title('object traj loss: CNN')
plt.show()

plt.figure(2)
plt.plot(losses_test_p,color="b")
plt.plot(losses_train_p,color="r")
plt.title('object p_traj loss: CNN')
plt.show()

plt.figure(3)
plt.plot(losses_test_th,color="b")
plt.plot(losses_train_th,color="r")
plt.title('object th_traj loss: CNN')
plt.show()

plt.figure(4)
plt.plot(losses_NN,color="y")
plt.title('action training loss: CNN')
plt.show()

plt.figure(5)
plt.plot(err_train,color="r")
plt.plot(err_test,color="b")
plt.title('err_p')
plt.show()

net.save(name = "cnn_nn_sag_model.pt")

np.savetxt("../experiments_sag/ablations/test_nn_curve.csv", np.array(losses_test), delimiter=",")
np.savetxt("../experiments_sag/ablations/train_nn_curve.csv", np.array(losses_train), delimiter=",")

np.savetxt("../experiments_sag/ablations/test_nn_curve_p.csv", np.array(losses_test_p), delimiter=",")
np.savetxt("../experiments_sag/ablations/train_nn_curve_p.csv", np.array(losses_train_p), delimiter=",")

np.savetxt("../experiments_sag/ablations/test_nn_curve_th.csv", np.array(losses_test_th), delimiter=",")
np.savetxt("../experiments_sag/ablations/train_nn_curve_th.csv", np.array(losses_train_th), delimiter=",")

np.savetxt("../experiments_sag/ablations/l2p_nn_curve.csv", np.array(losses_NN), delimiter=",")
np.savetxt("../experiments_sag/ablations/l2p_nn_test_curve.csv", np.array(losses_2p_test), delimiter=",")

outputs = net.forwardEndToEnd(torch.tensor(vids).float(), torch.tensor(polygons).float(), inputs_1.float(), render=True, bypass = True)