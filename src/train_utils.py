import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt

def LossShapeVAE(recon_x, x, mu, logvar):
	x = np.reshape(x,(-1,1,50,50))
	BCE = F.mse_loss(recon_x, x, size_average=False)
	KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
	return BCE + KLD

def LossFrameCVAE(recon_x, x, mu, logvar):
	x = np.reshape(x,(-1,3,100,100))
	BCE1 = F.mse_loss(recon_x, x, size_average=True)
	KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
	return 100*BCE1 + KLD


def LossObjectFrame(x_learned, x_label, xtraj):
	# reshapes each tensor
	xtraj = xtraj.view(-1,3,5)
	x_learned = x_learned.view(-1,4,5)
	x_label = x_label.view(-1,4,5)

	x_learned1 = x_learned.view(-1,4,5).clone()
	x_label1 = x_label.view(-1,4,5).clone()

	# for each time-step
	for k in range(np.shape(x_learned)[0]):
		for t in range(5):
			cos = torch.cos(xtraj[k,2,t]).view(1,1)
			sin = torch.sin(xtraj[k,2,t]).view(1,1)

			x_learned1[k,0,t] = cos*(x_learned[k,0,t] - xtraj[k,0,t]) + sin*(x_learned[k,1,t] - xtraj[k,1,t])
			x_learned1[k,1,t] = -sin*(x_learned[k,0,t] - xtraj[k,0,t]) + cos*(x_learned[k,1,t] - xtraj[k,1,t])

			x_learned1[k,2,t] = cos*(x_learned[k,2,t] - xtraj[k,0,t]) + sin*(x_learned[k,3,t] - xtraj[k,1,t])
			x_learned1[k,3,t] = -sin*(x_learned[k,2,t] - xtraj[k,0,t]) + cos*(x_learned[k,3,t] - xtraj[k,1,t])

			x_label1[k,0,t] = cos*(x_label[k,0,t] - xtraj[k,0,t]) + sin*(x_label[k,1,t] - xtraj[k,1,t])
			x_label1[k,1,t] = -sin*(x_label[k,0,t] - xtraj[k,0,t]) + cos*(x_label[k,1,t] - xtraj[k,1,t])

			x_label1[k,2,t] = cos*(x_label[k,2,t] - xtraj[k,0,t]) + sin*(x_label[k,3,t] - xtraj[k,1,t])
			x_label1[k,3,t] = -sin*(x_label[k,2,t] - xtraj[k,0,t]) + cos*(x_label[k,3,t] - xtraj[k,1,t])

	criterion = torch.nn.MSELoss(reduction='mean')
	return criterion(30*x_learned1.double(), 30*x_label1.double())

def TrainVideoCVAE(net, vids, epochs = 10):
	optimizer = optim.Adam(net.parameters(), lr=0.00001)
	losses_test = []
	dimVid = 3*100*100
	for epoch in range(epochs):
		# trains for each times-step
		for j in range(net.T):
			loss_t = 0
			optimizer.zero_grad()
			# frame_0 = torch.tensor(vids[:,0:dimVid])
			frame_0 = torch.tensor(vids[:,j*dimVid:(j+1)*dimVid])
			frame = torch.tensor(vids[:,j*dimVid:(j+1)*dimVid])

			frame_rec, mu, logvar = net.forwardFrameCVAE(frame.float(),frame_0.float())
			loss = LossFrameCVAE(frame_rec, frame.float(), mu, logvar)
			loss.backward()
			optimizer.step()
			
		loss_t = loss.item()
		losses_test.append(loss_t)
		print("Frame Autoencoder loss at epoch ",epoch," = ",loss_t)


def TrainVideoDecoders(net, vids, xtraj, x_img, x_env = [], epochs = 10, n_batches = 20, optimizer = None):
	criterion = torch.nn.MSELoss(reduction='mean')
	n_data = np.shape(vids)[0]
	losses_test = []
	for epoch in range(epochs):
		for batch in range(n_batches):
			idx0 = batch*n_data//n_batches
			idx2 = (batch+1)*n_data//n_batches
			loss_t = 0
			optimizer.zero_grad()
			output = net.forwardVideotoTraj(torch.tensor(vids[idx0:idx2,:]).float())
			loss = criterion(output, xtraj[idx0:idx2,:].float())
			loss_t = loss.item()
			loss.backward()
			optimizer.step()
			losses_test.append(loss_t)

		print("Traj.Reconstruction loss at epoch ",epoch," = ",loss_t)

	criterion = torch.nn.MSELoss(reduction='mean')
	losses_test = []

	for epoch in range(epochs):
		for batch in range(n_batches):
			idx0 = batch*n_data//n_batches
			idx2 = (batch+1)*n_data//n_batches
			loss_t = 0
			optimizer.zero_grad()
			output = net.forwardVideotoShapes(torch.tensor(vids[idx0:idx2,:]).float())
			loss = criterion(10*output[:,0:2500], 10*x_img[idx0:idx2,0:2500].float())
			loss_t = loss.item()
			losses_test.append(loss_t)
			loss.backward()
			optimizer.step()

		print("Shapes Reconstruction loss at epoch ",epoch," = ",loss_t)

	# for epoch in range(epochs):
	# 	for batch in range(n_batches):
	# 		idx0 = batch*n_data//n_batches
	# 		idx2 = (batch+1)*n_data//n_batches
	# 		loss_t = 0
	# 		optimizer.zero_grad()
	# 		output = net.forwardVideotoEnv(torch.tensor(vids[idx0:idx2,:]).float())
	# 		loss = criterion(10*output, 10*x_env[idx0:idx2,:2500].float())
	# 		loss_t = loss.item()
	# 		losses_test.append(loss_t)
	# 		loss.backward()
	# 		optimizer.step()

	# 	print("Env Reconstruction loss at epoch ",epoch," = ",loss_t)

def VizVideoCondDecoders(net, vids, xtraj, x_img, epochs = 1, n_batches = 20):
	dimVid = 3*100*100

	import torchvision.models as models
	import torchvision.transforms as transforms

	pilTrans = transforms.ToTensor()

	for epoch in range(epochs):
		# trains for each times-step
		for j in range(net.T):
			frame = torch.tensor(vids[:,j*dimVid:(j+1)*dimVid])
			frame_0 = torch.tensor(vids[:,0:dimVid])
			# frame_rec, mu, logvar = net.forwardFrameCVAE(frame.float(),frame_0.float())
			# for i in range(120):
			f1 = frame[0,:].view(3,100,100).detach().numpy()
			f2 = frame[0,:].view(3,100,100).detach().numpy()
			
			fig, ax = plt.subplots(nrows=2, sharex=True)
			ax[0].imshow(np.transpose(f1, (1, 2, 0)))
			ax[1].imshow(np.transpose(f2, (1, 2, 0)))
			plt.show()

def TrainShapeVAE(net, inputs_img, epochs = 10):
	# Trans the shape VAE
	optimizer = optim.Adam(net.parameters(), lr=0.001)
	for epoch in range(epochs):  # loop over the dataset multiple times
		loss_t = 0
		optimizer.zero_grad()
		outputs, mu, logvar = net.forwardShapeVAE(inputs_img.float())
		loss = LossShapeVAE(outputs, inputs_img.float(), mu, logvar)
		loss.backward()
		optimizer.step()
		
		loss_t = loss.item()

		print("Shape Autoencoder loss at epoch ",epoch," = ",loss_t)

def TrainDecoders(net, inputs_1, inputs_2, inputs_img, inputs_sdf, epochs = 10):
	N_data = np.shape(inputs_1)[0]
	criterion = torch.nn.MSELoss(reduction='mean')
	optimizer = optim.Adam(net.parameters(), lr=0.0001)
	for epoch in range(epochs):  # loop over the dataset multiple times
		loss_t = 0
		optimizer.zero_grad()
		dv = net.forward_v(inputs_1.float(),inputs_2.float(),inputs_img.float())
		loss = criterion(dv.float(), torch.tensor(np.zeros((N_data,40))).float())
		loss.backward()
		optimizer.step()
		
		loss_t = loss.item()

		print("Vertex decoder loss at epoch ",epoch," = ",loss_t)

		loss_t = 0
		optimizer.zero_grad()
		dv = net.forward_p(inputs_1.float(),inputs_2.float(),inputs_img.float())
		loss = criterion(dv.float(), torch.tensor(np.zeros((N_data,20))).float())
		loss.backward()
		optimizer.step()
		
		loss_t = loss.item()

		print("Guess decoder loss at epoch ",epoch," = ",loss_t)

		loss_t = 0
		optimizer.zero_grad()
		dfc = net.forward_fc(inputs_1.float(),inputs_2.float(),inputs_img.float())
		loss = criterion(dfc.float(), torch.tensor(np.zeros((N_data,40))).float())
		loss.backward()
		optimizer.step()
		
		loss_t = loss.item()

		print("FC decoder loss at epoch ",epoch," = ",loss_t)

		loss_t = 0
		optimizer.zero_grad()
		dfc = net.forward_fce(inputs_1.float(),inputs_2.float(),inputs_img.float())
		loss = criterion(dfc.float(), torch.tensor(np.zeros((N_data,40))).float())
		loss.backward()
		optimizer.step()
		
		loss_t = loss.item()

		print("FCe decoder loss at epoch ",epoch," = ",loss_t)

class LossShapeSDF(torch.autograd.Function):
	@staticmethod
	def forward(ctx, v, sdf, xtraj):
		ctx.save_for_backward(v, sdf, xtraj)
		r = xtraj[:,0:15].view(-1,3,5)
		v = v.view(-1,8,5)
		sdf = sdf.view(-1,50,50)

		# starts with zero
		loss = 0
		posdiff = np.array([[0.05],[0.05]]);
		scalediff = 500;
		for i in range(np.shape(v)[0]):
			for t in range(5):
				# rotation matrix
				p = r[i,0:2,t]
				th = r[i,2,t]
				c, s = np.cos(th), np.sin(th)
				rmat = torch.tensor(np.array(((c, s), (-s, c))))

				for c in range(2):
					# reprojects for each finger
					print(p.view(2,1))

					v1 = torch.tensor([[v[i,c*4,t]],[v[i,c*4+2,t]]])
					v2 = torch.tensor([[v[i,c*4+1,t]],[v[i,c*4+3,t]]])

					print(v1)
					print(v2)

					v1 = torch.matmul(rmat,v1 - p.view(2,1)) + posdiff
					v2 = torch.matmul(rmat,v2 - p.view(2,1)) + posdiff


					print(v1)
					print(v2)

					# rescales for each finger and sends to pixel space
					v1 = np.around(scalediff*v1)
					v2 = np.around(scalediff*v2)

					print(v1)
					print(v2)
					time.sleep(5)

					# loss function computation
					loss += sdf[i,np.min([np.max([v1[0],0]),49]).astype(int),np.min([np.max([49-v1[1],0]),49]).astype(int)] 
					loss += sdf[i,np.min([np.max([v2[0],0]),49]).astype(int),np.min([np.max([49-v2[1],0]),49]).astype(int)]
		return loss/np.shape(v)[0]

	@staticmethod
	def backward(ctx, grad_output):
		v, sdf, xtraj = ctx.saved_tensors

		r = xtraj[:,0:15].view(-1,3,5)
		v = v.view(-1,8,5)
		sdf = sdf.view(-1,50,50)

		# starts with zero
		grad_input = np.zeros((np.shape(v)[0],40))
		posdiff = np.array([[0.05],[0.05]]);
		scalediff = 500;
		for i in range(np.shape(v)[0]):
			grad_t = np.zeros((8,5))
			for t in range(5):
				# rotation matrix
				p = r[i,0:2,t]
				th = r[i,2,t]
				c, s = np.cos(th), np.sin(th)
				rmat = torch.tensor(np.array(((c, s), (-s, c))))

				for c in range(2):
					# reprojects for each finger
					v1 = torch.tensor([[v[i,c*4+0,t]],[v[i,c*4+2,t]]]) - p.view(2,1)
					v2 = torch.tensor([[v[i,c*4+1,t]],[v[i,c*4+3,t]]]) - p.view(2,1)

					v1 = torch.matmul(rmat,v1)
					v2 = torch.matmul(rmat,v2)

					# rescales for each finger and sends to pixel space
					v1 = np.around(scalediff*v1 + scalediff*posdiff)
					v2 = np.around(scalediff*v2 + scalediff*posdiff)

					# for v1
					x1 = sdf[i,np.min([np.max([v1[0]-1,0]),47]).astype(int),np.min([np.max([v1[1],1]),49]).astype(int)].numpy()
					x2 = sdf[i,np.min([np.max([v1[0]+1,2]),49]).astype(int),np.min([np.max([v1[1],1]),49]).astype(int)].numpy()

					y1 = sdf[i,np.min([np.max([v1[0],1]),49]).astype(int),np.min([np.max([v1[1]-1,0]),47]).astype(int)].numpy()
					y2 = sdf[i,np.min([np.max([v1[0],1]),49]).astype(int),np.min([np.max([v1[1]+1,2]),49]).astype(int)].numpy()

					dsdv1 = np.array([[x2-x1],[y2-y1]])/2

					grad_t[c*2+0,t] = dsdv1[0]
					grad_t[c*2+2,t] = dsdv1[1]

					# for v2
					x1 = sdf[i,np.min([np.max([v2[0]-1,0]),47]).astype(int),np.min([np.max([v2[1],0]),49]).astype(int)].numpy()
					x2 = sdf[i,np.min([np.max([v2[0]+1,2]),49]).astype(int),np.min([np.max([v2[1],0]),49]).astype(int)].numpy()

					y1 = sdf[i,np.min([np.max([v2[0],0]),49]).astype(int),np.min([np.max([v2[1]-1,0]),47]).astype(int)].numpy()
					y2 = sdf[i,np.min([np.max([v2[0],0]),49]).astype(int),np.min([np.max([v2[1]+1,2]),49]).astype(int)].numpy()

					dsdv2 = np.array([[x2-x1],[y2-y1]])/2

					grad_t[c*2+1,t] = dsdv2[0]
					grad_t[c*2+3,t] = dsdv2[1]
			# gradient computation
			grad_input[i,:] = grad_t.reshape((1,40))

		return torch.tensor(grad_input), None, None