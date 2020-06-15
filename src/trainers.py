import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import time

def LossShapeVAE(recon_x, x, mu, logvar):
	x = np.reshape(x,(-1,1,50,50))
	BCE = F.mse_loss(recon_x, x, size_average=False)
	KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
	return BCE + KLD

def LossFrameVAE(recon_x, x, mu, logvar):
	x = np.reshape(x,(-1,3,50,50))
	BCE = F.mse_loss(recon_x, x, size_average=False)
	KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
	return BCE + KLD

def TrainVideoVAE(net, vids, T, epochs = 10):
	optimizer = optim.Adam(net.parameters(), lr=0.001)
	for epochs in range(epochs):
		# trains for each times-step
		for j in range(T):
			loss_t = 0
			optimizer.zero_grad()
			frame = torch.tensor(vids[:,j*7500:(j+1)*7500])
			frame_rec, mu, logvar = net.forwardFrameVAE(frame.float())
			loss = LossFrameVAE(frame_rec, frame.float(), mu, logvar)
			loss.backward()
			optimizer.step()
			
			loss_t = loss.item()
			print("Frame Autoencoder loss at epoch ",epoch," = ",loss_t)


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
	optimizer = optim.Adam(net.parameters(), lr=0.001)
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