import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

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

def TrainDecoders(net, inputs_1, inputs_2, inputs_img, epochs = 10):
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
		r = xtraj[0:15].view(3,5)

		# starts with zero
		loss = 0
		posdiff = np.array((0.05),(0.05));
		scalediff = 500;

		for t in range(5):
			# rotation matrix
			p = r[0:2,t]
			th = r[2,t]
			c, s = np.cos(th), np.sin(th)
			rmat = np.array(((c, s), (-s, c)))

			for c in range(2):
				# reprojects for each finger
				v1 = np.array((v[c*2,t]),(v[c*2+2,t])) - p
				v2 = np.array((v[c*2+1,t]),(v[c*2+3,t])) - p
				v1 = rmat@v1
				v2 = rmat@v2

				# rescales for each finger and sends to pixel space
				v1 = np.around(scalediff*v1 + scalediff*posdiff)
				v2 = np.around(scalediff*v2 + scalediff*posdiff)

				# loss function computation
				loss += sdf[v1[1],v1[0]] 
				loss += sdf[v2[1],v2[0]]

		return loss

	@staticmethod
	def backward(ctx, grad_output):
		v, sdf, xtraj = ctx.saved_tensors

		r = xtraj[0:15].view(3,5)

		# starts with zero
		loss = 0
		posdiff = np.array((0.05),(0.05));
		scalediff = 500;

		grad_input = grad_output.clone()
		grad_input = 0.0

		for t in range(5):
			# rotation matrix
			p = r[0:2,t]
			th = r[2,t]
			c, s = np.cos(th), np.sin(th)
			rmat = np.array(((c, s), (-s, c)))

			for c in range(2):
				# reprojects for each finger
				v1 = np.array((v[c*2,t]),(v[c*2+2,t])) - p
				v2 = np.array((v[c*2+1,t]),(v[c*2+3,t])) - p
				v1 = rmat@v1
				v2 = rmat@v2

				# rescales for each finger and sends to pixel space
				v1 = np.around(scalediff*v1 + scalediff*posdiff)
				v2 = np.around(scalediff*v2 + scalediff*posdiff)

				# loss function computation
				grad_input -= sdf[v1[1],v1[0]]
				grad_input -= sdf[v2[1],v2[0]]

		return grad_input, None