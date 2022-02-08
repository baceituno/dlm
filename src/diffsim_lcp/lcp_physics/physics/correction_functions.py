def correct_penetrations(self):

	# object frame
	irot = torch.tensor([[math.cos(self.bodies[0].p[0].item()), math.sin(self.bodies[0].p[0].item())],[-math.sin(self.bodies[0].p[0].item()),math.cos(self.bodies[0].p[0].item())]]).view(2,2)
	pos = self.bodies[0].pos.view(2,1).float()

	# corrects obj wrt fingers
	for b in self.bodies:
		if b.name[0] == 'f':
			inside = False
			# finger in objec frame					
			ray1 = torch.matmul(irot,torch.tensor([b.p[1],b.p[2]]).view(2,1).float()-pos).view(2,1)

			if b.prev == None:
				b.prev = b.p

			ray0 = torch.matmul(irot,torch.tensor([b.prev[1],b.prev[2]]).view(2,1).float()-pos).view(2,1)

			# checks for each point in the segment
			for alpha in np.linspace(0,1,11):
				inside_ = False
				ray = alpha*ray0 + (1-alpha)*ray1
				for f in self.facets:
					# checks is the ray in y+ intercepts the facet (Jordan Polygon Theorem)
					if (ray[0] <= max(f[0][0],f[1][0])) and (ray[0] >= min(f[0][0],f[1][0])):
						if f[0][0] == f[1][0]:
							if (ray[1] <= f[0][1]):
								inside_ = True - inside_
							if (ray[1] <= f[1][1]):
								inside_ = True - inside_
						else:
							if (ray[1] <= ((ray[0]-f[1][0])*(f[0][1]-f[1][1])/(f[0][0]-f[1][0]) + f[1][1])):
								inside_ = True - inside_
				inside = inside + inside_

			# finds closest facet
			ray = ray1
			dist = float('inf')
			for i, f in enumerate(self.facets):
				v1 = torch.tensor(f[1].copy()).view(2,1)-torch.tensor(f[0].copy()).view(2,1) # AB
				v2 = ray - torch.tensor(f[0].copy()).view(2,1)
				d1 = torch.matmul(v1.view(1,2),v2)/torch.norm(v1)

				if d1.item() <= 0:
					p1 = torch.tensor(f[0].copy()).view(2,1)
				elif d1.item() >= torch.norm(torch.tensor(f[1].copy()-f[0].copy()).view(2,1)).item():
					p1 = torch.tensor(f[1].copy()).view(2,1)
				else:
					p1 = torch.tensor(f[0].copy()).view(2,1) + d1.view(1,1)*v1/torch.norm(v1)
				# print(torch.norm(p1 - ray).item())
				# print(torch.norm(p1 - ray).item() < dist)
				if torch.norm(p1 - ray).item() < dist:
					dist = torch.norm(p1 - ray).item()
					correction = (p1 - ray).view(2,1)
					near = i

					if b.nearest == -1:
						f1 = self.facets[near]
					else:
						f1 = self.facets[b.nearest]
					va1 = torch.tensor(f1[0].copy()).view(2,1)-torch.tensor(f1[1].copy()).view(2,1)
					va2 = ray1.view(2,1) - ray0.view(2,1)

					A = torch.cat((va1.float(),va2), axis = 1).view(2,2)
					b_vec = (ray1.view(2,1)-torch.tensor(f1[1].copy()).view(2,1)).view(2,1)
					try:
						alpha_beta, _ = torch.solve(b_vec.float(),A.float())
						ray_int = alpha_beta[0].item()*torch.tensor(f1[0].copy()).view(2,1) + (1-alpha_beta[0].item())*torch.tensor(f1[1].copy()).view(2,1)
						if alpha_beta[0].item() > 1.0 or alpha_beta[0].item() < 0:
							ray_int = ray
					except:
						ray_int = ray
			try:
				near
			except:
				print('no near LOL')
				pdb.set_trace()

			if near == b.nearest:
				pass
			else:
				if inside:
					if b.nearest == -1:
						b.nearest = near

						f = self.facets[b.nearest]
						v1 = torch.tensor(f[1].copy()).view(2,1)-torch.tensor(f[0].copy()).view(2,1) # AB
						v2 = ray - torch.tensor(f[0].copy()).view(2,1)
						d1 = torch.matmul(v1.view(1,2),v2)/torch.norm(v1)
						
						if d1.item() <= 0:
							p1 = torch.tensor(f[0].copy()).view(2,1)
						elif d1.item() >= torch.norm(torch.tensor(f[1].copy()-f[0].copy()).view(2,1)).item():
							p1 = torch.tensor(f[1].copy()).view(2,1)
						else:
							p1 = torch.tensor(f[0].copy()).view(2,1) + d1.view(1,1)*v1/torch.norm(v1)

					else:	
						f = self.facets[b.nearest]
						
						v1 = torch.tensor(f[0].copy()).view(2,1)-torch.tensor(f[1].copy()).view(2,1)
						v2 = ray1.view(2,1) - ray0.view(2,1)

						A = torch.cat((v1.float(),v2), axis = 1).view(2,2)
						b_vec = (ray1.view(2,1)-torch.tensor(f[1].copy()).view(2,1)).view(2,1)
						try:
							alpha_beta, _ = torch.solve(b_vec.float(),A.float())
							p1 = alpha_beta[0].item()*torch.tensor(f[0].copy()).view(2,1) + (1-alpha_beta[0].item())*torch.tensor(f[1].copy()).view(2,1)
							if alpha_beta[0].item() > 1.0 or alpha_beta[0].item() < 0:
								p1 = ray
						except:
							p1 = ray
					dist = torch.norm(p1 - ray).item()
					correction = (p1 - ray).view(2,1)
				else:
					b.nearest = near

			if inside:
				# projects to closes facet and assigns
				if dist > 1:
					correction = (ray_int - ray).view(2,1)
					v0 = (ray).view(2,1).float()
					v1 = (ray_int).view(2,1).float()

					cross = v1[0]*v0[1] - v1[1]*v0[0]
					dot = torch.matmul(v1.view(1,2),v0).view(1,1)
					cos = dot/(torch.norm(v0).view(1,1) * torch.norm(v1).view(1,1) + 1e-3)

					if cross.item() > 0:
						angle = torch.acos(cos).view(1)
					else:
						angle = -torch.acos(cos).view(1)

					if np.isnan(sum(angle)).sum().item() > 0:
						angle = torch.tensor([0.0])

					self.bodies[0].set_p(self.bodies[0].p + torch.cat([torch.tensor(0*angle), torch.matmul(irot.transpose(0, 1), -correction.float()).view(2)]))
	
	self.find_contacts()

	# corrects object wrt environmnet
	for c in self.contacts:
		# corrects for penetration with the environment
		if self.bodies[c[1]].name == 'env':
			if c[0][3].item() > 1e-1:
				corrected_p = self.bodies[0].p + torch.cat([torch.tensor([0.0]).double(), torch.tensor([0.0]).double(), -torch.tensor([c[0][3].item()]).double()])
				self.bodies[0].set_p(corrected_p)

	self.find_contacts()

def correct_penetrations_GN(self):

	# object frame
	irot = torch.tensor([[math.cos(self.bodies[0].p[0].item()), math.sin(self.bodies[0].p[0].item())],[-math.sin(self.bodies[0].p[0].item()),math.cos(self.bodies[0].p[0].item())]]).view(2,2)
	pos = self.bodies[0].pos.view(2,1).float()

	# corrects obj wrt fingers
	betas = []
	facets = []
	points = []
	p_ints = []

	insides = False
	signal = True
	for b in self.bodies:
		if b.name[0] == 'f':
			inside = False
			# finger in objec frame					
			ray1 = torch.matmul(irot,torch.tensor([b.p[1],b.p[2]]).view(2,1).float()-pos).view(2,1)

			if b.prev == None:
				b.prev = b.p

			ray0 = torch.matmul(irot,torch.tensor([b.prev[1],b.prev[2]]).view(2,1).float()-pos).view(2,1)

			# checks for each point in the segment
			for alpha in np.linspace(0,1,11):
				inside_ = False
				ray = alpha*ray0 + (1-alpha)*ray1
				for f in self.facets:
					# checks is the ray in y+ intercepts the facet (Jordan Polygon Theorem)
					if (ray[0] <= max(f[0][0],f[1][0])) and (ray[0] >= min(f[0][0],f[1][0])):
						if f[0][0] == f[1][0]:
							if (ray[1] <= f[0][1]):
								inside_ = True - inside_
							if (ray[1] <= f[1][1]):
								inside_ = True - inside_
						else:
							if (ray[1] <= ((ray[0]-f[1][0])*(f[0][1]-f[1][1])/(f[0][0]-f[1][0]) + f[1][1])):
								inside_ = True - inside_
				inside = inside + inside_

			# finds closest facet
			if inside:
				ray = ray1
				dist = float('inf')
				for i, f in enumerate(self.facets):
					v1 = torch.tensor(f[1].copy()).view(2,1)-torch.tensor(f[0].copy()).view(2,1) # AB
					v2 = ray - torch.tensor(f[0].copy()).view(2,1)
					d1 = torch.matmul(v1.view(1,2),v2)/torch.norm(v1)

					if d1.item() <= 0:
						p1 = torch.tensor(f[0].copy()).view(2,1)
					elif d1.item() >= torch.norm(torch.tensor(f[1].copy()-f[0].copy()).view(2,1)).item():
						p1 = torch.tensor(f[1].copy()).view(2,1)
					else:
						p1 = torch.tensor(f[0].copy()).view(2,1) + d1.view(1,1)*v1/torch.norm(v1)

					if torch.norm(p1 - ray).item() < dist:
						f1 = f
						intercept = p1
						va1 = torch.tensor(f1[0].copy()).view(2,1)-torch.tensor(f1[1].copy()).view(2,1)
						va2 = ray1.view(2,1) - ray0.view(2,1)

						A = torch.cat((va1.float(),va2), axis = 1).view(2,2)
						b_vec = (ray1.view(2,1)-torch.tensor(f1[1].copy()).view(2,1)).view(2,1)
						try:
							alpha_beta, _ = torch.solve(b_vec.float(),A.float())
							ray_int = alpha_beta[0].item()*torch.tensor(f1[0].copy()).view(2,1) + (1-alpha_beta[0].item())*torch.tensor(f1[1].copy()).view(2,1)
							beta = alpha_beta[0]
							if alpha_beta[0].item() > 1.0 or alpha_beta[0].item() < 0:
								ray_int = ray
								beta = 0
								signal = False
							else:
								signal = True
						except:
							signal = False
							ray_int = ray
							beta = 0

				betas.append(beta)
				# points.append(torch.matmul(irot,torch.tensor([b.p_goal[1],b.p_goal[2]]).view(2,1).float()-pos).view(2,1))
				points.append(ray)
				facets.append(f1)
				p_ints.append(intercept)

			insides = insides + inside
	
	if insides:
		if signal:
			# print(self.readjust_int(points, facets, p_ints).view(3))
			# self.bodies[0].set_p(self.bodies[0].p + self.readjust(points, facets, betas).view(3))
			self.bodies[0].set_p(self.bodies[0].p + self.readjust_int(points, facets, p_ints).view(3))

	self.find_contacts()

	# corrects object wrt environmnet
	for c in self.contacts:
		# corrects for penetration with the environment
		if self.bodies[c[1]].name == 'env':
			if c[0][3].item() > 1e-1:
				corrected_p = self.bodies[0].p + torch.cat([torch.tensor([0.0]).double(), torch.tensor([0.0]).double(), -torch.tensor([c[0][3].item()]).double()])
				# self.bodies[0].set_p(corrected_p)

	self.find_contacts()

def readjust(self, p, f, beta, maxIters = 100):
	p1 = p[0]
	f1 = f[0]
	beta1 = beta[0]
	int_1 = beta1*f1[0] + (1-beta1)*f1[1]

	# initialization
	x = torch.tensor([0.0])
	y = torch.tensor([0.0])
	th = torch.tensor([0.0])

	if len(p) == 1:
		for i in range(maxIters):
			# residual
			e1_0 = x + torch.cos(th)*int_1[0] - torch.sin(th)*int_1[1] - p1[0]
			e1_1 = y + torch.cos(th)*int_1[1] + torch.sin(th)*int_1[0] - p1[1]

			res = (e1_0**2 + e1_1**2).item()
			
			# partials
			dedx = 2*e1_0
			dedy = 2*e1_1

			dedth = 2*e1_0*(-torch.sin(th)*int_1[0] - torch.cos(th)*int_1[1]) 
			dedth = dedth + 2*e1_1*(-torch.sin(th)*int_1[1] + torch.cos(th)*int_1[0])

			J_pseudo = torch.pinverse(torch.cat([dedx, dedy, dedth]).view(1,3))

			x = x - J_pseudo[0]*res
			y = y - J_pseudo[1]*res
			# th = th - J_pseudo[2]*res

			if torch.sum(J_pseudo**2).item() < 1e-12:
				return torch.cat([th,x,y])
	else:
		# reads next
		p2 = p[1]
		f2 = f[1]
		beta2 = beta[1]
		int_2 = beta2*f2[0] + (1-beta2)*f2[1]

		for i in range(maxIters):
			# residual
			e1_0 = x + torch.cos(th)*int_1[0] - torch.sin(th)*int_1[1] - p1[0]
			e1_1 = y + torch.cos(th)*int_1[1] + torch.sin(th)*int_1[0] - p1[1]

			e2_0 = x + torch.cos(th)*int_2[0] - torch.sin(th)*int_2[1] - p2[0]	
			e2_1 = y + torch.cos(th)*int_2[1] + torch.sin(th)*int_2[0] - p2[1]

			res = (e1_0**2 + e1_1**2 + e2_0**2 + e2_1**2).item()
			
			# partials
			dedx = 2*e1_0 + 2*e2_0
			dedy = 2*e1_1 + 2*e2_1

			dedth = 2*e1_0*(-torch.sin(th)*int_1[0] - torch.cos(th)*int_1[1]) 
			dedth = dedth + 2*e1_1*(-torch.sin(th)*int_1[1] + torch.cos(th)*int_1[0])
			dedth = dedth + 2*e2_0*(-torch.sin(th)*int_2[0] - torch.cos(th)*int_2[1])
			dedth = dedth + 2*e2_1*(-torch.sin(th)*int_2[1] + torch.cos(th)*int_2[0])

			J_pseudo = torch.pinverse(torch.cat([dedx, dedy, dedth]).view(1,3))

			x = x - J_pseudo[0]*res
			y = y - J_pseudo[1]*res
			# th = th - J_pseudo[2]*res

			if torch.sum(J_pseudo**2).item() < 1e-12:
				return torch.cat([th,x,y])
	return torch.cat([th,x,y])

def readjust_int(self, p, f, p_int, maxIters = 100):
	p1 = p[0]
	f1 = f[0]
	int_1 = p_int[0]

	# initialization
	x = torch.tensor([0.0])
	y = torch.tensor([0.0])
	th = torch.tensor([0.0])
	# pdb.set_trace()
	if len(p) == 1:
		for i in range(maxIters):
			# residual
			e1_0 = x + torch.cos(th)*int_1[0] - torch.sin(th)*int_1[1] - p1[0]
			e1_1 = y + torch.cos(th)*int_1[1] + torch.sin(th)*int_1[0] - p1[1]

			res = (e1_0**2 + e1_1**2).item()
			print(res)
			# partials
			dedx = 2*e1_0
			dedy = 2*e1_1

			dedth = 2*e1_0*(-torch.sin(th)*int_1[0] - torch.cos(th)*int_1[1]) 
			dedth = dedth + 2*e1_1*(-torch.sin(th)*int_1[1] + torch.cos(th)*int_1[0])

			J = torch.cat([dedx, dedy, dedth]).view(1,3)
			J_pseudo = torch.pinverse(torch.cat([dedx, dedy, dedth]).view(1,3))

			# x = x - J_pseudo[0]*res
			# y = y - J_pseudo[1]*res
			# th = th - J_pseudo[2]*res

			x = x - 1e-2*dedx
			y = y - 1e-2*dedy
			th = th - 1e-2*dedth

			if torch.sum(J**2).item() < 1e-12:
				print('solved')
				return torch.cat([th.float(),x.float(),y.float()])
	else:
		# reads next
		p2 = p[1]
		f2 = f[1]
		int_2 = p_int[1]

		for i in range(maxIters):
			# residual
			e1_0 = x + int_1[0] - p1[0]
			e1_1 = y + int_1[1] - p1[1]

			e2_0 = x + int_2[0] - p2[0]	
			e2_1 = y + int_2[1] - p2[1]

			res = (e1_0**2 + e1_1**2 + e2_0**2 + e2_1**2).item()
			print(res)
			# partials
			dedx = 2*e1_0 + 2*e2_0
			dedy = 2*e1_1 + 2*e2_1

			dedth = 2*e1_0*(-torch.sin(th)*int_1[0] - torch.cos(th)*int_1[1]) 
			dedth = dedth + 2*e1_1*(-torch.sin(th)*int_1[1] + torch.cos(th)*int_1[0])
			dedth = dedth + 2*e2_0*(-torch.sin(th)*int_2[0] - torch.cos(th)*int_2[1])
			dedth = dedth + 2*e2_1*(-torch.sin(th)*int_2[1] + torch.cos(th)*int_2[0])

			J = torch.cat([dedx, dedy, dedth]).view(1,3)
			J_pseudo = torch.pinverse(torch.cat([dedx, dedy, dedth]).view(1,3))

			# x = x - J_pseudo[0]*res
			# y = y - J_pseudo[1]*res
			# th = th - J_pseudo[2]*res

			x = x - 1e-2*dedx
			y = y - 1e-2*dedy
			th = th - 1e-2*dedth

			if torch.sum(J**2).item() < 1e-12:	
				print('solved')
				return torch.cat([th.float(),x.float(),y.float()])
	print('solved')
	return torch.cat([th.float(),x.float(),y.float()])