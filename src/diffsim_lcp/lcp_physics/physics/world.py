import time
# from functools import lru_cache
from argparse import Namespace

import ode
import torch
import pdb
import math

from . import engines as engines_module
from . import contacts as contacts_module
from .utils import Indices, Defaults, cross_2d, get_instance, left_orthogonal
import numpy as np

X, Y = Indices.X, Indices.Y
DIM = Defaults.DIM

class Trajectory(object):
	"""Fingers velocity trajectory"""
	def __init__(self, vel=np.zeros((2,5)), name='TrajNo'):
		# super(Trajectory, self).__init__()
		self.vel = vel
		self.name = name

class Reference(object):
	"""Fingers pose trajectory"""
	def __init__(self, pos=np.zeros((3,5)), name='RefNo'):
		# super(Trajectory, self).__init__()
		self.ref = pos
		self.name = name
		

class World:
	"""A physics simulation world, with bodies and constraints.
	"""
	def __init__(self, bodies, constraints=[], dt=Defaults.DT, engine=Defaults.ENGINE,
				 contact_callback=Defaults.CONTACT, eps=Defaults.EPSILON,
				 tol=Defaults.TOL, fric_dirs=Defaults.FRIC_DIRS,
				 post_stab=Defaults.POST_STABILIZATION, strict_no_penetration=True, facets = []):
		self.contacts_debug = False  # XXX

		# Load classes from string name defined in utils
		self.engine = get_instance(engines_module, engine)
		self.contact_callback = get_instance(contacts_module, contact_callback)
		self.states = []
		self.fingers1 = []
		self.fingers2 = []
		self.times = []
		self.t = 0
		self.t_prev = -1
		self.idx = 0
		self.dt = dt
		self.traj = []
		self.ref = []
		self.eps = eps
		self.tol = tol
		self.fric_dirs = fric_dirs
		self.post_stab = post_stab
		self.gamma = 0.01
		self.applied = True
		self.facets = facets

		self.bodies = bodies
		self.vec_len = len(self.bodies[0].v)
		self.prev_p = self.bodies[0].p

		# XXX Using ODE for broadphase for now
		# self.space = ode.HashSpace()
		# for i, b in enumerate(bodies):
		# 	 b.geom.body = i
		# 	 self.space.add(b.geom)

		self.static_inverse = True
		self.num_constraints = 0
		self.joints = []
		for j in constraints:
			b1, b2 = j.body1, j.body2
			i1 = bodies.index(b1)
			i2 = bodies.index(b2) if b2 else None
			self.joints.append((j, i1, i2))
			self.num_constraints += j.num_constraints
			if not j.static:
				self.static_inverse = False

		M_size = bodies[0].M.size(0)
		self._M = bodies[0].M.new_zeros(M_size * len(bodies), M_size * len(bodies))
		# XXX Better way for diagonal block matrix?
		for i, b in enumerate(bodies):
			self._M[i * M_size:(i + 1) * M_size, i * M_size:(i + 1) * M_size] = b.M

		self.set_v(torch.cat([b.v for b in bodies]))

		self.contacts = None
		self.find_contacts()
		self.correct_finger_penetrations()
		self.strict_no_pen = strict_no_penetration
		# if self.strict_no_pen:
		# 	for b in self.bodies:
		# 		print(f'{b.__class__}: {vars(b)}\n')
		# 	assert all([c[0][3].item() <= self.tol for c in self.contacts]),'Interpenetration at start:\n{}'.format(self.contacts)

	def step(self, fixed_dt=True):
		dt = self.dt
		if fixed_dt:
			end_t = self.t + self.dt
			while self.t < end_t:
				dt = end_t - self.t
				self.step_dt(dt)
		else:
			self.step_dt(dt)

	# @profile
	def step_dt(self, dt):
		# moves fingers
		self.apply_trajectories()

		# solves LCP
		start_p = torch.cat([b.p for b in self.bodies])
		start_rot_joints = [(j[0].rot1, j[0].rot2) for j in self.joints]
		new_v = self.engine.solve_dynamics(self, dt)
		self.set_v(new_v)

		for body in self.bodies:
			body.move(dt)
		for joint in self.joints:
			joint[0].move(dt)
			joint[0].stabilize()

		# resolves penetrations
		self.find_contacts()
		self.solveInconsistencies()

		if self.post_stab:
			self.find_contacts()
			tmp_v = self.v
			dp = self.engine.post_stabilization(self).squeeze(0)
			dp /= 2 # XXX Why 1/2 factor?
			# XXX Clean up / Simplify this update?
			self.set_v(dp)
			for body in self.bodies:
				body.move(dt)
			for joint in self.joints:
				joint[0].move(dt)
			# print('s2')
			self.set_v(tmp_v)

			# self.find_contacts()  # XXX Necessary to recheck contacts?
			# self.correct_finger_penetrations()
		self.times.append(self.t)

		self.t += dt

	def get_v(self):
		return self.v

	def set_v(self, new_v):
		if np.isnan(new_v).sum().item() > 0:
			new_v[:] = 0.0
		self.v = new_v
		for i, b in enumerate(self.bodies):
			b.v = self.v[i * len(b.v):(i + 1) * len(b.v)]

	def set_p(self, new_p):
		for i, b in enumerate(self.bodies):
			b.set_p(new_p[i * self.vec_len:(i + 1) * self.vec_len])

	def apply_forces(self, t):
		return torch.cat([b.apply_forces(t) for b in self.bodies])

	def check_penetration(self, ray0, ray1, npoints):
		inside = False
		for alpha in np.linspace(0,1,npoints):
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
		return inside

	def apply_trajectories(self):
		# PI contrrol weights
		w1 = 0.5	# P-term
		w2 = 1 - w1 # I-term

		total_vel = 0
		# gets velocities
		if self.idx >= 0 and self.applied:
			for body in self.bodies:
				for tr in self.traj:
					for ref in self.ref:
						if body.name == tr.name and body.name == ref.name:
							if self.idx < np.shape(tr.vel)[1]:
								vel = w1*tr.vel[:,self.idx]
								vel = torch.cat([vel.new_zeros(1), vel])
								dt_ = math.ceil((self.t+1e-6)*10)/10 - self.t
								if self.idx == np.shape(tr.vel)[1]-1:
									vel += w2*(ref.ref[:,self.idx] - body.p)/dt_
									body.p_goal = ref.ref[:,self.idx].view(3,1)
								else:
									vel += w2*(ref.ref[:,self.idx+1] - body.p)/dt_
									body.p_goal = ref.ref[:,self.idx+1].view(3,1)
								body.v = vel
								total_vel += torch.norm(vel).item()

			# # updates velocities
			self.set_v(torch.cat([b.v for b in self.bodies]))
			self.applied = True

	def solveInconsistencies(self):
		# finds the penetrating bodies and the intersection points
		irot = torch.tensor([[math.cos(self.bodies[0].p[0].item()), math.sin(self.bodies[0].p[0].item())],[-math.sin(self.bodies[0].p[0].item()),math.cos(self.bodies[0].p[0].item())]]).view(2,2)
		pos = self.bodies[0].pos.view(2,1).float()

		irot_prev = torch.tensor([[math.cos(self.prev_p[0].item()), math.sin(self.prev_p[0].item())],[-math.sin(self.prev_p[0].item()),math.cos(self.prev_p[0].item())]]).view(2,2)
		pos_prev = self.prev_p[1:3].view(2,1).float()

		pen_bodies = []
		intercepts = []

		for b in self.bodies:
			if b.name[0] == 'f':
				# finger in objec frame					
				ray1 = torch.matmul(irot,torch.tensor([b.p[1],b.p[2]]).view(2,1).float()-pos).view(2,1)
				ray0 = torch.matmul(irot_prev,torch.tensor([b.prev[1],b.prev[2]]).view(2,1).float()-pos_prev).view(2,1)

				# checks for each point in the segment
				inside = self.check_penetration(ray1, ray0, 11)

				if inside:
					f = self.facets[b.nearest2]
					normal = torch.tensor([0,0]).float().view(2,1)
					normal[0] = f[0][1].copy()-f[1][1].copy()
					normal[1] = f[1][0].copy()-f[0][0].copy()
					
					normal = torch.matmul(irot.transpose(0, 1), normal)/(torch.norm(normal) + 1e-6)

					v1 = torch.tensor(f[0].copy()).view(2,1)-torch.tensor(f[1].copy()).view(2,1)
					v2 = ray1.view(2,1) - ray0.view(2,1)

					try:
						A = torch.cat((v1.float(),v2), axis = 1).view(2,2)
						b_vec = (ray1.view(2,1)-torch.tensor(f[1].copy()).view(2,1)).view(2,1)
						alpha_beta, _ = torch.solve(b_vec.float(),A.float())
						p_int = alpha_beta[0].item()*torch.tensor(f[0].copy()).view(2,1) + (1-alpha_beta[0].item())*torch.tensor(f[1].copy()).view(2,1)					
					except:
						p_int = ray0

					p_int = torch.matmul(irot.transpose(0, 1).double(), p_int.double()) + pos
					vel = torch.tensor([b.p[1],b.p[2]]).view(2,1) - torch.tensor([p_int[0],p_int[1]]).view(2,1)
					vel = vel/(torch.norm(vel) + 1e-5)

					if torch.matmul(normal.view(1,2).double(),vel.double()).item() > 0.5:
						pen_bodies.append(b)
						intercepts.append(p_int)

		# finds the angular diplacement assuming the two bodies stick

		######################################
		# (These operations assume infinite  #
		# finger mass and that all bodies 	 #
		# remain connected)					 #
		######################################

		w_bodies = torch.tensor([0.0]).view(1,1)	
		v_bodies = torch.tensor([0.0,0.0]).view(2,1)

		for i, b in enumerate(pen_bodies):
			# cross product
			p_int = intercepts[i]
			
			p_vec = p_int - self.bodies[0].pos.view(2,1)
			p_vec_f = torch.tensor([b.p[1],b.p[2]]).view(2,1) - self.bodies[0].pos.view(2,1)

			v = torch.tensor([b.p[1],b.p[2]]).view(2,1) - p_int
			w = p_vec[0]*v[1] - p_vec[1]*v[0]
			# w_bodies = w_bodies + w/(torch.norm(p_vec_f) + 1e-6)**2
			v_bodies = v_bodies + v

		# corrects the body locations based on the sticking assumption
		corrected_p = self.bodies[0].p + torch.cat([w_bodies.double().view(1), v_bodies.double().view(2)])
		pos_prev = self.bodies[0].pos.view(2,1).float()

		self.bodies[0].set_p(corrected_p)

		if len(pen_bodies) == 1:
			# finds second body
			if pen_bodies[0].name == "f0":
				b2 = self.bodies[2]
			else:
				b2 = self.bodies[1]

			# finger in object frame	
			irot_prev = irot
			irot = torch.tensor([[math.cos(self.bodies[0].p[0].item()), math.sin(self.bodies[0].p[0].item())],[-math.sin(self.bodies[0].p[0].item()),math.cos(self.bodies[0].p[0].item())]]).view(2,2)				
			
			# pdb.set_trace()

			ray1 = torch.matmul(irot.float()     ,torch.tensor([b2.p[1],b2.p[2]]).view(2,1).float() - pos).view(2,1)
			ray0 = torch.matmul(irot_prev.float(),torch.tensor([b2.p[1],b2.p[2]]).view(2,1).float() - pos_prev).view(2,1)

			# checks for each point in the segment
			inside = self.check_penetration(ray1, ray0, 11)

			if inside:
				f = self.facets[b2.nearest2]
				normal = torch.tensor([0,0]).float().view(2,1)
				normal[0] = f[0][1].copy()-f[1][1].copy()
				normal[1] = f[1][0].copy()-f[0][0].copy()
				
				normal = torch.matmul(irot.transpose(0, 1), normal)/(torch.norm(normal) + 1e-6)

				v1 = torch.tensor(f[0].copy()).view(2,1)-torch.tensor(f[1].copy()).view(2,1)
				v2 = ray1.view(2,1) - ray0.view(2,1)

				try:
					A = torch.cat((v1.float(),v2), axis = 1).view(2,2)
					b_vec = (ray1.view(2,1)-torch.tensor(f[1].copy()).view(2,1)).view(2,1)
					alpha_beta, _ = torch.solve(b_vec.float(),A.float())
					p_int = alpha_beta[0].item()*torch.tensor(f[0].copy()).view(2,1) + (1-alpha_beta[0].item())*torch.tensor(f[1].copy()).view(2,1)					
				except:
					p_int = ray0

				p_int = torch.matmul(irot.transpose(0, 1).double(), p_int.double()) + pos
				vel = torch.tensor([b2.p[1],b2.p[2]]).view(2,1) - torch.tensor([p_int[0],p_int[1]]).view(2,1)
				vel = vel/(torch.norm(vel) + 1e-5)

				p_vec = p_int - self.bodies[0].pos.view(2,1)
				p_vec_f = torch.tensor([b2.p[1],b2.p[2]]).view(2,1) - self.bodies[0].pos.view(2,1)

				w_bodies = torch.tensor([0.0]).view(1,1)	
				v_bodies = torch.tensor([0.0,0.0]).view(2,1)

				v = torch.tensor([b2.p[1],b2.p[2]]).view(2,1) - p_int
				w = p_vec[0]*v[1] - p_vec[1]*v[0]
				w_bodies = w/(torch.norm(p_vec_f) + 1e-6)**2
				v_bodies = v

				# updates pen
				corrected_p = self.bodies[0].p + torch.cat([w_bodies.double().view(1), v_bodies.double().view(2)])
				self.bodies[0].set_p(corrected_p)

		# fixes object position wrt environment and then fingers wrt object
		self.bodies[0].v = (self.bodies[0].p - self.prev_p)/self.dt
		self.correct_finger_penetrations()
		self.prev_p = self.bodies[0].p

	def solveInconsistenciesNLP(self):
		# checks if each finger is penetrating

		# checks distance from object to environment

		# solves NLP via casadi
		# from casadi import *
		opti = Opti();

		x = opti.variable()
		y = opti.variable()
		th = opti.variable()

		y1_p = opti.variable()
		y2_p = opti.variable()
		
		opti.minimize()
		opti.subject_to(x==0)


		# corrects object position
		

		# fixes object position wrt environment and then fingers wrt object
		self.correct_finger_penetrations()
		self.bodies[0].v = (self.bodies[0].p - self.prev_p)/self.dt
		self.prev_p = self.bodies[0].p

	def correct_finger_penetrations(self):
		aux_p = self.prev_p
		self.find_contacts()
		
		# corrects object wrt environmnet
		for c in self.contacts:
			# corrects for penetration with the environment
			if self.bodies[c[1]].name == 'env':
				if c[0][3].item() > 0.1:
					for b in self.bodies:
						if b.name[0] == 'o':
							corrected_p = b.p + torch.cat([torch.tensor([0.0]).double(), torch.tensor([0.0]).double(), -torch.tensor([c[0][3].item()]).double()])
							b.set_p(corrected_p)

		# object frame
		irot = torch.tensor([[math.cos(self.bodies[0].p[0].item()), math.sin(self.bodies[0].p[0].item())],[-math.sin(self.bodies[0].p[0].item()),math.cos(self.bodies[0].p[0].item())]]).view(2,2)
		pos = self.bodies[0].pos.view(2,1).float()
		
		irot_prev = torch.tensor([[math.cos(aux_p[0].item()), math.sin(aux_p[0].item())],[-math.sin(aux_p[0].item()),math.cos(aux_p[0].item())]]).view(2,2)
		pos_prev = aux_p[1:3].view(2,1).float()

		# correcs fingers wrt object
		for b in self.bodies:
			if b.name[0] == 'f':
				inside = False
				# finger in objec frame					
				ray1 = torch.matmul(irot,torch.tensor([b.p[1],b.p[2]]).view(2,1).float()-pos).view(2,1)
				ray0 = torch.matmul(irot_prev,torch.tensor([b.prev[1],b.prev[2]]).view(2,1).float()-pos_prev).view(2,1)

				# checks for each point in the segment
				inside = self.check_penetration(ray1, ray0, 10)

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
						# dist = torch.norm(p1 - ray).item()
						correction = (p1 - ray).view(2,1)
					else:
						b.nearest = near

				if inside:
					# projects to closes facet and assigns
					# if dist > 0:
					b.set_p(b.p + torch.cat([torch.tensor([0.0]), torch.matmul(irot.transpose(0, 1), correction.float()).view(2)]))
				b.prev = b.p

	def find_contacts(self):
		# import time
		# start_c1 = time.time()
		self.contacts = []
		# ODE contact detection
		# self.space.collide([self], self.contact_callback)
		# pdb.set_trace()
		for i, b1 in enumerate(self.bodies):
			g1 = Namespace()
			g1.no_contact = b1.no_contact
			g1.body_ref = b1
			g1.body = i
			for j, b2 in enumerate(self.bodies[:i]):
				g2 = Namespace()
				g2.no_contact = b2.no_contact
				g2.body_ref = b2
				g2.body = j
				self.contact_callback([self], g1, g2, self.gamma)

		self.contacts_debug = self.contacts  # XXX
		# end_c1 = time.time()
		# print("time per finding contacts: ")
		# print(end_c1 - start_c1)

	def restitutions(self):
		restitutions = self._M.new_empty(len(self.contacts))
		for i, c in enumerate(self.contacts):
			r1 = self.bodies[c[1]].restitution
			r2 = self.bodies[c[2]].restitution
			restitutions[i] = 0.0 * (r1 + r2) / 2
			# restitutions[i] = math.sqrt(r1 * r2)
		return restitutions

	def M(self):
		return self._M

	def Je(self):
		Je = self._M.new_zeros(self.num_constraints,
							   self.vec_len * len(self.bodies))
		row = 0
		for joint in self.joints:
			J1, J2 = joint[0].J()
			i1 = joint[1]
			i2 = joint[2]
			Je[row:row + J1.size(0),
			i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
			if J2 is not None:
				Je[row:row + J2.size(0),
				i2 * self.vec_len:(i2 + 1) * self.vec_len] = J2
			row += J1.size(0)
		return Je

	def Jc(self):
		Jc = self._M.new_zeros(len(self.contacts), self.vec_len * len(self.bodies))
		for i, contact in enumerate(self.contacts):
			c = contact[0]  # c = (normal, contact_pt_1, contact_pt_2, penetration_dist)
			i1 = contact[1]
			i2 = contact[2]
			J1 = torch.cat([cross_2d(c[1], c[0]).reshape(1, 1),
							c[0].unsqueeze(0)], dim=1)
			J2 = -torch.cat([cross_2d(c[2], c[0]).reshape(1, 1),
							 c[0].unsqueeze(0)], dim=1)
			Jc[i, i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
			Jc[i, i2 * self.vec_len:(i2 + 1) * self.vec_len] = J2
		return Jc

	def Jf(self):
		Jf = self._M.new_zeros(len(self.contacts) * self.fric_dirs,
							   self.vec_len * len(self.bodies))
		for i, contact in enumerate(self.contacts):
			c = contact[0]  # c = (normal, contact_pt_1, contact_pt_2)
			dir1 = left_orthogonal(c[0])
			dir2 = -dir1
			i1 = contact[1]  # body 1 index
			i2 = contact[2]  # body 2 index
			J1 = torch.cat([
				torch.cat([cross_2d(c[1], dir1).reshape(1, 1),
						   dir1.unsqueeze(0)], dim=1),
				torch.cat([cross_2d(c[1], dir2).reshape(1, 1),
						   dir2.unsqueeze(0)], dim=1),
			], dim=0)
			J2 = torch.cat([
				torch.cat([cross_2d(c[2], dir1).reshape(1, 1),
						   dir1.unsqueeze(0)], dim=1),
				torch.cat([cross_2d(c[2], dir2).reshape(1, 1),
						   dir2.unsqueeze(0)], dim=1),
			], dim=0)
			Jf[i * self.fric_dirs:(i + 1) * self.fric_dirs,
			i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
			Jf[i * self.fric_dirs:(i + 1) * self.fric_dirs,
			i2 * self.vec_len:(i2 + 1) * self.vec_len] = -J2
		return Jf

	def mu(self):
		return self._memoized_mu(*[(c[1], c[2]) for c in self.contacts])

	def _memoized_mu(self, *contacts):
		# contacts is argument so that cacheing can be implemented at some point
		mu = self._M.new_zeros(len(self.contacts))
		for i, contacts in enumerate(self.contacts):
			i1 = contacts[1]
			i2 = contacts[2]
			# mu[i] = torch.sqrt(self.bodies[i1].fric_coeff * self.bodies[i2].fric_coeff)
			mu[i] = 10.0
		return torch.diag(mu)

	def E(self):
		return self._memoized_E(len(self.contacts))

	def _memoized_E(self, num_contacts):
		n = self.fric_dirs * num_contacts
		E = self._M.new_zeros(n, num_contacts)
		for i in range(num_contacts):
			E[i * self.fric_dirs: (i + 1) * self.fric_dirs, i] += 1
		return E

	def save_state(self):
		raise NotImplementedError

	def load_state(self, state_dict):
		raise NotImplementedError

	def reset_engine(self):
		raise NotImplementedError



def run_world(world, animation_dt=None, run_time=10, print_time=True,
			  screen=None, recorder=None, pixels_per_meter=1, traj = [Trajectory()], pos_f = [Reference()]):
	"""Helper function to run a simulation forward once a world is created.
	"""
	import math
	# If in batched mode don't display simulation
	if hasattr(world, 'worlds'):
		screen = None

	if screen is not None:
		import pygame
		background = pygame.Surface(screen.get_size())
		background = background.convert()
		background.fill((255, 255, 255))

	if animation_dt is None:
		animation_dt = float(world.dt)
	elapsed_time = 0.
	prev_frame_time = -animation_dt
	start_time = time.time()

	world.idx = 0
	world.traj = traj
	world.ref = pos_f
	world.t_prev = -10.0
	# world.engine = get_instance(engines_module,'LemkeEngine')
	while world.t < run_time:
		if screen is not None:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					return

			if elapsed_time - prev_frame_time >= animation_dt or recorder:
				prev_frame_time = elapsed_time

				screen.blit(background, (0, 0))
				update_list = []
				for body in world.bodies:
					update_list += body.draw(screen, pixels_per_meter=pixels_per_meter)
				for joint in world.joints:
					update_list += joint[0].draw(screen, pixels_per_meter=pixels_per_meter)

				# Visualize contact points and normal for debug
				# (Uncomment contacts_debug line in contacts handler):
				if world.contacts_debug:
					for c in world.contacts_debug:
						(normal, p1, p2, penetration), b1, b2 = c
						b1_pos = world.bodies[b1].pos
						b2_pos = world.bodies[b2].pos
						p1 = p1 + b1_pos
						p2 = p2 + b2_pos
						# pygame.draw.circle(screen, (0, 255, 0), p1.data.numpy().astype(int), 5)
						# pygame.draw.circle(screen, (0, 0, 255), p2.data.numpy().astype(int), 5)
						# pygame.draw.line(screen, (0, 255, 0), p1.data.numpy().astype(int),
						# 				 (p1.data.numpy() + normal.data.numpy() * 100).astype(int), 3)

				if not recorder:
					pass
					# Don't refresh screen if recording
					pygame.display.update(update_list)
					pygame.display.flip()  # XXX
				else:	
					recorder.record(world.t)

			elapsed_time = time.time() - start_time
			if not recorder:
				# Adjust frame rate dynamically to keep real time
				wait_time = world.t - elapsed_time
				if wait_time >= 0 and not recorder:
					wait_time += animation_dt  # XXX
					time.sleep(max(wait_time - animation_dt, 0))
				#	animation_dt -= 0.005 * wait_time
				# elif wait_time < 0:
				#	animation_dt += 0.005 * -wait_time
				# elapsed_time = time.time() - start_time

		if world.t - world.t_prev >= 0.099:
			# print(world.t)
			for body in world.bodies:
				if body.name == "obj":
					world.states.append(body.p)
				if body.name == "f0":
					world.fingers1.append(body.p)
				if body.name == "f1":
					world.fingers2.append(body.p)
			world.idx += 1
			world.t_prev = round(world.t,2)
			world.applied = True

		world.step()
		# pdb.set_trace()

		elapsed_time = time.time() - start_time
		if print_time:
			print('\r ', '{} / {}  {} '.format(float(world.t), float(elapsed_time),
											   1 / animation_dt), end='')
