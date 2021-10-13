import torch
import pygame
from lcp_physics.physics.bodies import Circle, Rect, Hull, NonConvex
from lcp_physics.physics.constraints import Joint, TotalConstraint
from lcp_physics.physics.constraints import FixedJoint
from lcp_physics.physics.forces import Gravity, MDP, FingerTrajectory
from lcp_physics.physics.world import World, run_world, Trajectory, Reference
import numdifftools as nda
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np

class FinDiffSim(torch.autograd.Function):

    @staticmethod
    def forward(ctx, in0, in1, in2, in3, in4):
        ctx.save_for_backward(in0, in1, in2, torch.tensor(in4))
        y, p = forwardPlanarSim(in0, in1, in2, in3, in4)
        return torch.cat((torch.tensor(y).view(1,-1), torch.tensor(p).view(1,-1), 5), axis = 1)

    @staticmethod
    def backward(ctx, grad_output):
        in0, in1, in2, in4 = ctx.saved_tensors
        delta = 1e-2
        for i in range(20):
            inp = in0.clone()
            inm = in0.clone()

            inp[0,i] += delta
            inm[0,i] += -delta

            y_p, p_p = forwardPlanarSim(inp, in1, in2, False, in4, float('inf'))
            y_m, p_m = forwardPlanarSim(inm, in1, in2, False, in4, float('inf'))

            val_p = torch.cat((torch.tensor(y_p).view(1,-1), torch.tensor(p_p).view(1,-1)), axis = 1)
            val_m = torch.cat((torch.tensor(y_m).view(1,-1), torch.tensor(p_m).view(1,-1)), axis = 1)
            
            if i == 0:
                grad_input = torch.tensor((val_p - val_m)*0.5/delta)
            else:
                grad_input = torch.cat((grad_input, torch.tensor((val_p - val_m)*0.5/delta)), axis = 0)

        return torch.matmul(grad_output,torch.tensor(grad_input).T), None, None, None, None

class FinDiffSagSim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in0, in1, in2, in3, in4):
        ctx.save_for_backward(in0, in1, in2, torch.tensor(in4))
        y, p =  forwardSaggitalSim(in0, in1, in2, 0.0, 0.0, in3, in4, 5)
        return torch.cat((torch.tensor(y).view(1,-1), torch.tensor(p).view(1,-1)), axis = 1)

    @staticmethod
    def backward(ctx, grad_output):
        in0, in1, in2, in4 = ctx.saved_tensors
        delta = 1e-2
        for i in range(20):
            inp = in0.clone()
            inm = in0.clone()

            inp[0,i] += delta
            inm[0,i] += -delta

            y_p, p_p = forwardSaggitalSim(inp, in1, in2, 0.0, 0.0, False, in4, float('inf'))
            y_m, p_m = forwardSaggitalSim(inm, in1, in2, 0.0, 0.0, False, in4, float('inf'))

            val_p = torch.cat((torch.tensor(y_p).view(1,-1), torch.tensor(p_p).view(1,-1)), axis = 1)
            val_m = torch.cat((torch.tensor(y_m).view(1,-1), torch.tensor(p_m).view(1,-1)), axis = 1)
            
            if i == 0:
                grad_input = torch.tensor((val_p - val_m)*0.5/delta)
            else:
                grad_input = torch.cat((grad_input, torch.tensor((val_p - val_m)*0.5/delta)), axis = 0)

        return torch.matmul(grad_output,torch.tensor(grad_input).T), None, None, None, None

def forwardPlanarSim(p, polygon, xtraj0, render = False, gamma = 0.01, eps = 0.1):
    bodies = []
    joints = []
    facets = []
    restitution = 0 # no impacts in quasi-dynamics
    n_pol = int(polygon[0])
    scale = 2500
    cvx = False

    p = torch.tensor(p).view(4,5)
    xr = 500 + (scale * xtraj0[0])
    yr = 500 - (scale * xtraj0[1])

    # gets inputs  
    cto = setupDiff()
    dp, ddp = cto(p)
    dp = torch.clamp(dp, -0.5, 0.5)

    # adds body based on triangulation
    if cvx:
        verts = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        r0 = Hull([xr, yr], verts, restitution=restitution, fric_coeff=10, mass = 0.01, name="obj")
        r0.name = "obj"
        bodies.append(r0)

    for i in range(n_pol):
        x2 = [polygon[1+8*i], -polygon[2+8*i]]
        x1 = [polygon[3+8*i], -polygon[4+8*i]]
        x0 = [polygon[5+8*i], -polygon[6+8*i]]
        facets.append(scale*np.array([x0, x1]))
        facets.append(scale*np.array([x1, x2]))
        facets.append(scale*np.array([x2, x0]))
        if cvx:
            verts = scale*np.array([x0, x1, x2])
            p0 = np.array([xr + 1.0*polygon[7+8*i], yr - 1.0*polygon[8+8*i]])
            r1 = Hull(p0, verts, mass = 0.01/n_pol, restitution=restitution, fric_coeff=1e6, name="obj_"+str(i))
            r1.name = "obj_"+str(i)
            r1.col = (239, 188, 64)
            for j in range(i+1):
                r1.add_no_contact(bodies[j])
            joints += [FixedJoint(r1, bodies[0])]
            bodies.append(r1)

    # removes repeated facets
    exclude = []
    for i in range(len(facets)):
        for j in range(len(facets)):
            if i != j:
                if (facets[i][0][0] == facets[j][0][0]) and (facets[i][0][1] == facets[j][0][1]): 
                    if (facets[i][1][0] == facets[j][1][0]) and (facets[i][1][1] == facets[j][1][1]):
                        if j not in exclude:
                            exclude.append(j)
                if (facets[i][0][0] == facets[j][1][0]) and (facets[i][0][1] == facets[j][1][1]):
                    if (facets[i][1][0] == facets[j][0][0]) and (facets[i][1][1] == facets[j][0][1]):
                        if j not in exclude:
                            exclude.append(j)

    exclude.sort(reverse=True)

    for i in exclude:
        facets.pop(i)

    ordered_verts = []
    ordered_verts.append(facets[0][1])
    while True:
        for f in facets:
            if (f[0][0] == ordered_verts[-1][0]) and (f[0][1] == ordered_verts[-1][1]):
                ordered_verts.append(f[1])
        if len(ordered_verts) == len(facets):
            break

    # constructs object
    if cvx:
        pass
    else:
        r0 = NonConvex([xr, yr], np.array(ordered_verts), restitution=restitution, fric_coeff=1, mass = 1, name="obj")
        r0.name = "obj"
        r0.col = (239, 188, 64)
        r0.add_force(MDP(10))
        bodies.append(r0)

    # Point Fingers
    traj_f = []
    pos_f = []
    for i in range(2):
        pos0 = [500 + scale*p[i,0], 500 - scale*p[i+2,0]]
        c = Circle(pos0, 10, vel=(0, 0, 0), mass = 10, restitution=restitution, fric_coeff=0.01, name = "f"+str(i))
        c.name = "f"+str(i)
        c.col = (255,0,0)
        traj = torch.cat((scale*dp[i,:],-scale*dp[i+2,:]), axis=0).view(2,5)
        traj_f.append(Trajectory(vel = traj, name = "f"+str(i)))
        pos = torch.cat((0*p[i,:], 500 + scale*p[i,:],500 - scale*p[i+2,:]), axis=0).view(3,5)
        pos_f.append(Reference(pos = pos, name = "f"+str(i)))

        if i > 0:
            c.add_no_contact(bodies[-1])
        if cvx:
            c.add_no_contact(bodies[0])
        bodies.append(c)

    world = World(bodies, joints, dt=0.005, tol = 1e-6, eps = eps, post_stab = False, strict_no_penetration = False, facets = facets)
    world.gamma = gamma

    screen = None
    if render:
        pygame.init()
        screen = pygame.display.set_mode((1000, 1000), pygame.DOUBLEBUF)
        screen.set_alpha(None)

    run_world(world, run_time = 0.41, screen=screen, recorder=None, print_time=False, traj=traj_f, pos_f = pos_f)

    # import pdb
    # pdb.set_trace()
    for t in range(5):
        if t > 0:
            y0 = torch.cat((world.states[t][1].view(1,1) - 500, 500 - world.states[t][2].view(1,1), -scale*torch.sin(world.states[t][0].view(1,1))*0.03), axis = 0)/scale
            p0 = torch.cat((world.fingers1[t][1].view(1,1) - 500, world.fingers2[t][1].view(1,1) - 500, 500 - world.fingers1[t][2].view(1,1), 500 - world.fingers2[t][2].view(1,1)), axis = 0)/scale
            y = torch.cat((y,y0), axis = 1)
            p = torch.cat((p,p0), axis = 1)
        else:
            y = torch.cat((world.states[t][1].view(1,1) - 500, 500 - world.states[t][2].view(1,1), -scale*torch.sin(world.states[t][0].view(1,1))*0.03), axis = 0)/scale
            p = torch.cat((world.fingers1[t][1].view(1,1) - 500, world.fingers2[t][1].view(1,1) - 500, 500 - world.fingers1[t][2].view(1,1), 500 - world.fingers2[t][2].view(1,1)), axis = 0)/scale
    
    return y.clone().view(1,-1).detach().numpy(), p.clone().view(1,-1).detach().numpy()

def forwardSaggitalSim(p, polygon, xtraj0, angle = 0.0, height = 0.0, render = False, gamma = 0.01, eps = 0.1):
    bodies = []
    joints = []
    facets = []
    restitution = 0 # no impacts in quasi-dynamics
    n_pol = int(polygon[0])
    scale = 2500
    cvx = False

    p = torch.tensor(p).view(4,5)
    xr = 500 + (scale * xtraj0[0])
    yr = 500 - (scale * xtraj0[1])

    # gets inputs  
    cto = setupDiff()
    dp, ddp = cto(p)
    dp = torch.clamp(dp, -0.5, 0.5)

    # adds body based on triangulation
    if cvx:
        verts = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        r0 = Hull([xr, yr], verts, restitution=restitution, fric_coeff=10, mass = 0.01, name="obj")
        r0.name = "obj"
        r0.add_force(Gravity(g=100))
        bodies.append(r0)

    for i in range(n_pol):
        x2 = [polygon[1+8*i], -polygon[2+8*i]]
        x1 = [polygon[3+8*i], -polygon[4+8*i]]
        x0 = [polygon[5+8*i], -polygon[6+8*i]]
        facets.append(scale*np.array([x0, x1]))
        facets.append(scale*np.array([x1, x2]))
        facets.append(scale*np.array([x2, x0]))
        if cvx:
            verts = scale*np.array([x0, x1, x2])
            p0 = np.array([xr + 1.0*polygon[7+8*i], yr - 1.0*polygon[8+8*i]])
            r1 = Hull(p0, verts, mass = 0.01/n_pol, restitution=restitution, fric_coeff=1, name="obj_"+str(i))
            r1.name = "obj_"+str(i)
            r1.col = (239, 188, 64)
            for j in range(i+1):
                r1.add_no_contact(bodies[j])
            joints += [FixedJoint(r1, bodies[0])]
            bodies.append(r1)

    # removes repeated facets
    exclude = []
    for i in range(len(facets)):
        for j in range(len(facets)):
            if i != j:
                if (facets[i][0][0] == facets[j][0][0]) and (facets[i][0][1] == facets[j][0][1]): 
                    if (facets[i][1][0] == facets[j][1][0]) and (facets[i][1][1] == facets[j][1][1]):
                        if j not in exclude:
                            exclude.append(j)
                if (facets[i][0][0] == facets[j][1][0]) and (facets[i][0][1] == facets[j][1][1]):
                    if (facets[i][1][0] == facets[j][0][0]) and (facets[i][1][1] == facets[j][0][1]):
                        if j not in exclude:
                            exclude.append(j)

    exclude.sort(reverse=True)

    for i in exclude:
        facets.pop(i)

    ordered_verts = []
    ordered_verts.append(facets[0][1])
    while True:
        for f in facets:
            if (f[0][0] == ordered_verts[-1][0]) and (f[0][1] == ordered_verts[-1][1]):
                ordered_verts.append(f[1])
        if len(ordered_verts) == len(facets):
            break

    # constructs object
    if cvx:
        pass
    else:
        r0 = NonConvex([xr, yr], np.array(ordered_verts), restitution=restitution, fric_coeff=1, mass = 1, name="obj")
        r0.name = "obj"
        r0.add_force(Gravity(g=scale/2))
        r0.col = (239, 188, 64)
        bodies.append(r0)

    # Point Fingers
    traj_f = []
    pos_f = []
    for i in range(2):
        pos0 = [500 + scale*p[i,0], 500 - scale*p[i+2,0]]
        c = Circle(pos0, 10, vel=(0, 0, 0), mass = 100, restitution=restitution, fric_coeff=1, name = "f"+str(i))
        c.name = "f"+str(i)
        c.col = (255,0,0)
        c.prev = c.p
        c.prev2 = c.p
        traj_f.append(Trajectory(vel = torch.cat((scale*dp[i,:],-scale*dp[i+2,:]), axis=0).view(2,5), name = "f"+str(i)))
        pos = torch.cat((0*p[i,:], 500 + scale*p[i,:],500 - scale*p[i+2,:]), axis=0).view(3,5)
        pos_f.append(Reference(pos = pos, name = "f"+str(i)))
        if i > 0:
            c.add_no_contact(bodies[-1])
        if cvx:
            c.add_no_contact(bodies[0])
        # c.add_force(FingerTrajectory(torch.cat((0*dp[i,:],scale*dp[i,:],-scale*dp[i+2,:]), axis=0).view(3,5)))
        bodies.append(c)

    width = 10

    r = Rect([0, 500, 500 + width - scale*height], [1200, 2*width], mass = 10, restitution=0, fric_coeff=1, name = "env")
    r.name = "env"
    r.col = (0,0,0)
    r.add_no_contact(bodies[-1])
    r.add_no_contact(bodies[-2])
    bodies.append(r)
    joints.append(TotalConstraint(r))

    for i in range(2):
        pos0 = [500 + scale*p[i,0], 500 - scale*p[i+2,0]]
        c = Circle(pos0, 10, vel=(0, 0, 0), mass = 1, restitution=restitution, fric_coeff=1, name = "0f"+str(i))
        c.name = "0f"+str(i)
        c.col = (0,0,255)
        traj_f.append(Trajectory(vel = torch.cat((scale*dp[i,:],-scale*dp[i+2,:]), axis=0).view(2,5), name = "0f"+str(i)))
        pos = torch.cat((0*p[i,:], 500 + scale*p[i,:],500 - scale*p[i+2,:]), axis=0).view(3,5)
        pos_f.append(Reference(pos = pos, name = "0f"+str(i)))

        for b in bodies:
            c.add_no_contact(b)

        bodies.append(c)

    world = World(bodies, joints, dt=0.02, tol = 1e-6, eps = eps, post_stab = False, strict_no_penetration = False, facets = facets)
    world.gamma = gamma

    screen = None
    if render:
        pygame.init()
        screen = pygame.display.set_mode((1000, 1000), pygame.DOUBLEBUF)
        screen.set_alpha(None)

    run_world(world, run_time = 0.41, screen=screen, recorder=None, print_time=False, traj=traj_f, pos_f = pos_f)

    # import pdb
    # pdb.set_trace()
    for t in range(5):
        if t > 0:
            y0 = torch.cat((world.states[t][1].view(1,1) - 500, 500 - world.states[t][2].view(1,1), -scale*torch.sin(world.states[t][0].view(1,1))*0.03), axis = 0)/scale
            p0 = torch.cat((world.fingers1[t][1].view(1,1) - 500, world.fingers2[t][1].view(1,1) - 500, 500 - world.fingers1[t][2].view(1,1), 500 - world.fingers2[t][2].view(1,1)), axis = 0)/scale
            y = torch.cat((y,y0), axis = 1)
            p = torch.cat((p,p0), axis = 1)
        else:
            y = torch.cat((world.states[t][1].view(1,1) - 500, 500 - world.states[t][2].view(1,1), -scale*torch.sin(world.states[t][0].view(1,1))*0.03), axis = 0)/scale
            p = torch.cat((world.fingers1[t][1].view(1,1) - 500, world.fingers2[t][1].view(1,1) - 500, 500 - world.fingers1[t][2].view(1,1), 500 - world.fingers2[t][2].view(1,1)), axis = 0)/scale
    
    return y.clone().view(1,-1).detach().numpy(), p.clone().view(1,-1).detach().numpy()

def setupDiff():
        # decision variables
        dr = cp.Variable((4, 5))
        ddr = cp.Variable((4, 5))

        # parameters
        r = cp.Parameter((4, 5))
        
        # adds finite-diff constraints
        constraints = []
        for t in range(5):
            for d in range(4):
                if t == 0:
                    constraints.append(ddr[d,t]*(0.1**2) == 0)
                    constraints.append(dr[d,t]*(0.1) == r[d,t+1] - r[d,t])
                elif t == 4:
                    constraints.append(ddr[d,t]*(0.1**2) == 0)
                    constraints.append(dr[d,t]*(0.1) == 0)
                else:
                    constraints.append(ddr[d,t]*(0.1**2) == r[d,t-1] - 2*r[0,t] + r[d,t+1])
                    constraints.append(dr[d,t]*(0.1) == r[d,t+1] - r[d,t])

        objective = cp.Minimize(cp.pnorm(ddr, p=2))
        problem = cp.Problem(objective, constraints)
        
        return CvxpyLayer(problem, parameters=[r], variables=[dr, ddr])
