import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.FinDiffSim import *
import pdb
import numpy as np

# creates inputs
p = torch.tensor(np.zeros((4,5))).double().view(4,5)

pols = np.zeros(9)
pols[0] = 1

pols[1] = 0.1
pols[2] = 0.1

pols[3] = -0.1
pols[4] = 0.1

pols[5] = 0.0
pols[6] = -0.1

xtraj0 = torch.tensor(np.zeros((3))).double().view(3)

# inputz
ins = []

ins.append(p)
ins.append(pols)
ins.append(xtraj0)
ins.append(False)
ins.append(0.01)

# defines simulator
sim = FinDiffSim()
out = sim.apply(ins)

pdb.set_trace()
