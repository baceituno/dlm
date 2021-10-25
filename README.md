A Differentiable Recipe for Learning Visual Non-Prehensile Planar Manipulation
===================================================

This repository contains the source code of the paper [A Differentiable Recipe for Learning Visual Non-Prehensile Planar Manipulation](https://openreview.net/forum?id=f7KaqYLO3iE).

# Content

The code contains five sets of experiments:
```
1. Fully connneced Neural Network Trained on robot actions (Exp_NN)
2. Fully connneced Neural Network Trained on differnetiable simulator (Exp_NNM)
3. Differentiable Pipaline Trained on machanical parameters (Exp_MDR)
4. Differentiable Pipaline Trained on machanical parameters and CVX Layer (Exp_CVX)
5. Differentiable Pipaline Trained on end to end with simulation (Exp_DLM)
```
# Installation

Install the dependencies:

1. Pytorch 1.4.0 (https://pytorch.org/)
2. CVXPy Layers (https://github.com/cvxgrp/cvxpylayers)
2. PyGame (https://www.pygame.org/wiki/GettingStarted)
2. PyODE (http://pyode.sourceforge.net/)

To install the simulator, also install:

```
cd ./src/diffsim_lcp
python setup.py install --user
```

Tested in Python 3.7.7

# Usage 

## Example 1: Fully connneced Neural Network Trained on robot actions

```
Coming soon
```

# Citing
If you find this repository helpful in your publications, please cite the following:

```
@inproceedings{aceituno2021corl,
    title={A Differentiable Recipe for Learning Visual Non-Prehensile Planar Manipulation },
    author={B. Aceituno, and A. Rodriguez, and S. Tulsiani, and A. Gupta, and M. Mukadam},
    booktitle={CoRL},
    year={2021}
}
```

# License
This repository is licensed under the [MIT License](LICENSE.md).