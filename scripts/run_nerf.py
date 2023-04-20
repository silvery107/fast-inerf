import os
import inspect
import torch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from nerf import train


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()