import os
import inspect
import torch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from inerf import run_inerf


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run_inerf(_overlay=False, _debug=False)