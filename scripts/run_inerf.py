import os
import inspect
import torch
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from inerf import run_inerf


if __name__=='__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument("overlay", action="store_true")
    paser.add_argument("debug", action="store_false")
    args = paser.parse_args()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run_inerf(_overlay=args.overlay, _debug=args.debug)