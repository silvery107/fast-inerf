import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import argparse
from pose_cnn import train_posecnn, eval_posecnn
from utils.posecnn_utils import reset_seed

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--eval", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    reset_seed(0)
    if args.train:
        print("Training PoseCNN")
        train_posecnn("data/")
    else:
        print("Evaluating PoseCNN")
        eval_posecnn("data/")