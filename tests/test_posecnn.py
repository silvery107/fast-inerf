import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from pose_cnn import train_posecnn, eval_posecnn
from utils.posecnn_utils import reset_seed



if __name__ == "__main__":
    reset_seed(0)
    # train_posecnn("data/")
    eval_posecnn("data/")