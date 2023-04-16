import numpy as np
from numpy import savetxt
from time import time
import matplotlib.pyplot as plt

## load data and read
filename = ""
training_history = np.loadtxt(f"logs/{filename}.csv", delimiter=",", dtype=float)
ks = training_history[1:, 0]
losses = training_history[1:, 1]
rot_errors = training_history[1:, 2]
translation_errors = training_history[1:, 3]

# plot loss
plt.figure()
plt.plot(ks, losses)
plt.xlabel('iterations')
plt.xlabel('loss')
plt.title("Training Loss")

t1 = int(time())
plt.savefig(f'logs/{t1}_loss.png')

# plot rotation error
plt.figure()
plt.plot(ks, rot_errors)
plt.xlabel('iterations')
plt.xlabel('error, degree')
plt.title("Rotation Error")

t2 = int(time())
plt.savefig(f'logs/{t2}_rotationError.png')

# plot translation error
plt.figure()
plt.plot(ks, translation_errors)
plt.xlabel('iterations')
plt.xlabel('error, meter')
plt.title("Translation Error")

t3 = int(time())
plt.savefig(f'logs/{t3}_translationError.png')