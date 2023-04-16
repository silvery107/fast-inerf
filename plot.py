import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

## prepare file 
# parser = argparse.ArgumentParser(description='saved training history')
# parser.add_argument('filename', type=str, help='name of the file to be plotted')
# args = parser.parse_args()

# filename = args.filename

## load data and read
filenames = ["obs_imgs_2_trans_x_rot_mask", "obs_imgs_2_trans_x_rot"]
colors = ['r', 'b', 'g']
sns.set_theme()

fig1, ax1 = plt.subplots(dpi=500)
fig2, ax2 = plt.subplots(dpi=500)
fig3, ax3 = plt.subplots(dpi=500)

for ii in range(len(filenames)):
    training_history = np.loadtxt(f"logs/{filenames[ii]}.csv", delimiter=",", dtype=float)

    ks1 = training_history[1:, 0]
    losses = training_history[1:, 1] 
    if(training_history.shape[0] > 20):
        ks2 = training_history[1::20, 0]
        rot_errors = training_history[1::20, 2]
        translation_errors = training_history[1::20, 3]
    else:
        ks2 = training_history[1:, 0]
        rot_errors = training_history[1:, 2]
        translation_errors = training_history[1:, 3]   

    ax1.plot(ks1, losses, colors[ii], lw=2, label=f"{ii}")
    ax2.plot(ks2, rot_errors, colors[ii], lw=2, label=f"{ii}")
    ax3.plot(ks2, translation_errors, colors[ii], lw=2, label=f"{ii}")


ax1.set_xlabel('iterations', fontsize=16)
ax1.set_ylabel('loss', fontsize=16)
ax1.set_title("Training Loss", fontsize=16)
ax1.legend(fontsize=16)
fig1.savefig(f'logs/traing_loss.png')

ax2.set_xlabel('iterations', fontsize=16)
ax2.set_ylabel('error, degree', fontsize=16)
ax2.set_title("Rotation Error", fontsize=16)
ax2.legend(fontsize=16)
fig2.savefig(f'logs/rotationError.png')

ax3.set_xlabel('iterations', fontsize=16)
ax3.set_ylabel('error, meter', fontsize=16)
ax3.set_title("Translation Error", fontsize=16)
ax3.legend(fontsize=16)
fig3.savefig(f'logs/translationError.png')