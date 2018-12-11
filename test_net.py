# This will be a file that generates a ground truth video/picture and a version as predicted by the neural net. It will be used for comparison.

from NNunits import sess, Optimizer, NN, tf
from new_synthetic_sim import DifferentParticlesSim
from phizzy import TotalLoss, GetObjectData, PredictedAccelerationsPacked
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, default=5000,
                    help='Number of epochs to train.')
                    # I've usually been training with 100000, but this is a good default for a shorter test.
parser.add_argument('--sf', type=int, default=100,
                    help='Number of data points generated for training.')
                    # Increasing this seems like a good way to increase the amount of time training takes without getting better results?
parser.add_argument('--n', type=float, default=3,
                    help='The learning rate.')
                    # Easy case: .1 diverges, .001 is too slow. Goldylocks likes .01.
                    # Harder case: .01 diverges, but .001 gets stuck ~50 after 10000 trials.
args = parser.parse_args()

T = args.t
sample_freq = args.sf
n_balls = args.n


# Load the Neural Nets
numColors = 2

PairNet = NN([8+2*numColors, 14, 14, 2])
SoloNet = NN([4+numColors, 8, 8, 2])

SoloNet.loadFromFile('solo.npz')
PairNet.loadFromFile('pair.npz')


# Generate "Ground Truth" Case
sim = DifferentParticlesSim(n_balls = n_balls)
loc, vel, colors = sim.sample_trajectory(T=T, sample_freq=sample_freq)
colors = np.array(colors, dtype='float32')


# Use neural nets to generate data
counter = 0
T_save = int(T / sample_freq - 1)
NNloc = np.zeros((T_save, 2, n_balls))
NNvel = np.zeros((T_save, 2, n_balls))

loc_next = np.array(loc[0], dtype='float32')
vel_next = np.array(vel[0], dtype='float32')
ObjectData = np.vstack([loc_next, vel_next, colors.T])
for i in range(1, T):
    if i % sample_freq == 0:
        NNloc[counter, :, :], NNvel[counter, :, :] = loc_next, vel_next
        counter += 1
        print(counter, '/', T/sample_freq, 'points of data')

    # TODO: Why does this line get slower and slower??? :(
    accel = sess.run(PredictedAccelerationsPacked(ObjectData, PairNet, SoloNet)) / 1000
    loc_next += .001 * vel_next
    vel_next += .001 * accel
    ObjectData[:2,:] = loc_next
    ObjectData[2:4,:] = vel_next

# Create Visuals!
print(colors)

plt.figure()
axes = plt.gca()
axes.set_xlim([-5., 5.])
axes.set_ylim([-5., 5.])
for i in range(loc.shape[-1]):
    plt.plot(loc[:, 0, i], loc[:, 1, i])
    plt.plot(loc[0, 0, i], loc[0, 1, i], 'd')
plt.savefig('groundTruth.png')

plt.figure()
axes = plt.gca()
axes.set_xlim([-5., 5.])
axes.set_ylim([-5., 5.])
for i in range(NNloc.shape[-1]):
    plt.plot(NNloc[:, 0, i], NNloc[:, 1, i])
    plt.plot(NNloc[0, 0, i], NNloc[0, 1, i], 'd')
plt.savefig('NNGenerated.png')
plt.show()
