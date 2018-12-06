

from NNunits import sess, Optimizer, NNLayer, evalNN, trainNN

import numpy as np
import itertools
import tensorflow as tf
from collections import namedtuple

from new_synthetic_sim import DifferentParticlesSim


def Train(Objects, RealAccelerations, PairNet, SoloNet, steps=10):
    # RealAccelerations must be column vectors
    loss = sum(tf.losses.mean_squared_error(PredAccel,RealAccel)
        for (PredAccel,RealAccel) in zip(PredAccelerations,RealAccelerations))
    trainstep = Optimizer.minimize(loss)
    for i in range(steps): sess.run(trainstep)

def SomeObjects():
    Object = namedtuple("Object", ["vector"])

    RedBall = Object(np.array([0, 0, -1, 1.0, 1, 0, 0, 0],dtype='float32'))
    BlueBall= Object(np.array([1, 3, 1, 0.0, 0, 1, 0, 0],dtype='float32'))
    GreenBall=Object(np.array([-1,2,-1,-2.0, 0, 0, 1, 0],dtype='float32'))

    Objects = [RedBall,BlueBall,GreenBall]

    PairNet = NNLayer(16, 2, "Pairwise_interactions")
    SoloNet = NNLayer(8, 2, "Solo_acceleration")

def Losses(loc, vel, colors, PairNet, SoloNet):
    # Convert everything to float32 because otherwise tf yells at me:
    loc = np.array(loc, dtype='float32')
    vel = np.array(vel, dtype='float32')
    colors = np.array(colors, dtype='float32')
    # First, use the velocity matrix to get acceleration:
    acceleration = vel[1:,:] - vel[:-1,:]
    # And throw away the last timestep, b/c we don't know acceleration then:
    loc = loc[:-1,:]
    vel = vel[:-1,:]
    losses = []
    for t in range(len(loc)):
        RealAcceleration = acceleration[t]
        # Object data at time t gets encoded into columns of this matrix:
        ObjectData = np.vstack([loc[t], vel[t], colors.T])
        PredictedAccels = PredictedAccelerations(ObjectData, PairNet, SoloNet)

        loss = tf.losses.mean_squared_error(RealAcceleration,
                                            PredictedAccels)
        losses.append(loss)
        print(t,end=" ")
    return losses
        

def PredictedAccelerations(ObjectData, PairNet, SoloNet):
    """ Outputs a 2-by-(# of objects) tensor: ith column is predicted accel
    of ith object """
    numObjects = ObjectData.shape[1]
    def predictedAccel(i):
        O = ObjectData[:,i:i+1] # get ith column vector = data of ith object

        # get list of other column vectors (data of other objects):
        otherObjects = [ObjectData[:,j:j+1]
                        for j in range(numObjects) if j != i]

        # concatenate object data to get NN input
        pairVecs = [np.concatenate([O,Other]) for Other in otherObjects]
        # run neural network on input, then add each acceleration:
        return sum(PairNet(pv) for pv in pairVecs) + SoloNet(O)

    predAccels = [predictedAccel(i) for i in range(numObjects)]
    # Stack em up into the right shape (I don't get axes lol):
    return tf.stack(predAccels,axis=2)[:,0,:]


if __name__ == "__main__":
    PairNet = NNLayer(14, 2)
    SoloNet = NNLayer(7, 2)

    DSP = DifferentParticlesSim()
    loc, vel, colors = DSP.sample_trajectory(T=100)
    losses = Losses(loc, vel, colors, PairNet, SoloNet)

    totalLoss = sum(losses)
    trainstep = Optimizer(learning_rate=0.005).minimize(totalLoss)

    print("""Time for training. The first step takes a lot longer than the
          rest, for some reason.""")
    # Run the following line infinitely many times:
    sess.run([trainstep, totalLoss])
    
