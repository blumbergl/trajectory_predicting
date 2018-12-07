

from NNunits import sess, Optimizer, NNLayer, evalNN, trainNN

import numpy as np
import itertools
import tensorflow as tfs
from collections import namedtuple

from new_synthetic_sim import DifferentParticlesSim


def GetObjectData(loc, vel, colors):
    """ A generator. Yields tuples (ObjectData, RealAcceleration):
        - ObjectData is a matrix whose columns encode data for objects in
          a scene at time t.
        - RealAcceleration is a matrix whose length-2 columns are the
          correct accelerations for those objects at time t.

    TODO, maybe: instead of yielding results one time step at a time,
    handle everything in one big tensor. (Would require changing how
    PredictedAccelerationsPacked works, but not *too* hard, I think)
    """
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
        yield ObjectData, acceleration[t]

def Loss(ObjectData, RealAcceleration, PairNet, SoloNet):
    """ Loss function at time step t. """
    PredAccels = PredictedAccelerationsPacked(ObjectData, PairNet, SoloNet)
    return tf.losses.mean_squared_error(RealAcceleration, PredAccels)

def TotalLoss(loc, vel, colors, PairNet, SoloNet):
    """ Sum over all times t of the loss function at time step t."""
    losses = []
    for (ObjectData, RealAcceleration) in GetObjectData(loc, vel, colors):
        losses.append(Loss(ObjectData, RealAcceleration, PairNet, SoloNet))
    return sum(losses)


def PredictedAccelerationsPacked(ObjectData, PairNet, SoloNet):
    """ Returns a tensor whose length-2 columns encode our predictions
    for the accelerations for the objects in a scene. """
    numObjects = ObjectData.shape[1]

    # For each pair (i, j), we need to look at how object j affects object i.
    # So this is a list of all pairs (i, j).
    NNinputIndexes = list(itertools.permutations(range(numObjects), 2))

    def pairVec(i,j):
        """ Gives the input to PairNet, for which the intended output is:
        "What force does object j apply on object i?" """
        coli = ObjectData[:,i:i+1]
        colj = ObjectData[:,j:j+1]
        return np.concatenate([coli, colj])

    # Put our neural net inputs into columns of one big matrix:
    PairVecs = np.hstack([pairVec(i,j) for i,j in NNinputIndexes])
    # Apply the neural net:
    PredForces = PairNet(PairVecs)

    # OK, now we need to add some results -- e.g. for object 0, we need
    # to look at the outputs for all pairs (0, 1), (0, 2), ..., to get
    # the total predicted acceleration of object 0.
    # We do this using one matrix multiplication.
    # Sparse matrices, to be fast/fancy.
    sparseIndices = [(i,a) for i,(a,b) in enumerate(NNinputIndexes)]
    ForceAdder = tf.sparse.SparseTensor(sparseIndices,
                                        [np.float32(1)]*len(sparseIndices),
                                        (len(sparseIndices), 5))
    # And add 'em up:
    # TODO: Use tf.nn.embedding_lookup instead???
    R = tf.sparse.matmul(ForceAdder, PredForces, adjoint_a=True,adjoint_b=True)
    PredPairAccel = tf.transpose(R) # columns of this are length-2 vectors

    # Finally, get Solo accels:
    PredSoloAccel = SoloNet(ObjectData)
    return PredPairAccel + PredSoloAccel


if __name__ == "__main__":
    # TODO: Make these neural nets more than ONE LAYER deep.
    PairNet = NNLayer(14, 2)
    SoloNet = NNLayer(7, 2)

    # TODO: We need multiple runs of DSP to get training data.
    DSP = DifferentParticlesSim()
    loc, vel, colors = DSP.sample_trajectory(T=1000)
    loss = TotalLoss(loc, vel, colors, PairNet, SoloNet)

    # If the loss function diverges wildly, decrease the learning rate.
    # Higher learning rates mean faster learning, though.
    trainstep = Optimizer(learning_rate=0.0005).minimize(loss)

    print("""Time for training. The first step takes a lot longer than the
          rest, for some reason.""")
    # Run the following line infinitely many times:
    sess.run([trainstep, loss])
    
