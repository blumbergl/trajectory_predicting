

from NNunits import sess, Optimizer, NNLayer, evalNN, trainNN
from phizzyData import makeScene
import numpy as np
import itertools
import tensorflow as tf
from collections import namedtuple

if __name__ == "__main__" and 0:
    L = NNLayer(3, 4)

    inputs = [[1.0,4],[2,0],[3,-1]]
    outputs = [[4.0,0],[5,-6],[0,2],[-1,0]]

    trainNN(inputs, outputs, layers = [L], steps=100)
    print(evalNN(inputs, layers = [L]))



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

def PredictedAccelerations(Objects, PairNet, SoloNet):
    """ Outputs a dict
        {O -> predicted acceleration on O}
    for O in objects. """
    def predictedAccel(i):
        O = Objects[i]
        otherObjects = [Objects[j] for j in range(len(Objects)) if j != i]
        
        colVector = O.vector
        otherColVectors = [Q.vector for Q in otherObjects]

        pairVecs = [np.concatenate([colVector,ocv]) for ocv in otherColVectors]
        
        # gimme column vectors
        pairVecs = [np.concatenate([O.vector, T.vector]) for T in otherObjects]
        return sum(PairNet(pv) for pv in pairVecs) + SoloNet(colVector)
        
    return [predictedAccel(i) for i in range(len(Objects))]

if __name__ == "__main__":
    n = 3
    O, A = makeScene(n)
    O = np.matrix(O, dtype='float32')
    Object = namedtuple("Object", ["vector"])
    Scene = [Object(O[:,i]) for i in range(n)]
    PairNet = NNLayer(16, 2, "Pair_Net")
    SoloNet = NNLayer(8, 2, "Solo_Net")
    PA = PredictedAccelerations(Scene, PairNet, SoloNet)
    loss = sum(tf.losses.mean_squared_error(pai,ai) for (pai,ai) in zip(PA,A))
    trainstep = Optimizer.minimize(loss)
    
