

from phizzy import TotalLoss, GetObjectData, PredictedAccelerationsPacked
from NNunits import sess, Optimizer, NN
from new_synthetic_sim import DifferentParticlesSim
import random
import tensorflow as tf

numColors = 3

numberOfDataPoints = 100
numberOfTrainingEpochs = 10000
learningRate = 0.001 # ?????? what should this be ???????

PairNet = NN([8+2*numColors, 14, 14, 2])
SoloNet = NN([4+numColors, 8, 8, 2])

# Assemble a list of loss functions:
lossFunctions = []
# Also keep track of a baseline loss (loss from predicting 0 always):
lossOfZeroFn = 0

print("Gathering data...")
for i in range(numberOfDataPoints):
    DSP = DifferentParticlesSim(n_balls = random.randint(1, 3))
    loc,vel,colors = DSP.sample_trajectory(T=30) # two time steps
    loss = TotalLoss(loc, vel, colors, PairNet, SoloNet)
    lossFunctions.append(loss)

    # How good is the "zero prediction"?
    for ObjectData, RealAcceleration in GetObjectData(loc, vel, colors):
        zeroPred = tf.zeros(RealAcceleration.shape)
        lossOfZeroFn += tf.losses.mean_squared_error(zeroPred,
                                                     1000*RealAcceleration)
    print(i, end=" ")

print("Loss of predicting zero:", sess.run(lossOfZeroFn/len(lossFunctions)))

# Train, I guess???
avgLoss = sum(lossFunctions)/len(lossFunctions)
trainStep = Optimizer(learning_rate=learningRate).minimize(avgLoss)

try:
    print("""Training time. Note: the first step takes longer than
    the rest for some reason. """)
    for i in range(numberOfTrainingEpochs):
        noneValue, currentLoss = sess.run([trainStep, avgLoss])
        if i%100==99: print("Iteration:", i, "Loss:", currentLoss)
except KeyboardInterrupt:
    pass

# Let's print a test:
DSP = DifferentParticlesSim(n_balls = 3)
loc, vel, colors = DSP.sample_trajectory(T=30)
for ObjectData, RealAcceleration in GetObjectData(loc, vel, colors):
    print("Real accelerations:")
    print(RealAcceleration)
    print("And the prediction:")
    pred = sess.run(PredictedAccelerationsPacked(ObjectData, PairNet, SoloNet)) 
    print(pred/1000)
'''
# Save our juicy NNs
do something with PairNet.save()
do something with SoloNet.save()'''
