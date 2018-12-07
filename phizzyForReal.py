

from phizzy import TotalLoss
from NNunits import sess, Optimizer, NN
from new_synthetic_sim import DifferentParticlesSim
import random

numColors = 3

numberOfDataPoints = 100
numberOfTrainingEpochs = 10000
learningRate = 0.001 # ?????? what should this be ???????

PairNet = NN([8+2*numColors, 14, 14, 2])
SoloNet = NN([4+numColors, 8, 8, 2])

# Assemble a list of loss functions:
lossFunctions = []

for i in range(numberOfDataPoints):
    DSP = DifferentParticlesSim(n_balls = random.randint(1, 3))
    loc,vel,colors = DSP.sample_trajectory(T=30) # one time step
    loss = TotalLoss(loc, vel, colors, PairNet, SoloNet)
    lossFunctions.append(loss)
    print(i, end=" ")

# Train, I guess???
avgLoss = sum(lossFunctions)/len(lossFunctions)
trainStep = Optimizer(learning_rate=learningRate).minimize(avgLoss)

try:
    print("""Training time. Note: the first step takes longer than
    the rest for some reason. """)
    for i in range(numberOfTrainingEpochs):
        noneValue, currentLoss = sess.run([trainStep, avgLoss])
        if i%100==99: print(i, currentLoss)
except KeyboardInterrupt:
    pass

# Save our juicy NNs
do something with PairNet.save()
do something with SoloNet.save()
