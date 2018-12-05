
import tensorflow as tf
import random
from itertools import combinations

# b/c AdamOptimizer doesn't work for me:
Optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.005)
sess = tf.Session()

def trainNN(inputs, targetOutputs, layers, steps=100):
    realOutputs = inputs
    for L in layers: realOutputs = L(realOutputs)
    loss = tf.losses.mean_squared_error(realOutputs, targetOutputs)
    trainstep = Optimizer.minimize(loss)
    for i in range(steps):
        NoneVal, currentLoss = sess.run([trainstep, loss])
        if i%100 == 99: print("Iteration", i, "loss:", currentLoss)

def evalNN(inputs, layers):
    realOutputs = inputs
    for L in layers: realOutputs = L(realOutputs)
    return sess.run(realOutputs)

def NNLayer(inputSize, outputSize, name=None):
    if name == None: name = str(random.random())
    W = tf.get_variable(name=name, shape=[outputSize, inputSize])
    bias = tf.get_variable(name=name+"bias", shape=[outputSize, 1])

    sess.run(W.initializer)
    sess.run(bias.initializer)

    return lambda tensor: leakyRELU(tf.matmul(W, tensor) + bias)

def leakyRELU(tensor):
    return tensor/2 + tf.nn.relu(tensor)/2



if __name__ == "__main__":
    L = NNLayer(3, 4)

    inputs = tf.constant([[1,4],[2,0],[3,-1]], dtype='float32')
    outputs = tf.constant([[4,0],[5,-6],[0,2],[-1,0]], dtype='float32')

    trainNN(inputs, outputs, layers = [L], steps=100)
    print(evalNN(inputs, layers = [L]))
