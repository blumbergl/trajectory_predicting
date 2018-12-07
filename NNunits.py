
import tensorflow as tf
import random
from itertools import combinations

# b/c AdamOptimizer doesn't work for me:
Optimizer = tf.train.GradientDescentOptimizer
sess = tf.Session()

class NN():
    def __init__(self, layerSizes):
        # list of (W, bias, layer function)
        self.layers = newNNLayers(layerSizes)

    def __call__(self, inputs):
        outputs = inputs
        for W, bias, layerFn in self.layers:
            outputs = layerFn(outputs)
        return outputs

    def save(self):
        return [(sess.run(W),sess.run(bias))
                for W, bias, layerFn in self.layers]


def NNLayer(inputSize, outputSize, name=None):
    if name == None: name = str(random.random())
    W = tf.get_variable(name=name, shape=[outputSize, inputSize])
    bias = tf.get_variable(name=name+"bias", shape=[outputSize, 1])

    sess.run(W.initializer)
    sess.run(bias.initializer)

    return W, bias, lambda tensor: leakyRELU(tf.matmul(W, tensor) + bias)

def newNNLayers(layerSizes, name=None):
    if name == None: name = str(random.random())
    layers = []
    for i in range(len(layerSizes)-1):
        inSize, outSize = layerSizes[i], layerSizes[i+1]
        L = NNLayer(inSize, outSize, name = name+str(i))
        layers.append(L)
    return layers

def leakyRELU(tensor):
    return tensor/2 + tf.nn.relu(tensor)/2



if __name__ == "__main__":
    MyNet = NN([3, 4])

    inputs = tf.constant([[1,4],[2,0],[3,-1]], dtype='float32')
    outputs = tf.constant([[4,0],[5,-6],[0,2],[-1,0]], dtype='float32')

    print(sess.run(MyNet(inputs)))
