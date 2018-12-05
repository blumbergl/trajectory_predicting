
import numpy as np
import random

colors = ["red","green","blue","black"]

color2charge = {"red":0, "green":1, "blue":-1, "black":5}

def makeScene(n_objects = 5):
    #Objects = []
    Positions = np.matrix(np.random.randn(2, n_objects)) * 2
    Velocities = np.matrix(np.random.randn(2, n_objects))
    Colors = [random.choice(colors) for i in range(n_objects)]

    ColorVecs = np.zeros((len(colors), n_objects))
    for i,color in enumerate(Colors):
        ColorVecs[colors.index(color), i] = 1.0

    Objects = np.concatenate([Positions,Velocities,ColorVecs])
    
    Accel = [np.matrix([[0],[-1.0]]) for i in range(n_objects)]
    for i in range(n_objects):
        for j in range(n_objects):
            if i == j: continue
            # Calculate force ON i FROM j
            Pi = Positions[:,i]
            Pj = Positions[:,j]
            Pdiff = Pj - Pi
            Fv = Pdiff / np.linalg.norm(Pdiff)**3
            Accel[i] += -Fv * color2charge[Colors[i]] * color2charge[Colors[j]]

    return Objects, Accel
