

import numpy as np
from collections import namedtuple


class Thing:
    def __init__(self):
        self.position =
        self.velocity =

    def Update(self, Rule):
        if Rule.root == AccelerateXY:
            ax, ay = Rule.children
            self.velocity += [ax, ay]


def RuleToFunc(Rule):
    raise NotImplemented


RuleNode = namedtuple("RuleNode", ("returnType", "childTypes"))

Result, Number = "Result", "Number"

AccelerateXY = RuleNode(Result, [Number, Number])
Spacebar = RuleNode(Boolean, [])
VecPlus = RuleNode(Vector, [Vector, Vector])
