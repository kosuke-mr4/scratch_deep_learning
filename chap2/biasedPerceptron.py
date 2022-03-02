from tkinter import W
from xml.dom.expatbuilder import theDOMImplementation
import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    theta = 0
    if tmp <= theta:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])

    # Weight and Bias are different from AND
    w = np.array([-0.5, -0.5])
    b = 0.7

    tmp = np.sum(w * x) + b
    theta = 0
    if tmp <= theta:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])

    # Weight and Bias are different from AND
    w = np.array([0.5, 0.5])
    b = -0.2

    tmp = np.sum(w * x) + b
    theta = 0
    if tmp <= theta:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


print("AND :", AND(0, 0), AND(1, 0), AND(0, 1), AND(1, 1))

print("NAND :", NAND(0, 0), NAND(1, 0), NAND(0, 1), NAND(1, 1))

print("OR :", OR(0, 0), OR(1, 0), OR(0, 1), OR(1, 1))

print("XOR :", XOR(0, 0), XOR(1, 0), XOR(0, 1), XOR(1, 1))
