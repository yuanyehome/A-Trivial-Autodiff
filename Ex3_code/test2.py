# test more functions
import op
import numpy as np
import math as m
from numerical import Df


x = op.Value("x", 2)


def test1(x):
    return m.log(x) / x


def test2(x):
    return m.log(x) / x - m.exp(x) * m.sin(x)


def test3(x):
    return 1 / (1 + m.exp(-x))


def test4(x):
    return 1 / (1 + m.exp(-m.sin(x)))


mytest1 = op.log(x) / x
mytest2 = op.log(x) / x - op.exp(x) * op.sin(x)
mytest3 = op.sigmoid(x)
mytest4 = op.sigmoid(op.sin(x))

print("test1 numerical: %f" % (Df(test1, 2)))
print("test1 myfunc   : %f" % (mytest1.Df("x")))

print("test2 numerical: %f" % (Df(test2, 2)))
print("test2 myfunc   : %f" % (mytest2.Df("x")))

print("test3 numerical: %f" % (Df(test3, 2)))
print("test3 myfunc   : %f" % (mytest3.Df("x")))

print("test4 numerical: %f" % (Df(test4, 2)))
print("test4 myfunc   : %f" % (mytest4.Df("x")))
