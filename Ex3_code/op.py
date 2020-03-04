import numpy as np
from numerical import Df
import math as m


class Op:
    '''
    Designed as all ops' base class
    '''

    def __init__(self):
        pass

    def __add__(self, other):
        if not isinstance(other, Op):
            raise ValueError
        node = Add()
        node.setValue(self, other)
        return node

    def __mul__(self, other):
        if not isinstance(other, Op):
            raise ValueError
        node = Mul()
        node.setValue(self, other)
        return node

    def __sub__(self, other):
        if not isinstance(other, Op):
            raise ValueError
        node = Minus()
        node.setValue(self, other)
        return node

    def __truediv__(self, other):
        if not isinstance(other, Op):
            raise ValueError
        node = Div()
        node.setValue(self, other)
        return node

    def getRes(self):
        raise NotImplementedError

    def Df(self, name):
        raise NotImplementedError


class Add(Op):
    def __init__(self):
        self.name = "Add"
        self.left = None
        self.right = None

    def getRes(self):
        return self.left.getRes() + self.right.getRes()

    def Df(self, name):
        return self.left.Df(name) + self.right.Df(name)

    def setValue(self, x, y):
        self.left = x
        self.right = y
        return self


class Minus(Op):
    def __init__(self):
        self.name = "minus"
        self.left = None
        self.right = None

    def getRes(self):
        return self.left.getRes() - self.right.getRes()

    def Df(self, name):
        return self.left.Df(name) - self.right.Df(name)

    def setValue(self, x, y):
        self.left = x
        self.right = y
        return self


class Mul(Op):
    def __init__(self):
        self.name = "mul"
        self.left = None
        self.right = None

    def getRes(self):
        return self.left.getRes() * self.right.getRes()

    def Df(self, name):
        return self.left.Df(name) * self.right.getRes() + self.left.getRes() * self.right.Df(name)

    def setValue(self, x, y):
        self.left = x
        self.right = y
        return self


class Div(Op):
    '''
    TODO: handle division by zero
    '''

    def __init__(self):
        self.name = "div"
        self.left = None
        self.right = None

    def getRes(self):
        return self.left.getRes() / self.right.getRes()

    def Df(self, name):
        return (self.left.Df(name) * self.right.getRes() - self.left.getRes() * self.right.Df(name)) / self.right.getRes() ** 2

    def setValue(self, x, y):
        self.left = x
        self.right = y
        return self


class Sigmoid(Op):
    def __init__(self):
        self.name = "sigmoid"
        self.x = None

    def getRes(self):
        return 1.0 / (1 + np.exp(-self.x.getRes()))

    def Df(self, name):
        r_ = self.getRes()
        return r_ * (1 - r_) * self.x.Df(name)

    def setValue(self, x):
        self.x = x
        return self


class Sin(Op):
    def __init__(self):
        self.name = "sin"
        self.x = None

    def getRes(self):
        return np.sin(self.x.getRes())

    def Df(self, name):
        return np.cos(self.x.getRes()) * self.x.Df(name)

    def setValue(self, x):
        self.x = x
        return self


class Cos(Op):
    def __init__(self):
        self.name = "cos"
        self.x = None

    def getRes(self):
        return np.cos(self.x.getRes())

    def Df(self, name):
        return -np.sin(self.x.getRes()) * self.x.Df(name)

    def setValue(self, x):
        self.x = x
        return self


class Tan(Op):
    def __init__(self):
        self.name = "tan"
        self.x = None

    def getRes(self):
        r_ = self.x.getRes()
        return np.sin(r_) / np.cos(r_)

    def Df(self, name):
        return 1 / np.cos(self.x.getRes()) ** 2 * self.x.Df(name)

    def setValue(self, x):
        self.x = x
        return self


class Exp(Op):
    def __init__(self):
        self.name = "exp"
        self.x = None

    def getRes(self):
        return np.exp(self.x.getRes())

    def Df(self, name):
        return np.exp(self.x.getRes()) * self.x.Df(name)

    def setValue(self, x):
        self.x = x
        return self


class Log(Op):
    def __init__(self):
        self.name = "log"
        self.x = None

    def getRes(self):
        return np.log(self.x.getRes())

    def Df(self, name):
        return self.x.Df(name) / self.x.getRes()

    def setValue(self, x):
        self.x = x
        return self


class Value(Op):
    def __init__(self, name, x=None):
        self.name = name
        if isinstance(x, Value):
            self.x = x.getRes()
        else:
            self.x = x

    def getRes(self):
        return self.x

    def Df(self, name):
        if name == self.name:
            return 1
        else:
            return 0

    def setValue(self, x):
        self.x = x
        return self


def exp(x):
    if not isinstance(x, Op):
        raise ValueError
    node = Exp()
    node.setValue(x)
    return node


def log(x):
    if not isinstance(x, Op):
        raise ValueError
    node = Log()
    node.setValue(x)
    return node


def sin(x):
    if not isinstance(x, Op):
        raise ValueError
    node = Sin()
    node.setValue(x)
    return node


def cos(x):
    if not isinstance(x, Op):
        raise ValueError
    node = Cos()
    node.setValue(x)
    return node


def tan(x):
    if not isinstance(x, Op):
        raise ValueError
    node = Tan()
    node.setValue(x)
    return node


def sigmoid(x):
    if not isinstance(x, Op):
        raise ValueError
    node = Sigmoid()
    node.setValue(x)
    return node


if __name__ == "__main__":
    '''
    y = (sin(x1 + 1) + cos(2x2)) * tan(log(x3)) + (sin(x2 + 1) + cos(2x1))exp(1 + sin(x3))
    x1 = x2 = x3 = 2
    '''
    x1_, x2_, x3_ = list(map(int, input(
        "please input x1, x2, x3 splitted by space: ").split()))

    def func(x1, x2, x3):
        return (np.sin(x1 + 1) + np.cos(2 * x2)) * np.tan(np.log(x3)) \
            + (np.sin(x2 + 1) + np.cos(2 * x1)) * np.exp(1 + np.sin(x3))
    dx1 = Df(lambda x: func(x, x2_, x3_), 2)
    dx2 = Df(lambda x: func(x1_, x, x3_), 2)
    dx3 = Df(lambda x: func(x1_, x2_, x), 2)
    print("numerical results: ")
    print([dx1, dx2, dx3])
    x1 = Value("x1", x1_)
    x2 = Value("x2", x2_)
    x3 = Value("x3", x3_)
    const1 = Value("_Const1", 1)
    const2 = Value("_Const2", 2)
    nodeAdd1 = Add()
    nodeAdd1.setValue(x1, const1)
    nodeMul1 = Mul()
    nodeMul1.setValue(x2, const2)
    nodeSin1 = Sin()
    nodeSin1.setValue(nodeAdd1)
    nodeCos1 = Cos()
    nodeCos1.setValue(nodeMul1)
    nodeAdd2 = Add()
    nodeAdd2.setValue(nodeSin1, nodeCos1)
    nodeLog1 = Log()
    nodeLog1.setValue(x3)
    nodeTan1 = Tan()
    nodeTan1.setValue(nodeLog1)
    nodeMul2 = Mul()
    nodeMul2.setValue(nodeAdd2, nodeTan1)

    nodeAdd3 = Add()
    nodeAdd3.setValue(x2, const1)
    nodeSin2 = Sin()
    nodeSin2.setValue(nodeAdd3)
    nodeMul3 = Mul()
    nodeMul3.setValue(x1, const2)
    nodeCos2 = Cos()
    nodeCos2.setValue(nodeMul3)

    nodeAdd4 = Add()
    nodeAdd4.setValue(nodeSin2, nodeCos2)

    nodeSin3 = Sin()
    nodeSin3.setValue(x3)
    nodeAdd5 = Add()
    nodeAdd5.setValue(nodeSin3, const1)
    nodeExp = Exp()
    nodeExp.setValue(nodeAdd5)

    nodeMul4 = Mul()
    nodeMul4.setValue(nodeAdd4, nodeExp)

    nodeAdd6 = Add()
    nodeAdd6.setValue(nodeMul2, nodeMul4)
    print("my results: ")
    print([nodeAdd6.Df("x1"), nodeAdd6.Df("x2"), nodeAdd6.Df("x3")])

    print("standard results: ")
    print([m.cos(x1_ + 1)*m.tan(m.log(x3_)) - 2 * m.sin(2*x1_)*m.exp(1 + m.sin(x3_)),
           -2 * m.sin(2 * x2_)*m.tan(m.log(x3_)) +
           m.cos(x2_ + 1)*m.exp(1 + m.sin(x3_)),
           (m.sin(x1_ + 1) + m.cos(2 * x2_)) / (x3_ * m.cos(m.log(x3_))**2) +
           (m.sin(x2_ + 1) + m.cos(2 * x1_)) * m.cos(x3_) * m.exp(1 + m.sin(x3_))])
