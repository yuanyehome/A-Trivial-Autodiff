import numpy as np
import math


def Df(func, x0):
    delta = 1e-8
    return (func(x0 + delta) - func(x0)) / delta


if __name__ == "__main__":
    def func(x): return x ** 4
    print(Df(func, 2))
