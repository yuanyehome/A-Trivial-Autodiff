# test homework function
import op
import numpy as np
import math as m


def npFunc(x1, x2, x3):
    return (np.sin(x1 + 1) + np.cos(2 * x2)) * np.tan(np.log(x3)) \
        + (np.sin(x2 + 1) + np.cos(2 * x1)) * np.exp(1 + np.sin(x3))


x1_ = x2_ = x3_ = 2.5

# define a function
x1 = op.Value("x1", 2.5)
x2 = op.Value("x2", 2.5)
x3 = op.Value("x3", 2.5)
const1 = op.Value("const1", 1)
const2 = op.Value("const2", 2)
func = (op.sin(x1 + const1) + op.cos(const2 * x2)) * op.tan(op.log(x3)) \
    + (op.sin(x2 + const1) + op.cos(const2 * x1)) * op.exp(const1 + op.sin(x3))

print("test getRes(): ")
print(func.getRes())
print(npFunc(x1_, x2_, x3_))

print("test Df(): ")
print([func.Df("x1"), func.Df("x2"), func.Df("x3")])
print([m.cos(x1_ + 1)*m.tan(m.log(x3_)) - 2 * m.sin(2*x1_)*m.exp(1 + m.sin(x3_)),
       -2 * m.sin(2 * x2_)*m.tan(m.log(x3_)) +
       m.cos(x2_ + 1)*m.exp(1 + m.sin(x3_)),
       (m.sin(x1_ + 1) + m.cos(2 * x2_)) / (x3_ * m.cos(m.log(x3_))**2) +
       (m.sin(x2_ + 1) + m.cos(2 * x1_)) * m.cos(x3_) * m.exp(1 + m.sin(x3_))])
