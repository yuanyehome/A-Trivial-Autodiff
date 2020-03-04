import numpy as np
import op
import random

# global values are just for test
# used for initialize params
init1 = [[], []]
init2 = [[], [], [], []]
for i in range(2):
    for j in range(4):
        init1[i].append(random.random())
        init2[j].append(random.random())


class Linear:
    def __init__(self, shape, net, initialWeights=None):
        self.net = net
        thisNum = len(self.net.linearNames)
        self.name = "_Linear" + str(thisNum)
        self.net.linearNames[self.name] = shape
        if len(shape) != 2:
            raise ValueError
        self.input_shape = shape[0]
        self.output_shape = shape[1]
        self.params = []
        self.net.weights[self.name] = self.params
        for i in range(shape[0]):
            self.params.append([])
            for j in range(shape[1]):
                if initialWeights == None:
                    self.params[i].append(
                        op.Value(self.name + str(i) + str(j), 0))
                else:
                    self.params[i].append(
                        op.Value(self.name + str(i) + str(j), initialWeights[i][j]))
        self.net.weights[self.name] = self.params

    def __call__(self, inputArray):
        if self.input_shape != len(inputArray):
            raise ValueError
        output = []
        if not isinstance(inputArray[0], op.Op):
            inputArray = list(map(lambda t: op.Value("const", t), inputArray))
        for j in range(self.output_shape):
            res = None
            for i in range(self.input_shape):
                if res == None:
                    res = self.params[i][j] * inputArray[i]
                else:
                    res += self.params[i][j] * inputArray[i]
            output.append(res)
        return output


def Sigmoid(inputArray):
    return list(map(op.sigmoid, inputArray))


def Softmax(inputArray):
    S = None
    length = len(inputArray)
    for i in range(length):
        if S == None:
            S = op.exp(inputArray[i])
        else:
            S += op.exp(inputArray[i])
    ans = []
    for i in range(length):
        ans.append(op.exp(inputArray[i]) / S)
    return ans


class crossEntropyLoss:
    def __init__(self, net):
        self.net = net
        pass

    def __call__(self, output, std):
        assert(len(output) == len(std))
        lengthStd = len(std)
        self.ans = None
        for i in range(lengthStd):
            v = op.Value("const", std[i])
            if self.ans == None:
                self.ans = -v * op.log(output[i])
            else:
                self.ans = self.ans - v * op.log(output[i])
        return self

    def backward(self):
        linearNames = self.net.linearNames
        self.gradients = {}
        for name in linearNames.keys():
            self.gradients[name] = []
            shape = linearNames[name]
            for i in range(shape[0]):
                self.gradients[name].append([])
                for j in range(shape[1]):
                    nameStr = name + str(i) + str(j)
                    self.gradients[name][i].append(self.ans.Df(nameStr))

    def printGradients(self):
        for name in self.gradients.keys():
            print("Gradient of Layer " + name + ": ")
            shape = self.net.linearNames[name]
            for i in range(shape[0]):
                print(
                    list(map(lambda t: round(t, 4), self.gradients[name][i])))

    def getLoss(self):
        return self.ans.getRes()

    def sum(self, weights=None):
        s = 0
        for name in self.gradients.keys():
            shape = self.net.linearNames[name]
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if weights != None:
                        s += self.gradients[name][i][j] * weights[name][i][j]
                    else:
                        s += self.gradients[name][i][j]
        return s


def listAdd(l1, l2):
    assert(len(l1) == len(l2))
    length = len(l1)
    return list(map(lambda t: t[0] + t[1], zip(l1, l2)))


class baseNN:
    def __init__(self):
        self.linearNames = {}
        self.weights = {}

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class NN(baseNN):
    def __init__(self):
        global init1, init2
        super(NN, self).__init__()
        self.fc1 = Linear([2, 4], self, init1)
        self.fc2 = Linear([4, 2], self, init2)

    def forward(self, x):
        init_x = list(map(lambda t: op.Value("const", t), x))
        x = Sigmoid(self.fc1(x))
        x = listAdd(self.fc2(x), init_x)
        # x = self.fc2(x)
        return x


if __name__ == "__main__":
    t = 1e-8
    net1 = NN()
    for i in range(2):
        for j in range(4):
            init1[i][j] += t
            init2[j][i] += t
    net2 = NN()
    inputs = [2, 3]
    outputs1 = Softmax(net1(inputs))
    outputs2 = Softmax(net2(inputs))
    lossFun1 = crossEntropyLoss(net1)
    loss1 = lossFun1(outputs1, [1, 0])
    lossFun2 = crossEntropyLoss(net2)
    loss2 = lossFun2(outputs2, [1, 0])
    loss1.backward()
    loss1.printGradients()
    deltaLoss = loss2.getLoss() - loss1.getLoss()
    print("numerical test: ")
    print("LHS: " + str(round(deltaLoss / t, 6)))
    print("RHS: " + str(round(loss1.sum(), 6)))
