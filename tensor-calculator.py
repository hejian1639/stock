# coding:utf-8
import numpy as np

length = 100
data_x = np.random.rand(length).astype(np.float32)
deviation = np.random.rand(length).astype(np.float32)
data_y = data_x * 0.1 + 0.3 + (deviation - 0.5) * 0.01


class Variable:
    count = 0
    list = []

    def __init__(self, v):
        self.value = v
        self.index = Variable.count
        Variable.list.append(self)
        Variable.count += 1

    def __add__(self, other):
        return Operator(self, other, '+')

    def __sub__(self, other):
        return Operator(self, other, '-')

    def __pow__(self, p):
        pass

    def __mul__(self, scalar):
        return Operator(self, scalar, '*')

    def __rmul__(self, scalar):
        return Operator(self, scalar, '*')

    def derivative(self):
        matrix = []
        for i in range(Variable.count):
            matrix.append(0)

        matrix[self.index] = 1
        return DeviationVariable(matrix)

    def run(self):
        return self.value


class DeviationVariable:
    def __init__(self, v):
        self.vector = v

    def __mul__(self, scalar):
        vector = []
        for v in self.vector:
            vector.append(v * scalar)

        return DeviationVariable(vector)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __add__(self, other):
        vector = []
        if isinstance(other, DeviationVariable):
            for (v, o) in zip(self.vector, other.vector):
                vector.append(v + o)
        else:
            for v in self.vector:
                vector.append(v + other)

        return DeviationVariable(vector)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        vector = []
        if isinstance(other, DeviationVariable):
            for (v, o) in zip(self.vector, other.vector):
                vector.append(v - o)
        else:
            for v in self.vector:
                vector.append(v - other)

        return DeviationVariable(vector)

    def __rsub__(self, other):
        vector = []
        if isinstance(other, DeviationVariable):
            for (v, o) in zip(self.vector, other.vector):
                vector.append(o - v)
        else:
            for v in self.vector:
                vector.append(other - v)

        return DeviationVariable(vector)

    def sum(self):
        vector = []
        for v in self.vector:
            if isinstance(v, DeviationVariable):
                vector.append(v.sum())
            else:
                vector.append(np.sum(v))

        return DeviationVariable(vector)


class Input:

    def setValue(self, v):
        self.value = v

    def __add__(self, other):
        return Operator(self, other, '+')

    def __sub__(self, other):
        return Operator(self, other, '-')

    def __pow__(self, p):
        return Operator(self, p, '**')

    def __mul__(self, scalar):
        return Operator(self, scalar, '*')

    def derivative(self):
        return 0

    def run(self):
        return self.value


class Operator:
    def __init__(self, a, b, operator):
        self.a = a
        self.b = b
        self.operator = operator

    def __add__(self, other):
        return Operator(self, other, '+')

    def __sub__(self, other):
        return Operator(self, other, '-')

    def __pow__(self, p):
        return Operator(self, p, '**')

    def __mul__(self, scalar):
        return Operator(self, scalar, '*')

    def __rmul__(self, scalar):
        return Operator(self, scalar, '*')

    def derivative(self):

        if self.operator == '**':
            return Operator(self.b, self.a.derivative(), '*')

        if self.operator == '-':
            return Operator(self.a.derivative(), self.b.derivative(), '-')

        if self.operator == '+':
            return Operator(self.a.derivative(), self.b.derivative(), '+')

        if self.operator == '*':
            return Operator(Operator(self.a, self.b.derivative(), '*'), Operator(self.a.derivative(), self.b, '*'), '+')

    def run(self):
        if self.operator == '**':
            return run(self.a) ** self.b

        if self.operator == '-':
            return run(self.a) - run(self.b)

        if self.operator == '+':
            return run(self.a) + run(self.b)

        if self.operator == '*':
            return run(self.a) * run(self.b)


class Sum:
    def __init__(self, array):
        self.array = array

    def derivative(self):
        return Sum(self.array.derivative())

    def run(self):
        array = self.array.run()
        if isinstance(array, DeviationVariable):
            return array.sum()
        return np.sum(array)

    def gradientDescent(self):
        self.loss = 0
        for i in self.array:
            self.loss += i.gradientDescent()
        return self.loss


class Training:
    def __init__(self, e):
        self.express = e
        self.derivative = e.derivative()
        self.step = 0.0001

    def run(self):
        derivative = self.derivative.run()

        for (v, d) in zip(Variable.list, derivative.vector):
            v.value -= self.step * d

        return self.express.run()


def sum(array):
    return Sum(array)


def minimize(loss):
    return Training(loss)


def run(train, feed_dict=None):
    if feed_dict:
        for feed, feed_val in feed_dict.items():
            feed.setValue(feed_val)

    if type(train) in (Variable, Input, Operator, Sum, Training):
        return train.run()
    return train


w = Variable(0)
b = Variable(0)
x = Input()
y = Input()
t_y = w * x + b

loss = sum((y - t_y) ** 2)

print(run(t_y, {x: data_x}))
print(run(loss, {x: data_x, y: data_y}))

train = minimize(loss)
# res = run(train, {x: data_x, y: data_y})
# print(res)

for step in range(20):
    print('loss = ', run(train, {x: data_x, y: data_y}))
    print('w = ', run(w), ' b = ', run(b))
