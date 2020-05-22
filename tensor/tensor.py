# coding:utf-8
import numpy as np



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

    def __str__(self):
        return '%s' % self.vector

    def __mul__(self, scalar):
        vector = []
        for v in self.vector:
            vector.append(v * scalar)

        return DeviationVariable(vector)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __matmul__(self, mat):
        vector = []
        for v in self.vector:
            vector.append(v * mat)

        return DeviationVariable(vector)

    def __rmatmul__(self, mat):
        return self.__matmul__(mat)

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
            return Operator(self.a * self.b, self.a.derivative(), '*')

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
            a = run(self.a)
            b = run(self.b)
            if isinstance(b, DeviationVariable):
                return b * a
            return a * b


class Sum:
    def __init__(self, array):
        self.array = array

    def derivative(self):
        return Sum(self.array.derivative())

    def run(self):
        array = self.array.run()
        if isinstance(array, DeviationVariable):
            vector = []
            for v in array.vector:
                vector.append(np.sum(v))

            return DeviationVariable(vector)
        return np.sum(array)


class Training:
    def __init__(self, e):
        self.express = e
        self.derivative = e.derivative()
        self.step = 1

    def run(self):
        derivative = self.derivative.run()

        old = self.express.run()
        for (v, d) in zip(Variable.list, derivative.vector):
            v.value -= self.step * d
        loss = self.express.run()

        while loss > old:
            self.step /= 2
            for (v, d) in zip(Variable.list, derivative.vector):
                v.value += self.step * d

            loss = self.express.run()

        return loss


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


