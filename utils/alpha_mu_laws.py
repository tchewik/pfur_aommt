import numpy as np


class MuLaw:
    def __init__(self):
        self.mu = 255.

    def encode(self, array):
        out = np.sign(array) * np.log(1. + self.mu * np.abs(array)) / np.log(1. + self.mu)
        out = out.astype('float64')
        return out

    def decode(self, array):
        array = np.array(array)
        out = np.sign(array) / self.mu * ((1. + self.mu) ** np.abs(array) - 1.)
        out = np.where(np.equal(array, 0), array, out)
        return tuple(int(value) for value in np.around(out, decimals=1))  # to avoid 127.999... pruning to 127


class AlphaLaw:
    def __init__(self):
        self.A = 87.6

    def encode(self, array):
        def f(x):
            if np.abs(x) < 1./self.A:
                return np.sign(x) * self.A * np.abs(x) / (1. + np.log(self.A))
            else:
                return (1. + np.log(self.A * np.abs(x))) / (1. + np.log(self.A))

        y = np.sign(array) * list(map(f, array))
        return y.astype('float64')

    def decode(self, array):
        def f_(x):
            if np.abs(x) < 1./(1. + np.log(self.A)):
                return np.abs(x) * (1. + np.log(self.A)) / self.A
            else:
                return np.exp(np.abs(x) * (1. + np.log(self.A)) - 1) / self.A

        x = np.sign(array) * list(map(f_, array))
        return tuple(int(value) for value in np.around(x, decimals=1))
