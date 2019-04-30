import numpy as np
from functools import partial
from multiprocessing import Pool


def _idft_calc(u, data):
    return sum([data[x] * (np.cos(2 * np.pi * u * x / len(data)) + np.sin(2 * np.pi * u * x / len(data)) * 1j) for x in range(len(data))])


def _hanna_func(n, data):
    return .5 * (1 - np.cos(2. * np.pi * data[n] / len(data)))


class Fourier:
    @staticmethod
    def dft(data):
        """ discrete fourier transform """

        data = np.array(data).astype(float)
        N = len(data)
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, data)

    @staticmethod
    def idft(data):
        """ inversed discrete fourier transform """

        with Pool() as pool:
            result = pool.map(partial(_idft_calc, data=data), range(len(data)))

        return result

    @staticmethod
    def _hanna_window(data):
        """ applies hanna window function """

        with Pool() as pool:
            result = pool.map(partial(_hanna_func, data=data), range(len(data)))

        return result
