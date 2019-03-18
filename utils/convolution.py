import numpy as np


class Convolution:
    def __init__(self, kernel):
        self.kernel = kernel

        if max([max(line) for line in self.kernel]) > 1. or min([min(line) for line in self.kernel]) < 1.:
            self.normalize_kernel()

    def __call__(self, image):
        def get_pixel(image, i, j):
            res = 0
            for m in range(len(self.kernel)):
                for n in range(len(self.kernel[0])):
                    res += self.kernel[m][n] * image[i - m][j - n]
            return int(res)

        shift = np.min(self.kernel) < 0
        res = []
        for i in range(image.shape[0]):
            tmp = []
            for j in range(image.shape[1]):
                tmp.append(get_pixel(image, i, j))
            res.append(np.array(tmp))

        res = np.array(res)
        if shift:
            res = ((res - res.min()) * (1 / (res.max() - res.min()) * 255)).astype(
                'uint8')  # in case of negative values in kernel
            # res += 128  (another option, loses colors)

        return res

    def normalize_kernel(self):
        s = np.sum(np.abs(self.kernel))
        self.kernel = [[value / s for value in line] for line in self.kernel]
