import numpy as np


class HaarWavelet:
    def __init__(self):
        self.matrix = np.array([
            [0.5**0.5, 0.5**0.5],
            [-0.5**0.5, 0.5**0.5]])

    def encode_pair(self, x, y):
        return np.around(np.dot(self.matrix, np.array([x, y])), 1).tolist()

    def decode_pair(self, x, y):
        x, y = float(x), float(y)
        return np.around(np.dot(self.matrix.transpose(), np.array([x, y])), 1).tolist()  #np.dot((self.matrix.transpose()), np.array([x, y])).round(0).astype('uint8').tolist()

    def apply_to_raw(self, function, raw):
        result = []
        for i in range(0, len(raw)-1, 2):
            result += function(raw[i], raw[i+1])
        return result

    def encode(self, raw):
        result = self.apply_to_raw(self.encode_pair, raw)
        return [result[i] for i in range(0, len(result), 2)] + [result[i] for i in range(1, len(result), 2)]

    def decode(self, raw):
        reshaped_raw = []
        middle = len(raw) // 2
        for i in range(middle):
            reshaped_raw.append(raw[i])
            reshaped_raw.append(raw[middle + i])
        return self.apply_to_raw(self.decode_pair, reshaped_raw)
