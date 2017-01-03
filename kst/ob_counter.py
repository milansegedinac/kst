import numpy as np


def ob_counter(data):
    (n, m) = data.shape
    matrix = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            if i != j:
                matrix[i, j] = sum(np.logical_and(data.ix[:, i] == 0, data.ix[:, j] == 1))

    return matrix.astype(int)
