import numpy as np


def ob_counter(data):
    (n, m) = data.shape
    b = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            if i != j:
                b[i, j] = sum(np.logical_and(data.ix[:, i] == 0, data.ix[:, j] == 1))

    return b.astype(int)
