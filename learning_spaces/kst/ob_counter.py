import numpy as np
import pandas as pd


def ob_counter(dataset):
    """
    Computation of numbers of counterexamples
    Computes from a dataset for all item pairs the corresponding numbers of counterexamples.

    :param dataset: dataframe or matrix consisted of ones and zeros
    :return: matrix of the numbers of counterexamples for all pairs of items
    """

    (n, m) = dataset.shape
    b = np.zeros((m, m), dtype=np.int32)

    data = dataset
    if isinstance(dataset, pd.DataFrame):
        data = dataset.as_matrix()

    for i in range(m):
        for j in range(m):
            if i != j:
                b[i, j] = sum(np.logical_and(data[:, i] == 0, data[:, j] == 1))

    return b
