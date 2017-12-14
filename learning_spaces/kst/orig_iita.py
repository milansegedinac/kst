import sys
import numpy as np
import pandas as pd
from kst import ob_counter


def orig_iita(dataset, A):
    """
    Original Inductive Item Tree Analysis
    Performs the original inductive item tree analysis procedure and returns the corresponding diff values.

    :param dataset: dataframe or matrix consisted of ones and zeros
    :param A: list of competing quasi orders
    :return: dictionary
    """

    data = dataset
    if isinstance(dataset, pd.DataFrame):
        data = dataset.as_matrix()

    b = ob_counter(data)
    if sum(b.sum(axis=0) == 0):
        sys.exit('Each item must be solved at least once')

    n, m = data.shape

    bs = []
    for i in range(len(A)):
        bs.insert(i, np.zeros(b.shape))

    diff_value_alt = np.repeat(0.0, len(A))
    error = np.repeat(0.0, len(A))

    # computation of error rate
    for k in range(len(A)):
        for i in A[k]:
            error[k] += (b[i[0]][i[1]] / float(data[:, i[1]].sum()))
        if not A[k]:
            error[k] = None
        else:
            error[k] /= len(A[k])

    # computation of diff values
    all_imp = set()
    for i in range(m-1):
        for j in range(i+1, m):
            all_imp = all_imp.union(all_imp, {(i, j), (j, i)})

    for k in range(len(A)):
        if not A[k]:
            diff_value_alt[k] = None
        else:
            for i in all_imp:
                if i in A[k]:
                    bs[k][i[0]][i[1]] = error[k] * data[:, i[1]].sum()
                else:
                    bs[k][i[0]][i[1]] = (1.0 - data[:, i[0]].sum() / float(n)) * data[:, i[1]].sum() * (1.0 - error[k])
            diff_value_alt[k] = ((b - bs[k]) ** 2).sum() / (m ** 2 - m)

    return {'diff.value': diff_value_alt, 'error.rate': error}
