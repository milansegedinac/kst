import numpy as np
import pandas as pd
from kst import ob_counter


def mini_iita(dataset, A):
    """
    Minimized Corrected Inductive Item Tree Analysis
    Performs the minimized corrected inductive item tree analysis procedure and returns the corresponding diff values.

    :param dataset: dataframe or matrix consisted of ones and zeros
    :param A: list of competing quasi orders
    :return: dictionary
    """

    data = dataset
    if isinstance(dataset, pd.DataFrame):
        data = dataset.as_matrix()

    b = ob_counter(data)
    n, m = data.shape

    bs_num = []
    for i in range(len(A)):
        bs_num.insert(i, np.zeros((m, m)))

    p = []
    for i in range(m):
        p.insert(i, data[:, i].sum())

    diff_value_alt = np.repeat(0.0, len(A))
    error = np.repeat(0.0, len(A))

    # computation of error rate
    for k in range(len(A)):
        x = np.repeat(0.0, 4)
        for i in range(m):
            for j in range(m):
                if (i != j) and ((i, j) in A[k]):
                    x[1] += -2 * b[i, j] * p[j]
                    x[3] += 2 * p[j] ** 2
                if (i != j) and ((i, j) not in A[k]) and ((j, i) in A[k]):
                    x[0] += -2 * b[i, j] * p[i] + 2 * p[i] * p[j] - 2 * p[i] ** 2
                    x[2] += 2 * p[i] ** 2

        error[k] = -(x[0] + x[1]) / (x[2] + x[3])

    # computation of diff values
    all_imp = set()
    for i in range(m - 1):
        for j in range(i + 1, m):
            all_imp = all_imp.union(all_imp, {(i, j), (j, i)})

    for k in range(len(A)):
        if not A[k]:
            diff_value_alt[k] = None
        else:
            for i in all_imp:
                if i in A[k]:
                    bs_num[k][i[0]][i[1]] = error[k] * data[:, i[1]].sum()
                if (i not in A[k]) and ((i[1], i[0]) not in A[k]):
                    bs_num[k][i[0]][i[1]] = (1.0 - data[:, i[0]].sum() / float(n)) * data[:, i[1]].sum()
                if (i not in A[k]) and ((i[1], i[0]) in A[k]):
                    bs_num[k][i[0]][i[1]] = data[:, i[1]].sum() - data[:, i[0]].sum() + data[:, i[0]].sum() * error[k]
            diff_value_alt[k] = ((b - bs_num[k]) ** 2).sum() / (m ** 2 - m)

    return {'diff.value': diff_value_alt, 'error.rate': error}
