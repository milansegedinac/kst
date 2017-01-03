import numpy as np
from kst import ob_counter


def mini_iita(dataset, A):
    b = ob_counter(dataset)
    n, m = dataset.shape

    bs = [None] * len(A)
    for i in range(len(A)):
        bs[i] = np.zeros((m, m))

    p = [None] * m
    for i in range(m):
        p[i] = sum(dataset.ix[:, i])

    diff_value_alt = np.repeat(0.0, len(A))
    error = np.repeat(0.0, len(A))

    # computation of error rate
    for k in range(len(A)):
        x = np.repeat(0.0, 4)
        for i in range(m):
            for j in range(m):
                if ((i, j) in A[k]) and (i != j):
                    x[1] += -2 * b[i][j] * p[j]
                    x[3] += 2 * p[j]**2
                if ((i, j) not in A[k]) and ((j, i) in A[k]) and (i != j):
                    x[0] += -2 * b[i][j] + 2 * p[i] * p[j] - 2 * p[i]**2
                    x[2] += 2 * p[i]**2

        error[k] = -(x[0] + x[1]) / (x[2] + x[3])

    # computation of diff values
    all_imp = set()
    for i in range(m - 1):
        for j in range(i + 1, m):
            all_imp = all_imp.union(all_imp, set([(i, j), (j, i)]))

    for k in range(len(A)):
        if not A[k]:
            diff_value_alt[k] = None
        else:
            for i in all_imp:
                if i in A[k]:
                    bs[k][int(i[0])][int(i[1])] = error[k] * sum(dataset.ix[:, int(i[1])])
                if (i not in A[k]) and ((i[1], i[0]) not in A[k]):
                    bs[k][int(i[0])][int(i[1])] = (1.0 - float(sum(dataset.ix[:, int(i[0])])) / n) * sum(dataset.ix[:, int(i[1])])
                if (i not in A[k]) and ((i[1], i[0]) in A[k]):
                    bs[k][int(i[0])][int(i[1])] = sum(dataset.ix[:, int(i[1])]) - sum(dataset.ix[:, int(i[0])]) + sum(dataset.ix[:, int(i[0])]) * error[k]
            diff_value_alt[k] = sum(sum((b - bs[k]) ** 2)) / (m ** 2 - m)

    return {'diff.value': diff_value_alt, 'error.rate': error}
