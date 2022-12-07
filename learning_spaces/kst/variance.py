import sys
from itertools import product

import numpy as np
import pandas as pd

from learning_spaces.kst import pattern


def variance(dataset, imp, v):
    """
    Variance
    computes estimated asymptotic variances of the
    maximum likelihood estimators diff from data,
    assuming a multinomial probability distribution on the set
    of all response patterns.

    :param dataset: dataframe or matrix consisted of ones and zeros
    :param imp: list of implications in tuple form
    :param v: algorithm: v=1 (minimized corrected), v=2 (corrected)
    :return: variance
    """

    if len(imp) == 0:
        sys.exit('Number of implications must be greater than zero.')
    if v != 1 and v !=2:
        sys.exit('IITA version must be specified.')

    data = dataset
    if isinstance(dataset, pd.DataFrame):
        data = dataset.values

    n_samples, n_items = data.shape

    # Number of times a pattern occurs
    pat = np.array(list(product([0, 1], repeat=n_items)), dtype='int8')
    patterns = pattern(dataset, p=pat)['states']

    # Relative frequencies - sum dataset[:,i]/n_samples
    rho_sum = np.zeros(n_items)
    for i in range(n_items):
        rho_sum[i] = np.sum(patterns[np.where(patterns[:, i] == 1), n_items]) / n_samples

    rho_sum_counter = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue

            rho_sum_counter[i, j] = np.sum(patterns[np.where(np.logical_and(patterns[:, i] == 0, patterns[:, j] == 1)), n_items]) / n_samples

    # Expected fisher information
    exp_fish = np.zeros((2**n_items-1, 2**n_items-1))
    for i in range(1, 2**n_items):
        for j in range(1, 2**n_items):
            if i == j:
                exp_fish[i-1, j-1] = (patterns[i, n_items] / n_samples) * (1 - patterns[i, n_items] / n_samples)
            else:
                exp_fish[i-1, j-1] = -1 * (patterns[i, n_items] / n_samples) * (patterns[j, n_items] / n_samples)

    # Error
    error = 0

    # Original and corrected
    if v == 2:
        for i in imp:
            error += rho_sum_counter[i] * n_items / np.sum(dataset[:, i[1]])
        error /= len(imp)

    # Minimized corrected
    if v == 1:
        x = np.zeros(4)
        for i in range(n_items):
            for j in range(n_items):
                if i == j:
                    continue

                if (i, j) in imp:
                    x[1] += -2 * rho_sum_counter[i, j] * rho_sum[j]
                    x[3] += 2 * rho_sum[j] ** 2
                elif (j, i) in imp:
                    x[0] += -2 * rho_sum_counter[i, j] * rho_sum[i] + 2 * rho_sum[i] * rho_sum[j] - 2 * rho_sum[i] ** 2
                    x[2] += 2 * rho_sum[i] ** 2
        error = - (x[0] + x[1]) / (x[2] + x[3])

    # Gamma derivative

    gamma_deriv = np.zeros(2 ** n_items - 1)

    # Original and corrected
    if v == 2:
        for i in range(1, 2 ** n_items):
            for imp_x, imp_y in imp:
                if patterns[i,  imp_y] == 1:
                    if patterns[i, imp_x] == 0:
                        gamma_deriv[i-1] += (rho_sum[imp_y] - rho_sum_counter[imp_x, imp_y]) / (rho_sum[imp_y] ** 2)
                    else:
                        gamma_deriv[i-1] += rho_sum_counter[imp_x, imp_y] / (rho_sum[imp_y] ** 2)

        gamma_deriv /= len(imp)

    # Minimized corrected
    if v == 1:
        x = np.zeros(4)
        for k in range(n_items):
            for h in range(n_items):
                if k == h:
                    continue

                if (k, h) in imp:
                    x[1] += -2 * rho_sum_counter[k, h] * rho_sum[h]
                    x[3] += 2 * rho_sum[h] ** 2
                elif (h, k) in imp:
                    x[0] += -2 * rho_sum_counter[k, h] * rho_sum[k] + 2 * rho_sum[k] * rho_sum[h] - 2 * rho_sum[k] ** 2
                    x[2] += 2 * rho_sum[k] ** 2

        tmp1, tmp2 = 0, 0
        for i in range(1, 2 ** n_items):
            for k in range(n_items):
                for h in range(n_items):
                    if k == h:
                        continue

                    if (k, h) not in imp and (h, k) in imp:
                        if patterns[i, k] == 0 and patterns[i, h] == 1:
                            tmp1 += -2 * rho_sum[k]
                        if patterns[i, k] == 1:
                            tmp1 += -2 * rho_sum_counter[k, h] + 2 * rho_sum[h] - 4 * rho_sum[k]
                            tmp2 += 4 * rho_sum[k]
                        if patterns[i, h] == 1:
                            tmp1 += 2 * rho_sum[k]

                    if (h, k) in imp:
                        if patterns[i, k] == 0 and patterns[i, h] == 1:
                            tmp1 += -2 * rho_sum[h]
                        if patterns[i, h] == 1:
                            tmp1 += -2 * rho_sum_counter[k, h]
                            tmp2 += 4 * rho_sum[h]

            gamma_deriv[i-1] = - (tmp1 * (x[2] + x[3]) - tmp2 * (x[0] + x[1])) / ((x[2] + x[3]) ** 2)

    # Gradient of diff for corrected and minimized corrected
    grad = np.zeros(2 ** n_items - 1)

    for i in range(1, 2 ** n_items):
        for k in range(n_items):
            for h in range(n_items):
                if k == h:
                    continue

                if (k, h) in imp:
                    if patterns[i,k] == 0 and patterns[i, h] == 1:
                        grad[i-1] += 2 * (
                                    rho_sum_counter[k, h] - rho_sum[h] * error) * (
                                    1 - error - rho_sum[h] * gamma_deriv[i-1])
                    else:
                        if patterns[i, h] == 1:
                            grad[i-1] += 2 * (
                                        rho_sum_counter[k, h] - rho_sum[h] * error) * (
                                        -error - rho_sum[h] * gamma_deriv[i-1])
                        else:
                            grad[i-1] += 2 * (
                                        rho_sum_counter[k, h] - rho_sum[h] * error) * (
                                        -rho_sum[h] * gamma_deriv[i-1])

                elif (h, k) in imp:
                    if patterns[i, k] == 1:
                        if patterns[i, h] == 1:
                            grad[i-1] += 2 * (
                                        rho_sum_counter[k, h] - rho_sum[h] + rho_sum[k] - rho_sum[k] * error) * (
                                        - error - rho_sum[k] * gamma_deriv[i-1])
                        else:
                            grad[i-1] += 2 * (
                                        rho_sum_counter[k, h] - rho_sum[h] + rho_sum[k] - rho_sum[k] * error) * (
                                        1 - error - rho_sum[k] * gamma_deriv[i-1])
                    else:
                        if patterns[i, k] == 1:
                            grad[i-1] += 2 * (
                                        rho_sum_counter[k, h] - rho_sum[h] + rho_sum[k] - rho_sum[k] * error) * (
                                        -rho_sum[k] * gamma_deriv[i-1])
                        else:
                            grad[i-1] += 2 * (
                                        rho_sum_counter[k, h] - rho_sum[h] + rho_sum[k] - rho_sum[k] * error) * (
                                        -rho_sum[k] * gamma_deriv[i-1])

                else:
                    if patterns[i, k] == 0 and patterns[i, h] == 1:
                        grad[i-1] += 2 * (rho_sum_counter[k, h] - (1 - rho_sum[k]) * rho_sum[h]) * rho_sum[k]
                    else:
                        if patterns[i, h] == 1:
                            if patterns[i, k] == 1:
                                grad[i-1] += 2 * (
                                            rho_sum_counter[k, h] - (1 - rho_sum[k]) * rho_sum[h]) * (
                                            rho_sum[h] - 1 + rho_sum[k])
                            else:
                                grad[i-1] += 2 * (rho_sum_counter[k, h] - (1 - rho_sum[k]) * rho_sum[h]) * rho_sum[h]

    grad /= (n_items * (n_items - 1))

    # Final computation
    variance = grad @ exp_fish @ grad
    return variance
