import sys
from collections import Counter

import numpy as np
import pandas as pd


def pattern(dataset, n=5, p=None):
    """
    pattern
    computes the absolute frequencies of the response patterns,
    and optionally, the absolute frequencies of a collection of
    specified knowledge states in a dataset.

    :param dataset: dataframe or matrix consisted of ones and zeros
    :param n: number of patterns (must be greater than zero)
    :param p: dataframe or matrix 
    :return: dictionary representing pattern data
    """

    if n < 1:
        sys.exit('Number of patterns must be greater than zero.')

    data = dataset
    if isinstance(dataset, pd.DataFrame):
        data = dataset.values

    def ks_to_str(ks): return ''.join((str(is_correct_answer) for is_correct_answer in ks))

    pattern = Counter(np.apply_along_axis(ks_to_str, axis=1, arr=data))
    if n > len(pattern):
        n = len(pattern)

    if p is None:
        return {'response.patterns': pattern.most_common(n), 'states': p, 'n': n}

    def getKnowledgeStatesFrequencies(p):
        return np.apply_along_axis(lambda row: pattern[ks_to_str(row)], axis=1, arr=p)

    if isinstance(p, pd.DataFrame):
        states = p.assign(size=getKnowledgeStatesFrequencies(p.values))
    else:
        frequencies = getKnowledgeStatesFrequencies(p)
        states = np.hstack((p, np.reshape(frequencies, (-1, 1))))

    return {'response.patterns': pattern.most_common(n), 'states': states, 'n': n}
