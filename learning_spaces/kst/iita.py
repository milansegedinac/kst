import sys
import pandas as pd
import numpy as np
from learning_spaces.kst import ind_gen
from learning_spaces.kst import ob_counter
from learning_spaces.kst import orig_iita
from learning_spaces.kst import mini_iita
from learning_spaces.kst import corr_iita


def iita(dataset, v):
    """
    Inductive Item Tree Analysis
    Performs one of the three inductive item tree analysis algorithms (minimized corrected, corrected and original).

    :param dataset: dataframe or matrix consisted of ones and zeros
    :param v: algorithm: v=1 (minimized corrected), v=2 (corrected) and v=3 (original)
    :return: dictionary
    """

    if (not isinstance(dataset, pd.DataFrame) and not isinstance(dataset, np.ndarray)) or (dataset.shape[1] == 1):
        sys.exit('data must be either a numeric matrix or a dataframe, with at least two columns.')

    data = dataset
    if isinstance(dataset, pd.DataFrame):
        data = dataset.values

    if np.logical_not(np.logical_or(data == 0, data == 1)).sum() != 0:
        sys.exit('data must contain only 0 and 1')

    if v not in (1, 2, 3):
        sys.exit('IITA version must be specified')

    # inductively generated set of competing quasi orders
    i = ind_gen(ob_counter(data))

    # call chosen algorithm
    if v == 1:
        ii = mini_iita(data, i)
    elif v == 2:
        ii = corr_iita(data, i)
    elif v == 3:
        ii = orig_iita(data, i)

    index = list(ii['diff.value']).index(min(ii['diff.value']))
    return {'diff': ii['diff.value'], 'implications': i[index], 'error.rate': ii['error.rate'][index], 'selection.set.index': index, 'v': v}


def iita_exclude_transitive(dataset, v):
    """
    Inductive Item Tree Analysis
    Performs one of the three inductive item tree analysis algorithms (minimized corrected, corrected and original)
    and then performs transitive reduction (removes transitive edges).
    Implications array will have the same vertices and as few edges as possible.

    :param dataset: dataframe or matrix consisted of ones and zeros
    :param v: algorithm: v=1 (minimized corrected), v=2 (corrected) and v=3 (original)
    :return: dictionary
    """
    response = iita(dataset, v)
    impl = response['implications']

    # reflexive reduction
    # edges is a list of implication without reflexive edges
    edges = []
    for x, y in impl:
        if (y, x) not in edges:
            edges.append((x, y))


    # nodes is a list of all nodes extracted from edges
    nodes = list(set([node for pair in edges for node in pair]))

    # transitive reduction
    # remove transitive edges from the list of edges
    for x in nodes:
        for y in nodes:
            for z in nodes:
                if (x, y) in edges and (y, z) in edges:
                    try:
                        edges.remove((x, z))
                    except:
                        pass
    
    # update a list of implications after transitive reduction
    response['implications'] = edges
    return response