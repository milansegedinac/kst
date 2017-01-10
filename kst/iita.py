import sys
import pandas as pd
import numpy as np
from kst import ind_gen
from kst import ob_counter
from kst import orig_iita
from kst import mini_iita
from kst import corr_iita


def iita(dataset, v):
    if (not isinstance(dataset, pd.DataFrame) and not isinstance(dataset, np.ndarray)) or (dataset.shape[1] == 1):
        sys.exit('data must be either a numeric matrix or a data.frame, with at least two columns.')

    if sum(np.logical_not(np.logical_or(dataset == 0, dataset == 1)).sum(skipna=False)) != 0:
        sys.exit('data must contain only 0 and 1')

    if v not in (1, 2, 3):
        sys.exit('IITA version must be specified')

    # inductively generated set of competing quasi orders
    i = ind_gen(ob_counter(dataset))

    # call chosen algorithm
    if v == 1:
        ii = mini_iita(dataset, i)
    elif v == 2:
        ii = corr_iita(dataset, i)
    elif v == 3:
        ii = orig_iita(dataset, i)

    index = list(ii['diff.value']).index(min(ii['diff.value']))
    return {'diff': ii['diff.value'], 'implications': i[index], 'error.rate': ii['error.rate'][index], 'selection.set.index': index, 'v': v}
