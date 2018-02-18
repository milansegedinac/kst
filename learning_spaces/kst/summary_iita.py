import numpy as np


def summary_iita(obj):

    print('\n\tInductive Item Tree Analysis\n')

    algorithm = '-'
    if obj['v'] == 1:
        algorithm = 'minimized corrected'
    elif obj['v'] == 2:
        algorithm = 'corrected'
    elif obj['v'] == 3:
        algorithm = 'original'

    print('\nAlgorithm: {} IITA'.format(algorithm))
    print("error rate: ")
    print("diff values: {}".format(round(obj['diff'], 3)))
    print('\nQuasi order: {}'.format(obj['implications']))
    print("index in the selection set: ")
    print(str(obj['selection.set.index']))