def print_iita(obj):
    """
    Formatted print of iita response

    :param obj: dictionary - response from iita function
    :return:
    """

    print('\n\tInductive Item Tree Analysis\n')

    algorithm = '-'
    if obj['v'] == 1:
        algorithm = 'minimized corrected'
    elif obj['v'] == 2:
        algorithm = 'corrected'
    elif obj['v'] == 3:
        algorithm = 'original'

    print('\nAlgorithm: {} IITA'.format(algorithm))
    print('\nQuasi order: {}'.format(obj['implications']))
