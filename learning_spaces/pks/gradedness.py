from .conversion import convert_as_pattern


def is_forward_graded(data):
    """
    Checks if a knowledge structure is forward-graded in any item
    :param data: dataframe with binary matrix representing the knowledge structure
    :return: logical dict of items
    """
    ret_val = {}
    data_pattern = convert_as_pattern(data)
    header = list(data)
    for item in header:
        new_data = data.copy(deep=True)
        new_data[item] = 1
        new_data_pattern = convert_as_pattern(new_data)
        graded = []
        for pattern in new_data_pattern:
            graded.append(pattern in data_pattern)
        ret_val[item] = all(graded)
    return ret_val


def is_backward_graded(data):
    """
    Checks if a knowledge structure is backward-graded in any item
    :param data: dataframe with binary matrix representing the knowledge structure
    :return: logical dict of items
    """
    ret_val = {}
    data_pattern = convert_as_pattern(data)
    header = list(data)
    for item in header:
        new_data = data.copy(deep=True)
        new_data[item] = 0
        new_data_pattern = convert_as_pattern(new_data)
        graded = []
        for pattern in new_data_pattern:
            graded.append(pattern in data_pattern)
        ret_val[item] = all(graded)
    return ret_val
