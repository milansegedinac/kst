import pandas as pd
import string


def convert_as_pattern(data, freq=False, as_letters=False):
    """
    Convert binary matrix of response patterns or knowledge spaces into pattern representation
    :param data: dataframe with binary matrix
    :param freq: displaying frequencies of response patterns
    :param as_letters: return response patterns as combination of header letters
    :return: list of patterns or list of patterns with list of frequencies of patterns
    """
    ret_val = []
    for row in data.itertuples():
        pattern = ""
        for i in range(1, len(row)):
            if as_letters:
                if row[i] == 1:
                    pattern += list(data)[i-1]
            else:
                pattern += str(row[i])
        if pattern == "":
            ret_val.append(str(0))
        else:
            ret_val.append(pattern)

    if freq:
        ret_pat = []
        counts = []
        for pattern in ret_val:
            if pattern not in ret_pat:
                ret_pat.append(pattern)
                counts.append(ret_val.count(pattern))
        return ret_pat, counts
    else:
        return ret_val


def convert_as_bin_mat(data, col_names=None):
    """
    Convert pattern representation of response patterns or knowledge spaces into binary matrix
    :param data: list of response patterns
    :param col_names: list of names of matrix columns
    :return: dataframe with binary matrix
    """
    header = []
    if col_names is None:
        num_of_letters = 0
        for pattern in data:
            if len(pattern) > num_of_letters:
                num_of_letters = len(pattern)
        header = list(string.ascii_lowercase[:num_of_letters])
    else:
        header = col_names

    values = []
    for pattern in data:
        value = []
        if "0" in pattern or "1" in pattern:
            if pattern == "0":  # empty set
                value = [int(0)] * len(header)
            else:
                for p in pattern:
                    value.append(int(p))
        else:  # pattern is combination of header letters
            for h in header:
                if h in pattern:
                    value.append(int(1))
                else:
                    value.append(int(0))
        values.append(value)

    return pd.DataFrame(values, columns=header)
