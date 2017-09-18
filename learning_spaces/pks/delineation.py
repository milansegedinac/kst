import itertools
from .conversion import convert_as_bin_mat


def delineate(skill_fun, item_id=0):
    """
    Computes knowledge structure delineated by a skill function
    :param skill_fun: dataframe representing the skill function. Consists of an item indicator and a
    problem-by-skill indicator binary matrix
    :param item_id: index of a column in skill_fun that holds the item indicator
    :return: dataframe representing the knowledge structure and a dict of equivalence classes of competence states
    """
    # extracting skill set
    data = skill_fun.drop(skill_fun.columns[item_id], axis=1)
    skills = list(data)
    # extracting item set with corresponding skills
    items = []
    items_skills = {}
    for row in data.itertuples():
        item = skill_fun.iloc[row.Index, item_id]
        item_skill = ""
        for i in range(1, len(row)):
            if int(row[i]) == 1:
                item_skill += skills[i-1]
        if item not in items:
            items.append(item)
            items_skills[item] = []
        items_skills[item].append(item_skill)
    # generating 2 ^ skills mapping
    combinations = get_all_combinations(skills)
    # empty set
    # generating knowledge structure and appropriate classes
    values = ['0' * len(items)]
    classes = {}
    classes['0'] = values[0]
    # generating from skill function
    for combination in combinations:
        value = ""
        for item in items:
            if contains_string(combination, items_skills[item]):
                value += "1"
            else:
                value += "0"
        classes[combination] = value
        values.append(value)
    return convert_as_bin_mat(values, items), classes


def get_all_combinations(input_chars):
    """
    Generate all combinations of given characters
    :param input_chars: input characters
    :return: list of all combinations
    """
    ret_val = []
    for i in range(len(input_chars)):
        temp = list(itertools.combinations(input_chars, i + 1))
        for t in temp:
            ret_val.append(''.join(t))
    return ret_val


def contains_string(src, dest):
    """
    Checking if destination string contains any subset of source string
    :param src: source string
    :param dest: list of destination strings
    :return: True or False
    """
    chars = list(src)
    combinations = get_all_combinations(chars)
    for combination in combinations:
        if combination in dest:
            return True
    return False
