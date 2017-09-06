import numpy as np


def imp2state(imp, items):
    """
    Transformation from Implications to Knowledge States
    Transforms a set of implications to the corresponding set of knowledge states (the quasi ordinal knowledge space).

    :param imp: list of implications
    :param items: number of items
    :return: matrix consisted of ones and zeros
    """

    R_2 = np.ones((items, items))
    for i in range(items):
        for j in range(items):
            if (i != j) and ((i, j) not in imp):
                R_2[j, i] = 0

    # base
    base = []

    for i in range(items):
        tmp = []
        for j in range(items):
            if R_2[i, j] == 1:
                tmp.append(j)
        base.insert(i, tmp)

    base_list = []
    for i in range(items):
        base_list.insert(i, set())
        for j in range(len(base[i])):
            base_list[i].update(frozenset([base[i][j]]))

    # span of base
    G = []
    G.insert(0, {frozenset()})
    G.insert(1, set())
    for i in range(len(base[0])):
        G[1].update(frozenset([base[0][i]]))
    G[1] = {frozenset(), frozenset(G[1])}

    for i in range(1, items):
        H = {frozenset()}
        for j in G[i]:
            if not base_list[i].issubset(j):
                for d in range(i):
                    if base_list[d].issubset(j.union(base_list[i])):
                        if base_list[d].issubset(j):
                            H.update(frozenset([j.union(base_list[i])]))
                    if not base_list[d].issubset(j.union(base_list[i])):
                        H.update(frozenset([j.union(base_list[i])]))
        G.insert(i+1, G[i].union(H))

    # patterns
    P = np.zeros((len(G[items]), items), dtype=np.int8)
    i = 0
    sorted_g = [list(i) for i in G[items]]
    sorted_g.sort(key=lambda x: (len(x), x))

    for k in sorted_g:
        for j in range(items):
            if j in k:
                P[i, j] = 1
        i += 1

    return P
