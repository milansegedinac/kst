import numpy as np
from kst import imp2state


def simu(items, size, ce, lg, delta, imp=None):
    """
    Data and Quasi Order Simulation Tool

    :param items: number of items of the domain taken as a basis for the simulation
    :param size: number of response patterns to be simulated (the sample size)
    :param ce: probability for a careless error
    :param lg: probability for a lucky guess
    :param delta: probability for adding an item pair to the randomly generated quasi order
    :param imp: list of implications (assumed to be a quasi order) used for simulating the data
    :return: dictionary
    """

    R = set()

    if imp is None:
        # computation of transitive relations
        for i in range(items):
            for j in range(items):
                if (i != j) and (delta > np.random.uniform(1, 0, 1)):
                    R.update({(i, j)})
                if i == j:
                    R.update({(i, j)})

        R_2 = np.zeros((items, items), dtype=np.int8)
        for t in R:
            R_2[t[0], t[1]] = 1

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
            G.insert(i + 1, G[i].union(H))

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

        # implications
        imp = set()
        for i in range(items):
            for j in range(items):
                if (i != j) and (base_list[i].issubset(base_list[j])):
                    imp.update({(i, j)})
    else:
        # patterns
        P = imp2state(imp, items)

    # simulating the dataset
    sim = np.zeros((size, items), dtype=np.int8)

    for i in range(size):
        sim[i,] = P[np.random.randint(0, P.shape[0], 1), ]
        for j in range(items):
            if (sim[i, j] == 1) and (np.random.uniform(1, 0, 1) < ce):
                sim[i, j] = 0
            if (sim[i, j] == 0) and (np.random.uniform(1, 0, 1) < lg):
                sim[i, j] = 1

    return {'dataset': sim, 'implications': imp, 'states': P}
