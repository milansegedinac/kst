def ind_gen(b):
    """
    Inductive Generation Procedure
    Generates inductively a list of competing quasi orders.

    :param b: matrix of the numbers of counterexamples for all pairs of items
    :return: list of inductively generated quasi orders
    """

    (n, m) = b.shape

    # set of all pairs with a maximum of k-1 counterexamples
    S = []

    # constructed relation for a maximum of k-1 counterexamples
    A = []

    # set of non-transitive triples
    M = []
    M.append([])
    S.append([])
    for i in range(m):
        for j in range(m):
            if (i != j) and (b[i, j] == b.min()):
                S[0].append((i, j))

    A.append(list(S[0]))

    # inductive generation process
    elements = list(set(b.flatten().ravel()))
    elements.sort()
    if 0 in elements:
        elements = elements[1:]

    k = 1

    for element in elements:
        S.insert(k, [])
        A.insert(k, [])
        M.insert(k, [])

        # building of S
        for i in range(m):
            for j in range(m):
                if (i != j) and (b[i, j] <= element) and ((i, j) not in A[k-1]):
                    S[k].append((i, j))

        # transitivity test
        if S[k]:
            M[k] = list(S[k])
            brake_test = 1
            while brake_test != 0:
                brake = list(M[k])
                for i in list(M[k]):
                    for h in range(m):
                        if (h != i[0]) and (h != i[1]) and ((i[1], h) in (A[k-1] + M[k])) and ((i[0], h) not in (A[k-1] + M[k])):
                            if i in M[k]:
                                M[k].remove(i)
                        if (h != i[0]) and (h != i[1]) and ((h, i[0]) in (A[k-1] + M[k])) and ((h, i[1]) not in (A[k-1] + M[k])):
                            if i in M[k]:
                                M[k].remove(i)
                if brake == M[k]:
                    brake_test = 0
            A[k] = A[k-1] + M[k]

        k += 1

    # deletion of empty and duplicated quasi orders
    A = {frozenset(x) for x in A}
    A.discard(set())
    # sort
    A = [sorted(list(x)) for x in A]
    A.sort(key=len)

    return A