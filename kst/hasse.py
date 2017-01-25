import pydot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tempfile


def hasse(imp, items):
    """
    Hasse diagram of Surmise Relation
    Plots the Hasse diagram of surmise relation.

    :param imp: list of implications
    :param items: number of items of the domain
    :return: produces a plot and returns a list of the equally informative items
    """

    parallel_items = {}
    implications = list(imp)

    # generate partially ordered set
    for i in implications:
        if (i[1], i[0]) in implications:
            if i[0] in parallel_items:
                parallel_items[i[0]].append(i[1])
            else:
                parallel_items[i[0]] = [i[0], i[1]]
            implications.remove(i)
            implications.remove((i[1], i[0]))
            for j in range(len(implications)):
                if i[1] == implications[j][0]:
                    implications[j] = (i[0], implications[j][1])
                elif i[1] == implications[j][1]:
                    implications[j] = (implications[j][0], i[0])

    implications = list(set(implications))

    for i in implications:
        for j in range(items):
            if (i[0] != j) and (i[1] != j) and ((i[0], j) in implications) and ((i[1], j) in implications):
                implications.remove((i[0], j))

    # bottom-up approach
    for i in range(len(implications)):
        implications[i] = (implications[i][1], implications[i][0])

    graph = pydot.Dot(graph_type='graph')
    for i in implications:
        graph.add_edge(pydot.Edge(i[0], i[1]))

    fout = tempfile.NamedTemporaryFile(suffix=".png")
    graph.write(fout.name, format="png")
    img = mpimg.imread(fout.name)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    return [list(set(value)) for key, value in parallel_items.items()]