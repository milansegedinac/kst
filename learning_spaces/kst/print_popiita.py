import numpy as np


def print_popiita(obj):
    print("\n \t Inductive Item Tree Analysis in population values\n")
    print("\nAlgorithm:")
    if (obj['v'] == 1):
        print(" minimized corrected IITA\n")
    if (obj['v'] == 2):
        print(" corrected IITA\n")
    if (obj['v'] == 3):
        print(" original IITA\n")
    print("\npopulation diff values:\n")
    print(round(obj['pop.diff'],  3))
    print("\npopulation error rates:\n")
    print(round(obj['error.pop'],  3))
    print("\nquasi order:\n")
    selection_set = obj['selection.set']
    diff = obj['pop.diff']
    index = np.min(np.where(selection_set == diff))
    print(selection_set[index][0])






