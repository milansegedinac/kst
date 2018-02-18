import numpy as np


def print_sumpopiita(x):
    print("\n \t Inductive Item Tree Analysis in population values\n")
    print("\nAlgorithm:")
    if (x['v'] == 1) :
        print(" minimized corrected IITA\n")
    if (x['v'] == 2):
        print(" corrected IITA\n")
    if (x['v'] == 3):
        print(" original IITA\n")
    print("\npopulation diff values:\n")
    print(round(x['pop.diff'],  3))
    print("\npopulation error rates:\n")
    print(round(x['error.pop'],  3))
    print("\npopulation matrix:\n")
    print(round(x['pop.matrix'],  3))
    print("\nobtained selection set:\n")
    print(x['selection.set'])