import unittest
import pandas as pd
import numpy as np
import sys
sys.path.append('../learning_spaces/')
from learning_spaces.kst import orig_iita


class TestOrigIita(unittest.TestCase):

    def test_orig_iita_with_dataframe(self):
        data_frame = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 0], 'c': [0, 1, 1]})
        A = [[(2, 1)], [(0, 1), (0, 2), (2, 0), (2, 1)]]
        response = orig_iita(data_frame, A)

        self.assertEqual([0.20370370370370369, 0.39814814814814814], response['diff.value'].tolist())
        self.assertEqual([0.0, 0.5], response['error.rate'].tolist())

    def test_orig_iita_with_martrix(self):
        matrix = np.matrix([[1, 0, 0], [0, 1, 1], [1, 0, 1]])
        A = [[(2, 1)], [(0, 1), (0, 2), (2, 0), (2, 1)]]
        response = orig_iita(matrix, A)

        self.assertEqual([0.20370370370370369, 0.39814814814814814], response['diff.value'].tolist())
        self.assertEqual([0.0, 0.5], response['error.rate'].tolist())


if __name__ == '__main__':
    unittest.main()
