import unittest
import pandas as pd
import numpy as np
import sys
sys.path.append('../learning_spaces/')
from learning_spaces.kst import mini_iita


class TestMiniIita(unittest.TestCase):

    def test_mini_iita_with_dataframe(self):
        data_frame = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 0], 'c': [0, 1, 1]})
        A = [[(2, 1)], [(0, 1), (0, 2), (2, 0), (2, 1)]]
        response = mini_iita(data_frame, A)

        self.assertEqual([0.18518518518518515, 0.16666666666666666], response['diff.value'].tolist())
        self.assertEqual([0.0, 0.5], response['error.rate'].tolist())

    def test_mini_iita_with_matrix(self):
        matrix = np.matrix([[1, 0, 0], [0, 1, 1], [1, 0, 1]])
        A = [[(2, 1)], [(0, 1), (0, 2), (2, 0), (2, 1)]]
        response = mini_iita(matrix, A)

        self.assertEqual([0.18518518518518515, 0.16666666666666666], response['diff.value'].tolist())
        self.assertEqual([0.0, 0.5], response['error.rate'].tolist())


if __name__ == '__main__':
    unittest.main()
