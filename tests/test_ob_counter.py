import unittest
import pandas as pd
import numpy as np
import sys
sys.path.append('../learning_spaces/')
from learning_spaces.kst import ob_counter


class TestObCounter(unittest.TestCase):

    def test_ob_counter_with_dataframe(self):
        data_frame = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 0], 'c': [0, 1, 1]})
        response = ob_counter(data_frame)

        self.assertEqual([[0, 1, 1], [2, 0, 1], [1, 0, 0]], response.tolist())

    def test_ob_counter_with_matrix(self):
        matrix = np.matrix([[1, 0, 0], [0, 1, 1], [1, 0, 1]])
        response = ob_counter(matrix)

        self.assertEqual([[0, 1, 1], [2, 0, 1], [1, 0, 0]], response.tolist())


if __name__ == '__main__':
    unittest.main()
