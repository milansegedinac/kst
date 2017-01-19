import unittest
import pandas as pd
from kst import mini_iita


class MiniIitaTest(unittest.TestCase):

    def test_mini_iita(self):
        data_frame = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 0], 'c': [0, 1, 1]})
        A = [[(2, 1)], [(0, 1), (0, 2), (2, 0), (2, 1)]]
        response = mini_iita(data_frame, A)

        self.assertCountEqual([0.18518518518518515, 0.16666666666666666], response['diff.value'])
        self.assertCountEqual([0.0, 0.5], response['error.rate'])


if __name__ == '__main__':
    unittest.main()
