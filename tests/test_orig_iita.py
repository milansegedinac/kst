import unittest
import pandas as pd
from kst import orig_iita


class OrigIitaTest(unittest.TestCase):

    def test_orig_iita(self):
        data_frame = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 0], 'c': [0, 1, 1]})
        A = [[(2, 1)], [(0, 1), (0, 2), (2, 0), (2, 1)]]
        response = orig_iita(data_frame, A)

        self.assertCountEqual([0.20370370370370369, 0.39814814814814814], response['diff.value'])
        self.assertCountEqual([0.0, 0.5], response['error.rate'])


if __name__ == '__main__':
    unittest.main()
