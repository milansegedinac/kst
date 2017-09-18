import unittest
import numpy as np
import sys
sys.path.append('../learning_spaces/')
from learning_spaces.kst import ind_gen


class TestIndGen(unittest.TestCase):

    def test_ind_gen(self):
        b = np.array([[0, 1, 1], [2, 0, 1], [1, 0, 0]])
        result = ind_gen(b)

        self.assertEqual(3, len(result))
        self.assertEqual([(2, 1)], result[0])
        self.assertEqual([(0, 1), (0, 2), (2, 0), (2, 1)], result[1])
        self.assertEqual([(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)], result[2])


if __name__ == '__main__':
    unittest.main()
