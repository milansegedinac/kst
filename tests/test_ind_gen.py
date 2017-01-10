import unittest
import numpy as np
from kst import ind_gen


class IndGenTest(unittest.TestCase):

    def test_ind_gen(self):
        b = np.array([[0, 1, 1], [2, 0, 1], [1, 0, 0]])
        result = ind_gen(b)

        self.assertEqual(3, len(result))
        self.assertIn([(2, 1)], result)
        self.assertIn([(2, 1), (0, 1), (0, 2), (2, 0)], result)
        self.assertIn([(2, 1), (0, 1), (0, 2), (2, 0), (1, 0), (1, 2)], result)


if __name__ == '__main__':
    unittest.main()
