import unittest
import sys
sys.path.append('../learning_spaces/')
from learning_spaces.kst import imp2state


class TestImp2state(unittest.TestCase):

    def setUp(self):
        # data-provider alternative
        self.tests = [
            {
                'imp': [(1, 0)],
                'items': 2,
                'expected': [[0, 0], [0, 1], [1, 1]]
            },
            {
                'imp': [(0, 1), (0, 2), (2, 0), (2, 1)],
                'items': 3,
                'expected': [[0, 0, 0], [1, 0, 1], [1, 1, 1]]
            },
            {
                'imp': [(0, 1), (0, 3), (1, 0), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1)],
                'items': 4,
                'expected': [[0, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1]]
            },
            {
                'imp': [(0, 3), (0, 4), (2, 0), (2, 3), (2, 4), (3, 0), (3, 4), (4, 0), (4, 3)],
                'items': 5,
                'expected': [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [1, 0, 1, 1, 1], [1, 1, 1, 1, 1]]
            }
        ]

    def test_imp2state(self):
        for test in self.tests:
            self.assertEqual(test['expected'], imp2state(test['imp'], test['items']).tolist())


if __name__ == '__main__':
    unittest.main()
