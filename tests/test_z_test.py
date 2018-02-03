import unittest
import sys
sys.path.append('../learning_spaces/')
from learning_spaces.kst import z_test


class TestZTest(unittest.TestCase):

    def test_ZTest(self):
        result = z_test()

        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()