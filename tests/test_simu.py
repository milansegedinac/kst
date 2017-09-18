import unittest
import sys
sys.path.append('../learning_spaces/')
from learning_spaces.kst import simu


class TestSimu(unittest.TestCase):

    def test_simu(self):
        result = simu(items=3, size=3, ce=0.0, lg=0.0, delta=0.0)

        self.assertTrue('dataset' in result)
        self.assertTrue('implications' in result)
        self.assertTrue('states' in result)


if __name__ == '__main__':
    unittest.main()
