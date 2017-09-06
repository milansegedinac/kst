import unittest
from kst import simu


class SimuTest(unittest.TestCase):

    def test_simu(self):
        result = simu(items=3, size=3, ce=0.0, lg=0.0, delta=0.0)

        self.assertTrue('dataset' in result)
        self.assertTrue('implications' in result)
        self.assertTrue('states' in result)


if __name__ == '__main__':
    unittest.main()
