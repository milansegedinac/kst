import unittest
import pandas as pd
from kst import iita


class IitaTest(unittest.TestCase):

    def setUp(self):
        self.dataset = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 0], 'c': [0, 1, 1]})

    def test_iita_with_invalid_first_argument(self):
        self.assertRaises(SystemExit, lambda: iita(pd.DataFrame({'a': [1, 0, 1]}), v=1))
        self.assertRaises(SystemExit, lambda: iita('Invalid dataset', v=1))

    def test_iita_when_dataset_has_nan_values(self):
        dataset = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, float('nan')], 'c': [0, 1, 1]})
        self.assertRaises(SystemExit, lambda: iita(dataset, v=1))

    def test_iita_when_dataset_has_invalid_values(self):
        dataset = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 5], 'c': [0, 1, 1]})
        self.assertRaises(SystemExit, lambda: iita(dataset, v=1))

    def test_iita_with_invalid_second_argument(self):
        self.assertRaises(SystemExit, lambda: iita(self.dataset, -100))
        self.assertRaises(SystemExit, lambda: iita(self.dataset, -1))
        self.assertRaises(SystemExit, lambda: iita(self.dataset, 0))
        self.assertRaises(SystemExit, lambda: iita(self.dataset, 4))
        self.assertRaises(SystemExit, lambda: iita(self.dataset, 100))
        self.assertRaises(SystemExit, lambda: iita(self.dataset, (1, 2)))
        self.assertRaises(SystemExit, lambda: iita(self.dataset, [1, 2]))

    def test_mini_iita(self):
        response = iita(self.dataset, v=1)

        self.assertEqual([0.18518518518518515, 0.16666666666666666, 0.21296296296296294], response['diff'].tolist())
        self.assertEqual(0.5, response['error.rate'])
        self.assertEqual([(0, 1), (0, 2), (2, 0), (2, 1)], response['implications'])
        self.assertEqual(1, response['selection.set.index'])
        self.assertEqual(1, response['v'])

    def test_corr_iita(self):
        response = iita(self.dataset, v=2)

        self.assertEqual([0.18518518518518515, 0.16666666666666666, 0.21527777777777779], response['diff'].tolist())
        self.assertEqual(0.5, response['error.rate'])
        self.assertEqual([(0, 1), (0, 2), (2, 0), (2, 1)], response['implications'])
        self.assertEqual(1, response['selection.set.index'])
        self.assertEqual(2, response['v'])

    def test_orig_iita(self):
        response = iita(self.dataset, v=3)

        self.assertEqual([0.20370370370370369, 0.39814814814814814, 0.21527777777777779], response['diff'].tolist())
        self.assertEqual(0, response['error.rate'])
        self.assertEqual([(2, 1)], response['implications'])
        self.assertEqual(0, response['selection.set.index'])
        self.assertEqual(3, response['v'])

if __name__ == '__main__':
    unittest.main()
