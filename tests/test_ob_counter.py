import unittest
import pandas as pd
from kst import ob_counter


class ObCounterTest(unittest.TestCase):

    def test_ob_counter(self):
        data_frame = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 0], 'c': [0, 1, 1]})
        response = ob_counter(data_frame)

        self.assertEqual([[0, 1, 1], [2, 0, 1], [1, 0, 0]], response.tolist())


if __name__ == '__main__':
    unittest.main()
