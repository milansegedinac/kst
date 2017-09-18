import unittest
import pandas as pd
import sys
sys.path.append('../learning_spaces/')
from learning_spaces.pks import delineation


class TestDelineation(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv("data/test_delineate.csv")
        self.columns = ['e', 'f', 'g', 'h']
        self.values = [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ]
        self.ddf = pd.DataFrame(self.values, columns=self.columns)
        self.classes = {'s': '0010', 'su': '1110', 'st': '1011', 'u': '0100', 'tu': '0111', 'stu': '1111',
                        '0': '0000', 't': '0011'}

    def test_delineate_df(self):
        dataframe, classes = delineation.delineate(self.df)
        self.assertDictEqual(self.classes, classes)
        self.assertListEqual(list(self.ddf), list(dataframe))
        self.assertListEqual(self.ddf.values.tolist(), dataframe.values.tolist())
