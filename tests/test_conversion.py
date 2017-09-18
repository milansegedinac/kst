import unittest
import pandas as pd
import sys
sys.path.append('../learning_spaces/')
from learning_spaces.pks import conversion


class TestConversion(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv("data/test_data.csv")

    def test_convert_as_pattern_df(self):
        response = conversion.convert_as_pattern(self.df)
        self.assertListEqual(['00000', '10000', '01000', '11000', '11100', '11010', '11110', '11101', '11111'], response)

    def test_convert_as_pattern_df_freq(self):
        patterns, freq = conversion.convert_as_pattern(self.df, freq=True)
        self.assertListEqual(['00000', '10000', '01000', '11000', '11100', '11010', '11110', '11101', '11111'], patterns)
        self.assertListEqual([1, 1, 1, 1, 1, 1, 1, 1, 1], freq)

    def test_convert_as_bin_mat_df(self):
        pattern = conversion.convert_as_pattern(self.df)
        response = conversion.convert_as_bin_mat(pattern)
        self.assertListEqual(list(self.df), list(response))
        self.assertListEqual(self.df.values.tolist(), response.values.tolist())

    def test_convert_as_bin_mat_df_col_names(self):
        pattern = conversion.convert_as_pattern(self.df)
        col_names = ['i', 'j', 'k', 'l', 'm']
        response = conversion.convert_as_bin_mat(pattern, col_names=col_names)
        self.assertListEqual(col_names, list(response))
        self.assertListEqual(self.df.values.tolist(), response.values.tolist())
