import unittest
import pandas as pd
import sys
sys.path.append('../learning_spaces/')
from learning_spaces.pks import gradedness


class TestGradedness(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv("data/test_data.csv")

    def test_is_forward_graded_df(self):
        response = gradedness.is_forward_graded(self.df)
        self.assertTrue(response['a'])
        self.assertTrue(response['b'])
        self.assertFalse(response['c'])
        self.assertFalse(response['d'])
        self.assertFalse(response['e'])

    def test_is_backward_graded_df(self):
        response = gradedness.is_backward_graded(self.df)
        self.assertFalse(response['a'])
        self.assertFalse(response['b'])
        self.assertFalse(response['c'])
        self.assertTrue(response['d'])
        self.assertTrue(response['e'])
