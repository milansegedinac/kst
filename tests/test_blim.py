import unittest
import pandas as pd
from collections import OrderedDict
import sys
sys.path.append('../learning_spaces/')
from learning_spaces.pks.blim import BLIM


class TestBlim(unittest.TestCase):

    def setUp(self):
        self.k = pd.read_csv("data/test_data.csv")
        self.n_r = OrderedDict()
        self.n_r["00000"] = 80
        self.n_r["10000"] = 92
        self.n_r["01000"] = 89
        self.n_r["00100"] = 3
        self.n_r["00010"] = 2
        self.n_r["00001"] = 1
        self.n_r["11000"] = 89
        self.n_r["10100"] = 16
        self.n_r["10010"] = 18
        self.n_r["10001"] = 10
        self.n_r["01100"] = 18
        self.n_r["01010"] = 20
        self.n_r["01001"] = 4
        self.n_r["00110"] = 2
        self.n_r["00101"] = 2
        self.n_r["00011"] = 3
        self.n_r["11100"] = 89
        self.n_r["11010"] = 89
        self.n_r["11001"] = 19
        self.n_r["10110"] = 16
        self.n_r["10101"] = 16
        self.n_r["10011"] = 3
        self.n_r["01110"] = 18
        self.n_r["01101"] = 16
        self.n_r["01011"] = 2
        self.n_r["00111"] = 2
        self.n_r["11110"] = 73
        self.n_r["11101"] = 82
        self.n_r["11011"] = 19
        self.n_r["10111"] = 15
        self.n_r["01111"] = 15
        self.n_r["11111"] = 77

    def test_blim_md(self):
        blim_md = BLIM(self.k, self.n_r)
        self.assertEqual(9, blim_md.n_states)
        self.assertEqual(32, blim_md.n_patterns)
        self.assertEqual(1000, blim_md.n_total)
        self.assertEqual("MD", blim_md.method)
        self.assertEqual(1, blim_md.iteration)
        self.assertEqual(91.28362323477515, blim_md.goodness_of_fit['g2'])
        self.assertEqual(13, blim_md.goodness_of_fit['df'])
        self.assertEqual(7.938094626069869e-14, blim_md.goodness_of_fit['pval'])
        self.assertEqual(0.254, blim_md.discrepancy)
        self.assertEqual(0.090000000000000011, blim_md.n_errors['lucky'])
        self.assertEqual(0.16399999999999998, blim_md.n_errors['careless'])
        self.assertListEqual([0.09208874005860192, 0.08871989860583017, 0.04505813953488372, 0.0, 0.0],
                             blim_md.beta.values.tolist()[0])
        self.assertListEqual([0.0, 0.0,  0.04064039408866995,  0.04085801838610828,  0.05472197705207414],
                             blim_md.eta.values.tolist()[0])

    def test_log_likelihood_md(self):
        blim_md = BLIM(self.k, self.n_r)
        self.assertEqual(blim_md.log_lik, blim_md.log_likelihood())

    def test_number_of_obs_md(self):
        blim_md = BLIM(self.k, self.n_r)
        self.assertEqual(blim_md.n_patterns, blim_md.number_of_observations())

    def test_deviance_md(self):
        blim_md = BLIM(self.k, self.n_r)
        self.assertEqual(blim_md.goodness_of_fit['g2'], blim_md.deviance())

    def test_coef_md(self):
        blim_md = BLIM(self.k, self.n_r)
        beta, eta, p_k = blim_md.coef()
        self.assertListEqual(list(blim_md.beta), list(beta))
        self.assertListEqual(blim_md.beta.values.tolist(), beta.values.tolist())
        self.assertListEqual(list(blim_md.eta), list(eta))
        self.assertListEqual(blim_md.eta.values.tolist(), eta.values.tolist())
        self.assertListEqual(list(blim_md.p_k), list(p_k))
        self.assertListEqual(blim_md.p_k.values.tolist(), p_k.values.tolist())

    def test_blim_ml(self):
        blim_ml = BLIM(self.k, self.n_r, method="ML")
        self.assertEqual(9, blim_ml.n_states)
        self.assertEqual(32, blim_ml.n_patterns)
        self.assertEqual(1000, blim_ml.n_total)
        self.assertEqual("ML", blim_ml.method)
        self.assertEqual(300, blim_ml.iteration)
        self.assertEqual(12.622816435940905, blim_ml.goodness_of_fit['g2'])
        self.assertEqual(13, blim_ml.goodness_of_fit['df'])
        self.assertEqual(0.477349992130788, blim_ml.goodness_of_fit['pval'])
        self.assertEqual(0.254, blim_ml.discrepancy)
        self.assertEqual(0.044865390859123146, blim_ml.n_errors['lucky'])
        self.assertEqual(0.44280715825096656, blim_ml.n_errors['careless'])
        self.assertListEqual([0.1648712647718087, 0.16311278151263192, 0.18883863747163213,
                              0.07983530446636058, 0.08864829052919883], blim_ml.beta.values.tolist()[0])
        self.assertListEqual([0.10306473120044671, 0.09507429143942243, 3.5426760020042067e-06,
                              3.157133824028973e-06, 0.019909716488346413], blim_ml.eta.values.tolist()[0])

    def test_log_likelihood_ml(self):
        blim_ml = BLIM(self.k, self.n_r, method="ML")
        self.assertEqual(blim_ml.log_lik, blim_ml.log_likelihood())

    def test_number_of_obs_ml(self):
        blim_ml = BLIM(self.k, self.n_r, method="ML")
        self.assertEqual(blim_ml.n_patterns, blim_ml.number_of_observations())

    def test_deviance_ml(self):
        blim_ml = BLIM(self.k, self.n_r, method="ML")
        self.assertEqual(blim_ml.goodness_of_fit['g2'], blim_ml.deviance())

    def test_coef_ml(self):
        blim_ml = BLIM(self.k, self.n_r, method="ML")
        beta, eta, p_k = blim_ml.coef()
        self.assertListEqual(list(blim_ml.beta), list(beta))
        self.assertListEqual(blim_ml.beta.values.tolist(), beta.values.tolist())
        self.assertListEqual(list(blim_ml.eta), list(eta))
        self.assertListEqual(blim_ml.eta.values.tolist(), eta.values.tolist())
        self.assertListEqual(list(blim_ml.p_k), list(p_k))
        self.assertListEqual(blim_ml.p_k.values.tolist(), p_k.values.tolist())
