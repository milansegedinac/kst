import pandas as pd
import numpy as np
from scipy.stats import chi2
from .conversion import convert_as_pattern, convert_as_bin_mat


class BLIM:
    """
    Fits a basic local independence model (BLIM) for probabilistic knowledge structures by
    Minimum Discrepancy and Maximum Likelihood estimation
    """

    def __init__(self, k, n_r, method="MD", r=None, p_k=None, beta=None, eta=None, rand_init=False, inc_radius=0,
                 tol=1e-07, max_iter=10000):
        """
        Fit a Basic Local Independence Model (BLIM) for probabilistic knowledge structures by
        Minimum Discrepancy Maximum Likelihood estimation
        :param k: a dataframe with binary matrix representing the knowledge structure
        :param n_r: dictionary of absolute frequencies of response patterns
        :param method: MD for Minimum Discrepancy estimation, ML for Maximum Likelihood estimation
        :param r: dataframe with binary matrix of unique response patterns. Per default inferred from the names of n_r
        :param p_k: list of initial parameter values for probabilities of knowledge states
        :param beta: list of initial parameter values for probabilities of a careless error
        :param eta: list of initial parameter values for probabilities of a lucky guess
        :param rand_init: if True then initial parameter values are sampled uniformly with constraints
        :param inc_radius: include knowledge states of distance from the minimum discrepant states less than or equal
         to inc_radius
        :param tol: tolerance, stopping criterion for iteration
        :param max_iter: the maximum number of iterations
        """
        # setting initial BLIM object values
        self.k = k
        self.n_r = n_r
        self.n_total = sum(n_r.values())
        self.method = method
        self.n_items = len(list(k))
        if r is None:
            r = convert_as_bin_mat(list(self.n_r), col_names=list(self.k))
        self.n_patterns = len(r.index)
        self.n_states = len(k.index)
        if p_k is None:
            self.p_k = [1 / self.n_states] * self.n_states
        else:
            self.p_k = p_k
        if beta is None:
            self.beta = [0.1] * self.n_items
        else:
            self.beta = beta
        if eta is None:
            self.eta = [0.1] * self.n_items
        else:
            self.eta = eta

        # uniformly random initial values
        if rand_init:
            self.beta = np.random.uniform(0, 1, self.n_items)
            self.eta = np.random.uniform(0, 1, self.n_items)
            # constraint: beta + eta < 1
            for i in range(len(self.beta)):
                if self.beta[i] + self.eta[i] >= 1:
                    self.beta[i] = 1 - self.beta[i]
                    self.eta[i] = 1 - self.eta[i]
            # constraint sum(p_k) == 1
            x = []
            x.append(0)
            x += list(np.random.uniform(0, 1, self.n_states - 1))
            x.append(1)
            x.sort()
            a = x[1:]
            b = x[:-1]
            for i in range(len(self.p_k)):
                self.p_k[i] = a[i] - b[i]

        # converting to dataframes
        self.p_k = pd.DataFrame([self.p_k], columns=convert_as_pattern(self.k))
        self.beta = pd.DataFrame([self.beta], columns=list(self.k))
        self.eta = pd.DataFrame([self.eta], columns=list(self.k))
        # assigning state K given response R
        d_rk_header = convert_as_pattern(self.k)
        d_rk = pd.DataFrame(columns=d_rk_header)
        for i in range(len(self.k.index)):
            rk_matrix = np.logical_xor(r, list(self.k.iloc[i]))
            d_rk[d_rk_header[i]] = list(rk_matrix.sum(axis=1))
        # minimum discrepancy
        d_min = d_rk.apply(min, axis=1)
        i_rk = np.logical_and(d_rk <= list(d_min + inc_radius), ~(d_rk is None))
        # minimum discrepancy distribution
        frequencies = list(self.n_r.values())
        values = pd.unique(d_min)
        sums = {}
        disc_sum = 0
        disc_count = 0
        for value in values:
            sums[value] = 0
        for i in range(len(d_min)):
            sums[d_min[i]] += frequencies[i]
            disc_sum += d_min[i] * frequencies[i]
            disc_count += frequencies[i]
        self.discrepancy = disc_sum / disc_count
        self.disc_tab = pd.DataFrame(sums, columns=sums.keys(), index=[0])

        # selected method
        em = 1
        if method == "MD":
            em = 0
        md = 1
        if method == "ML":
            md = 0

        self.iteration = 0
        max_diff = 2 * tol
        beta_num = self.beta.copy(deep=True)
        beta_denom = self.beta.copy(deep=True)
        eta_num = self.beta.copy(deep=True)
        eta_denom = self.beta.copy(deep=True)

        while (max_diff > tol) and (self.iteration < max_iter) and ((md * (1 - em) != 1) or (self.iteration == 0)):
            pi_old = self.p_k.copy(deep=True)
            beta_old = self.beta.copy(deep=True)
            eta_old = self.eta.copy(deep=True)

            p_r_k = pd.DataFrame(columns=d_rk_header)
            for i in range(len(self.k.index)):
                p_r_k[d_rk_header[i]] = calculate_p_r_k(self.k.iloc[i], self.beta, self.eta, r)

            p_r = numpy_list_to_list(np.inner(np.asmatrix(p_r_k), np.asarray(self.p_k)).tolist())
            # prediction of P(K|R)
            p_k_r = pd.DataFrame(np.multiply(np.asmatrix(p_r_k), np.outer((1 / np.asarray(p_r)), np.asarray(self.p_k))),
                                 columns=d_rk_header)

            mat_rk = pd.DataFrame(np.multiply(np.asmatrix(i_rk ** md), np.asmatrix(p_k_r ** em)), columns=d_rk_header)

            # m_r_k = E(M_RK) = P(K|R) * N(R)
            np_mat_rk = np.asmatrix(mat_rk)
            mat_rk_row_sum = np_mat_rk / np_mat_rk.sum(axis=1)
            list_n_r = np.array(list(self.n_r.values()))
            m_r_k = pd.DataFrame(np.multiply(mat_rk_row_sum, list_n_r[:, np.newaxis]), columns=d_rk_header)

            # distribution of knowledge states
            self.p_k = m_r_k.sum(axis=0) / self.n_total

            # careless error and guessing parameters
            k_header = list(self.k)
            for i in range(self.n_items):
                current_header = k_header[i]
                # filter by columns first
                del_col_0 = np.where(np.array(self.k[current_header]) == 0)[0]
                m_r_k_0 = m_r_k.drop(m_r_k.columns[del_col_0], axis=1)
                del_col_1 = np.where(np.array(self.k[current_header]) == 1)[0]
                m_r_k_1 = m_r_k.drop(m_r_k.columns[del_col_1], axis=1)
                # calculate errors
                beta_num[current_header] = m_r_k_0.loc[r[current_header] == 0].values.sum()
                beta_denom[current_header] = m_r_k_0.values.sum()
                eta_num[current_header] = m_r_k_1.loc[r[current_header] == 1].values.sum()
                eta_denom[current_header] = m_r_k_1.values.sum()

            # updating error values
            for header in k_header:
                self.beta[header] = beta_num[header] / beta_denom[header]
                self.beta.fillna(0)
                self.eta[header] = eta_num[header] / eta_denom[header]
                self.eta.fillna(0)

            # updating max_diff
            p_max = np.amax(abs(self.p_k - pi_old).values)
            beta_max = np.amax(abs(self.beta - beta_old).values)
            eta_max = np.amax(abs(self.eta - eta_old).values)
            max_diff = max(p_max, beta_max, eta_max)
            # updating iterations
            self.iteration += 1

        if self.iteration >= max_iter:
            print("Iteration maximum has been exceeded")

        # mean number of errors
        p_kq = [0] * self.n_items
        for i in range(self.n_items):
            current_header = k_header[i]
            selected_headers = np.where(np.array(self.k[current_header] == 1))[0]
            sums = 0
            for header in selected_headers:
                sums += self.p_k[d_rk_header[header]]
            p_kq[i] = sums

        self.n_errors = {}
        self.n_errors['careless'] = (self.beta * p_kq).values.sum()
        self.n_errors['lucky'] = (self.eta * (1 - np.array(p_kq))).values.sum()

        # recompute predictions and likelihood
        for i in range(len(self.k.index)):
            p_r_k[d_rk_header[i]] = calculate_p_r_k(self.k.iloc[i], self.beta, self.eta, r)

        p_r = np.inner(np.asmatrix(p_r_k), np.asarray(self.p_k)).tolist()[0]
        if sum(p_r) < 1:
            p_r = p_r / sum(p_r)

        self.log_lik = sum(np.log(p_r) * list(self.n_r.values()))

        # goodness of fit
        self.goodness_of_fit = {}
        fitted = np.asarray(p_r) * self.n_total
        self.fitted_values = pd.DataFrame([fitted], columns=self.n_r.keys())
        n_r_list = list(self.n_r.values())

        self.goodness_of_fit['g2'] = 2 * sum(n_r_list * np.log(n_r_list / fitted))
        self.goodness_of_fit['df'] = min(2 ** self.n_items - 1, self.n_total) - 2 * self.n_states
        self.goodness_of_fit['pval'] = 1 - chi2.cdf(self.goodness_of_fit['g2'], self.goodness_of_fit['df'])

    def describe(self):
        """
        Print BLIM object values
        """
        print("\nBasic local independence models (BLIMs)\n")
        print("Number of knowledge states: {0}".format(self.n_states))
        print("Number of response patterns: {0}".format(self.n_patterns))
        print("Number of respondents: {0}".format(self.n_total))
        print("\nMethod: " + self.method)
        print("Number of iterations: {0}".format(self.iteration))
        g2 = self.goodness_of_fit['g2']
        df = self.goodness_of_fit['df']
        pval = self.goodness_of_fit['pval']
        print("Goodness of fit (2 log likelihood ratio):\n")
        print("\tG2({0}) = {1}, p = {2} \n".format(df, g2, pval))
        print("Minimum discrepancy distribution (mean = {0})\n".format(self.discrepancy))
        print("Mean number of errors (total = {0})".format(sum(self.n_errors.values())))
        print(self.n_errors)
        print("\nDistribution of knowledge states:")
        print(self.p_k)
        print("\nError and guessing parameters:")
        print("Beta")
        print(self.beta)
        print("Eta")
        print(self.eta)

    def log_likelihood(self):
        """
        Log-Likelihood for BLIM object
        """
        return self.log_lik

    def number_of_observations(self):
        """
        Number of observations
        """
        return self.n_patterns

    def simulate(self):
        """
        Simulates responses from the distribution corresponding to a fitted BLIM model object.
        :return: dataframe of frequencies of response patterns
        """
        seq_len = list(range(len(self.p_k.values)))
        states_id = np.random.choice(seq_len, size=self.n_total, replace=True, p=self.p_k.values)
        beta_inv = 1 - self.beta
        # P(resp = 1 | K)
        p_1_k = np.multiply(np.asmatrix(self.k), np.asarray(beta_inv.values)) + np.multiply(np.asmatrix(1 - self.k), np.asarray(self.eta.values))
        p_1_k_df = pd.DataFrame(np.transpose(p_1_k), columns=convert_as_pattern(self.k))
        # initialize response matrix
        r_mat = pd.DataFrame(0, index=np.arange(self.n_total), columns=list(self.k))
        # draw a response
        for i in range(self.n_total):
            r_mat.loc[i, :] = np.random.binomial(n=1, size=self.n_items, p=np.array(p_1_k_df.iloc[:, states_id[i]]))

        patterns, frequencies = convert_as_pattern(r_mat, freq=True)
        return pd.DataFrame([frequencies], columns=patterns)

    def deviance(self):
        """
        Deviance
        """
        return self.goodness_of_fit['g2']

    def coef(self):
        """
        BLIM object parameters
        :return: dataframe for beta, eta nad p_k
        """
        return self.beta, self.eta, self.p_k


def calculate_p_r_k(k_row, beta, eta, r):
    """
    Calculating P(R|K) for every row from knowledge structure matrix
    :param k_row: dataframe representing knowledge structure matrix row
    :param beta: dataframe representing beta
    :param eta: dataframe representing eta
    :param r: dataframe with binary matrix of unique response patterns
    :return: list of calculated values
    """
    # converting data into numpy arrays and matrices
    k = np.asarray(k_row)
    k_inv = np.asarray(1 - k_row)
    beta_mat = np.asmatrix(beta)
    beta_inv_mat = np.asmatrix(1 - beta)
    eta_mat = np.asmatrix(eta)
    eta_inv_mat = np.asmatrix(1 - eta)
    r_mat = np.asmatrix(r)
    r_inv_mat = np.asmatrix(1 - r)
    # calculating betas
    beta1 = np.power(beta_mat, np.multiply(r_inv_mat, k))
    beta2 = np.power(beta_inv_mat, np.multiply(r_mat, k))
    eta1 = np.power(eta_mat, np.multiply(r_mat, k_inv))
    eta2 = np.power(eta_inv_mat, np.multiply(r_inv_mat, k_inv))
    # multiply betas and etas
    mul_mat = np.multiply(np.multiply(beta1, beta2), np.multiply(eta1, eta2))
    # multiply by row
    row_prod = np.prod(mul_mat, axis=1).tolist()
    return numpy_list_to_list(row_prod)


def numpy_list_to_list(numpy_list):
    """
    Convert nested list to list
    :param numpy_list: nested list
    :return: list
    """
    ret_val = [x[0] for x in numpy_list]
    return ret_val
