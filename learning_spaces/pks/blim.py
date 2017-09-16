

class Blim:
    """
    Fits a basic local independence model (BLIM) for probabilistic knowledge structures by
    minimum disperancy maximum likelihood estimation
    """

    def __init__(self):
        print("blim")

    def describe(self, p_k_show=False, err_show=True):
        """

        :param p_k_show:
        :param err_show:
        :return:
        """
        print("\nBasic local independence models (BLIMs)\n")
        print("\nNumber of knowledge states: {0}".format(self.n_states))
        print("\nNumber of response patterns: {0}".format(self.n_patterns))
        print("\nNumber of respondents: {0}".format(self.n_total))
        print("\n\nMethod: " + self.method)
        print("\nNumber of iterations: {0}".format(self.iter))
        g2 = self.goodness_of_fit[0]
        df = self.goodness_of_fit[1]
        pval = self.goodness_of_fit[2]
        print("\nGoodness of fit (2 log likelihood ratio):\n")
        print("\tG2({0}) = {1}, p = {2} \n".format(df, g2, pval))
        print("\nMinimum discrepancy distribution (mean = {0})\n".format(self.discrepancy))
        print("\nMean number of errors (total = {0})\n".format(self.n_error))
        if p_k_show:
            print("\nDistribution of knowledge states\n")  # TODO: upotpuniti
        if err_show:
            print("\nError and guessing parameters\n")  # TODO: upotpuniti


    def log_likelihood(self):
        return None

    def number_of_observations(self):
        return self.n_patterns

    def residuals(self):
        return None

    def display(self):
        return None

    def simulate(self):
        return None

    def deviance(self):
        return self.goodness_of_fit[0]

    def coef(self):
        return None

    def jacobian(self):
        return None
