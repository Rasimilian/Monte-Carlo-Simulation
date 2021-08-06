import numpy as np
import scipy as sc
from scipy.stats import levy_stable, chisquare, cramervonmises_2samp, ks_2samp, ks_1samp, ttest_ind
import matplotlib.pyplot as plt


class Returns:
    """
    Class representing n-days overlapping proportional returns.
    1-day returns are generated using Levy stable distribution.
    """
    def __init__(self, num_observations, ndays, levy_parameters=[1.7, 0.0, 1.0, 1.0]):
        """
        :param int num_observations: number of observations of 1-day returns
        :param int ndays: n-days for overlapping returns
        :param list[int] levy_parameters: parameters for Levy stable distribution
        """
        self.num_observations = num_observations
        self.ndays = ndays
        self.parameters = levy_parameters

    def generate_1day_returns(self):
        """
        Generate 1-day returns using Levy stable distribution.

        :return: 1D array of 1-day returns
        """
        # np.random.seed(42)
        returns_1day = levy_stable.rvs(*self.parameters, size=self.num_observations)
        return returns_1day

    def generate_ndays_returns(self):
        """
        Generate n-days returns using 1-day returns.

        :return: 1D array of n-days returns
        """
        returns_1day = self.generate_1day_returns()
        returns_1day += 1
        returns_ndays = []
        num_returns_ndays = len(returns_1day) - self.ndays + 1  # number of n-days returns
        for i in range(num_returns_ndays):
            return_ndays = np.prod(returns_1day[i:i+self.ndays])
            returns_ndays.append(return_ndays)
        returns_ndays = np.array(returns_ndays) - 1
        return returns_ndays

    def get_percentile(self, q_value=0.01):
        """
        Get percentile(or q_value quantile) of n-days returns.

        :param float q_value: probability value
        :return: percentile
        """
        returns_ndays = self.generate_ndays_returns()
        percentile = np.percentile(returns_ndays, q_value)
        return percentile


def visualize_distribution(data, bins='fd', title='Percentile distribution'):
    """
    Visualize histograms.

    :param float data:
    :param int bins:
    :param str title:
    :return:
    """
    plt.figure()
    plt.title(title)
    plt.xlabel('Values')
    plt.ylabel('Counts')
    plt.hist(data, bins=bins)
    plt.show()


def simulate_monte_carlo(num_observations, ndays, trials=15000):
    """
    Monte Carlo simulation to get percentiles distribution.

    :param int num_observations: number of observations of 1-day returns
    :param int ndays: n-days for overlapping returns
    :param int trials: Monte Carlo trials
    :return: list[float] percentiles
    """
    # Initializing returns
    returns = Returns(num_observations, ndays)
    percentiles = []
    for i in range(trials):
        percentiles.append(returns.get_percentile())
    return percentiles


def check_monte_carlo_trials(method, num_observations, ndays, significance_level=0.6, discrepancy=0.01):
    """
    Check sufficiency of Monte Carlo trials. There are 3 methods to test it numerically.
    Methods:
        'CLT' - Central Limit Theorem approximation. Mean of N percentiles tends to normal distributed variable;
        'monitoring' - observing any data statistic doesn't change abruptly. Mean of percentiles is observed;
        'test_statistic' - using two-sample statistical criteria to test hypotheses;

    :param str method: method name
    :param int num_observations: number of observations of 1-day returns
    :param int ndays: n-days for overlapping returns
    :param float significance_level: the data would be unlikely to occur if the null hypothesis were true
    :param float discrepancy: level for criterion statistic pass
    :return: list[float] percentiles, int trials
    """
    # Initializing returns
    returns = Returns(num_observations, ndays)

    if method == 'CLT':
        std = []    # standard deviation of N = trials percentiles
        mean = []   # mean of N = trials percentiles
        samples = 0
        MAX_TRIALS = 15000
        MAX_SAMPLES = 100
        # samples to plot histogram of means
        while samples < MAX_SAMPLES:
            samples += 1
            percentile = []
            for i in range(MAX_TRIALS):
                percentile.append(returns.get_percentile())
            mean.append(np.mean(percentile))
            std.append(np.std(percentile))

        bins = np.arange(-400000,-200000,5000)
        plt.hist(mean,bins=bins)
        plt.title('Distribution of percentile means')
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.show()
        return mean, MAX_TRIALS

    elif method == 'monitoring':
        mean_perc = []    # mean of percentiles
        percentiles = []    # percentiles
        trial = 0
        MAX_TRIALS = 30000
        trials = []
        while trial < MAX_TRIALS:
            trial += 1
            percentiles.append(returns.get_percentile())
            mean_perc.append(np.mean(percentiles))
            trials.append(trial)

        plt.plot(trials, mean_perc)
        plt.title('Value of percentiles mean ')
        plt.xlabel('Trials')
        plt.ylabel('Means')
        plt.legend()
        plt.show()
        return mean_perc, trials

    elif method == 'test_statistic':
        trials = 0
        p_value = 0
        statistic = 10e8
        p_list1 = []    # the first sample
        p_list2 = []    # the second sample
        while (statistic > discrepancy) or (p_value < significance_level):
            trials += 1
            p_list1.append(returns.get_percentile())
            p_list2.append(returns.get_percentile())

            if trials % 100 == 0:
                ## Using omega-squared test
                # omega_test = cramervonmises_2samp(p_list1, p_list2)
                # statistic, p_value = omega_test.statistic, omega_test.pvalue

                ## Using Kolmogorov-Smirnov test
                statistic, p_value = ks_2samp(p_list1, p_list2)
                print("Statistic: %f, P-value: %f, Trials: %d" % (statistic, p_value, trials))

            if trials > 200000:
                print("Simulation failed!")
                break

        # Original sample
        percentiles = p_list1 + p_list2
        return percentiles, trials


percentiles, trials = check_monte_carlo_trials('test_statistic', 750, 10)
visualize_distribution(percentiles)