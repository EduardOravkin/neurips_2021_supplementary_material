import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sc

from neurips_code.utils import *
from neurips_code.predicted_risk import *
from neurips_code.Regressor import LinearRegressor
from datetime import datetime as dt


def display_cov_errors_gamma(n = 100,
                        r2=1,
                        sigma2=1,

                        start_gamma = 1.4,
                        end_gamma = 4,

                        regime='exponential',
                        ro=0.5,

                        strong_feature_ratio = 1/2,
                        strong_feature = 1,
                        weak_feature = 1/5,

                        ord = 'fro',

                        empir = 'gl',
                        alpha = 0.25,

                        savefile = False,
                        ):

    ''' This function generates a plot of risk versus gamma (level of overparametrization d/n) for given regime
        of covariance matrices.

        Parameters:

    ------------------------------

    n : int

        Number of datapoints in simulation.

    r2 : int > 0

        Signal. When the prior of the true parameter is isotropic, r^2 is its expected squared norm.

    sigma2 : int > 0

        Variance of the noise variables.

    start_gamma : float > 0

        Smallest value of gamma (n/d) in the plot.

    end_gamma : float > 0

        Largest value of gamma (n/d) in the plot.

    regime : 'id', 'autoregressive', 'strong_weak_features', 'exponential'

        Specifies the regime of covariance matrices of the features to be used.

    ro : float in (0,1)

        Parameter used by the 'autoregressive' regime of covariance matrices.

    strong_feature_ratio : float in (0,1)

        Parameter used by the 'strong_weak_features' regime of covariance matrices.
        int(gamma*n*strong_feature_ratio) is the number of 'strong_feature' eigenvalues on the diagonal of the covariance matrix.

    strong_feature: float > 0

        Parameter used by the 'strong_weak_features' regime of covariance matrices.

    weak_feature: float > 0

        Parameter used by the 'strong_weak_features' regime of covariance matrices.

    ord : None, 'fro', 'nuc', 1, 2, np.inf

        Which norm to use when computing the distance between matrices

    empir : {'gl', 'basic', 'lw'}

        Specifies what kind of approximation of the population covariance matrix we use. 'gl' is the GraphicalLasso
        approximation, 'basic' is the standard X^TX/n, 'lw' is the LedoitWolf approximation.

    alpha : float > 0

        Only used if empir = 'gl'. Specifies the regularization of the GraphicalLasso approximation.
        This can be also crossvalidated, using GraphicalLassoCV from sklearn.

    savefile : Boolean

        If true then saves the generated plot.

        '''

    snr = r2/sigma2

    # generate sequence of gammas for plotting
    gammas = np.concatenate((np.linspace(start_gamma,start_gamma+(end_gamma-start_gamma)/3,8),np.linspace(start_gamma+(end_gamma-start_gamma)/3+(end_gamma-start_gamma)/15,end_gamma,7)))

    errors = np.zeros(len(gammas))

    count = 0
    # do experiment for each gamma
    for i in range(len(gammas)):
        count = count+1
        print(f'{count}/15')

        gamma = gammas[i]
        d = int(gamma*n)

        # generate covariance matrix of data
        c = generate_c(ro = ro, regime = regime, n = n, d = d,
                        strong_feature = strong_feature,
                        strong_feature_ratio = strong_feature_ratio,
                        weak_feature = weak_feature)

        # generate data
        X, y, xi = generate_data(np.zeros(d), c, n, d, sigma2)
        print('data generated')

        # generate empirical estimate of the covariance matrix
        c_e = generate_c_empir(X, empir, alpha)
        print('covariance matrix estimated')

        errors[i] = np.linalg.norm(c_e - c, ord = ord)

    # initialize plots
    fig, ax = plt.subplots()

    ax.plot(gammas, errors, 'bo')

    ax.set_ylabel('Error', fontsize = 'large')
    ax.set_xlabel(r'$\gamma$',fontsize = 'large')
    ax.set_title('Approximation of the covariance matrix')
    if ord == 'fro':
        ax.set_ylim([0,5])
    elif ord == 2:
        ax.set_ylim([0,1])

    if savefile:
        dtstamp = str(dt.now()).replace(' ', '_').replace(':','-').replace('.','_')
        fig.savefig(f'images/cov_approx_changing_gamma_n_{n}_ord_{ord}_r2_{r2}_sigma2_{sigma2}_ro_{str(ro)}_alpha_{str(alpha)}_regime_{regime}_alpha_{alpha}_final_{dtstamp}.pdf', format = 'pdf')

    return





if __name__ == '__main__':

    display_cov_errors_gamma(n = 100,
                        r2=1,
                        sigma2=1,

                        start_gamma = 1.4,
                        end_gamma = 4,

                        regime='exponential',
                        ro=0.5,

                        strong_feature_ratio = 1/2,
                        strong_feature = 1,
                        weak_feature = 1/5,

                        empir = 'gl',
                        alpha = 0.25,

                        savefile = False,
                        )
