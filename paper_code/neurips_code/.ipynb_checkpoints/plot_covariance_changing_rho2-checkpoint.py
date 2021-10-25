import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sc

from neurips_code.utils import *
from neurips_code.predicted_risk import *
from neurips_code.Regressor import LinearRegressor
from datetime import datetime as dt
from multiprocessing import Pool





def display_cov_errors_rho_2(n = 100,
                        gamma = 2,
                        r2 = 2,
                        sigma2 = 1,

                        strong_feature_ratio = 1/2,
                        start_eval_ratio = 0.5,
                        end_eval_ratio = 5,
                        change_strong_feature = True,

                        ord = 'fro',

                        empir = 'gl',
                        alpha = 0.25,

                        savefile = False,
                        ):

    ''' This function generates a plot of errors in the approximation of the covariance matrix versus eigenvalue ratio in the strong weak features model
        of covariance matrices.

        Parameters:

    ------------------------------

    n : int

        Number of datapoints in simulation.

    gamma : int > 1

        Level of overparametrization. The dimension of the data is int(gamm*n)

    r2 : int > 0

        Signal. When the prior of the true parameter is isotropic, r^2 is its expected squared norm.

    sigma2 : int > 0

        Variance of the noise variables.

    strong_feature_ratio : float in (0,1)

        int(gamma*n*strong_feature_ratio) is the number of 'strong_feature' eigenvalues on the diagonal of the covariance matrix.

    start_eval_ratio : float > 0

        Smallest value of the eigenvalue_ratio = strong_feature/weak_feature in the plot.

    end_eval_ratio : float > 0

        Largest value of the eigenvalue_ratio = strong_feature/weak_feature in the plot.

    change_strong_feature : Boolean

        If true we fix the weak_feature eigenvalue to 1 and change the strong_feature eigenvalue. Vice versa if false.

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



    d = int(gamma*n)
    snr = r2/sigma2

    # list of eval_ratios in the plots
    eval_ratios = np.concatenate((np.linspace(start_eval_ratio,start_eval_ratio+(end_eval_ratio-start_eval_ratio)/3,8),np.linspace(start_eval_ratio+(end_eval_ratio-start_eval_ratio)/3+(end_eval_ratio-start_eval_ratio)/15, end_eval_ratio,7)))

    errors = np.zeros(len(eval_ratios))

    count = 0
    for i in range(len(eval_ratios)):
        eval_ratio = eval_ratios[i]
        count = count+1
        print(count)

        # which eigenvalue of the covariance matrix we change
        if change_strong_feature:
            weak_feature = 1
            strong_feature = eval_ratio
        else:
            strong_feature = 1
            weak_feature = 1/eval_ratio

        # generate covariance matrix of data
        c = generate_c(ro = None, regime = 'strong_weak_features', n = n, d = d,
                        strong_feature = strong_feature,
                        strong_feature_ratio = strong_feature_ratio,
                        weak_feature = weak_feature)


        # generate data
        X, y, xi = generate_data(np.zeros(d), c, n, d, sigma2)
        print('data generated')

        # generate empirical estimate of the covariance matrix
        c_e = generate_c_empir(X, empir, alpha)
        print('matrix approximation computed')

        errors[i] = np.linalg.norm(c_e-c, ord = ord)

    # make plots
    fig, ax = plt.subplots()

    ax.plot(eval_ratios, errors, 'bo')


    ax.set_ylabel('Error', fontsize = 'large')
    ax.set_xlabel(r'$\rho_1/\rho_2$',fontsize = 'large')
    ax.set_title('Approximation of the covariance matrix')
    if ord == 'fro':
        ax.set_ylim([0,5])
    elif ord == 2:
        ax.set_ylim([0,1])
    #ax.legend()

    # save the plots
    if savefile:
        dtstamp = str(dt.now()).replace(' ', '_').replace(':','-').replace('.','_')
        fig.savefig(f'images/cov_approx_changing_rho2_n_{n}_ord_{ord}_r2_{r2}_sigma2_{sigma2}_gamma_{gamma}_strongfeature_{strong_feature}_strongfeatureratio_{strong_feature_ratio}_alpha_{alpha}_final_{dtstamp}.pdf', format = 'pdf')

    return



if __name__ == '__main__':

    display_cov_errors_rho_2(n = 100,
                            gamma = 2,
                            r2 = 2,
                            sigma2 = 1,

                            strong_feature_ratio = 1/2,
                            start_eval_ratio = 0.5,
                            end_eval_ratio = 5,
                            change_strong_feature = True,

                            empir = 'gl',
                            alpha = 0.25,

                            savefile = False,
                            )
