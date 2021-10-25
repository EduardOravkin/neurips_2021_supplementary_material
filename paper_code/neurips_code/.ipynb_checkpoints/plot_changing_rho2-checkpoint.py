import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sc

from neurips_code.utils import *
from neurips_code.predicted_risk import *
from neurips_code.Regressor import LinearRegressor
from datetime import datetime as dt
from multiprocessing import Pool





def display_risks_rho_2(n = 100,
                        gamma = 2,
                        r2 = 2,
                        sigma2 = 1,

                        strong_feature_ratio = 1/2,
                        start_eval_ratio = 0.5,
                        end_eval_ratio = 5,
                        change_strong_feature = True,

                        source_condition = 'id',

                        empir = 'gl',
                        alpha = 0.25,

                        include_predictions = False,
                        include_gd = False,
                        include_md = False,
                        include_md_empirical = False,
                        include_best_achievable = True,
                        include_best_achievable_empirical = True,

                        snr_estimation = list(np.linspace(0.1,1,20))+list(np.linspace(1,10,20)),
                        crossval_param = 100,

                        savefile = False,
                        ):

    ''' This function generates a plot of risks versus eigenvalue ratio in the strong weak features model
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

    source_condition : {'id', 'eaesy', 'hard'}

        If 'id' then the covariance matrix of the prior, m, is the idenity. If 'easy' then m = c, if 'hard' then m = c^{-1}
        where c is the covariance matrix of the data.

    empir : {'gl', 'basic', 'lw'}

        Specifies what kind of approximation of the population covariance matrix we use. 'gl' is the GraphicalLasso
        approximation, 'basic' is the standard X^TX/n, 'lw' is the LedoitWolf approximation.

    alpha : float > 0

        Only used if empir = 'gl'. Specifies the regularization of the GraphicalLasso approximation.
        This can be also crossvalidated, using GraphicalLassoCV from sklearn.

    include_predictions : Boolean

        If true then includes the theoretical predictions of the asymptotic riks from the paper.

    include_gd : Boolean

        If true then includes the minimum norm interpolator in the plots.

    include_md : Boolean

        If true then includes the best variance interpolator (covariance mirror descent initialized at 0) in the plots.

    include_md_empirical : Boolean

        If true then includes the empirical approximation of the best variance interpolator.

    include_best_achievable : Boolean

        If true then includes the best linearly achievable interpolator in the plots.

    include_best_achievable_empirical : Boolean

        If true then includes the empirical approximation of the best linearly achievable interpolator in the plots.

    snr_estimation : list

        The list of possible signal-to-noise ratio that the crossvalidation should try when approximating the snr.
        Only used if best_achievable_empirical = True.

    crossval_param : int

        Number of crossvalidation splits to use when estimating the signal-to-noise ratio. Only used if
        best_achievable_empirical = True.

    savefile : Boolean

        If true then saves the generated plot.

        '''



    d = int(gamma*n)
    snr = r2/sigma2

    # list of eval_ratios in the plots
    eval_ratios = np.concatenate((np.linspace(start_eval_ratio,start_eval_ratio+(end_eval_ratio-start_eval_ratio)/3,8),np.linspace(start_eval_ratio+(end_eval_ratio-start_eval_ratio)/3+(end_eval_ratio-start_eval_ratio)/15, end_eval_ratio,7)))

    risks = np.zeros((len(eval_ratios), 6))
    predictions = np.zeros((len(eval_ratios), 3))

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

        # generate covariance matrix of the prior of the true parameter
        m = generate_m(c, n=n, d=d, source_condition = source_condition)

        # generate true parameter
        w_star = generate_true_parameter(n, d, r2, m = m)

        # generate data
        X, y, xi = generate_data(w_star, c, n, d, sigma2)
        print('data generated')

        if sigma2 < 0.0001:
            xi = np.zeros(n)

        # generate empirical estimate of the covariance matrix
        c_e = generate_c_empir(X, empir, alpha)
        print('matrix approximation computed')

        # initialize models
        reg_2 = LinearRegressor()
        reg_c = LinearRegressor()
        reg_ce = LinearRegressor()

        # generate predictors
        # matrix specifies which mirror descent we are using (GD if None)
        reg_2.fit(X, y, matrix = None)
        reg_c.fit(X, y, matrix = c)
        reg_ce.fit(X, y, matrix = c_e)

        # generate the best possible linearly achievable interpolator
        w_a = compute_best_achievable_interpolator(X, y, c, m, snr)
        reg_a = LinearRegressor(init = w_a)

        # generate the empirical approximation of the best possible linearly achievable interpolator
        if include_best_achievable_empirical:
            w_ae = compute_best_achievable_interpolator(X, y, c = c_e, m = np.eye(d), snr = snr_estimation, crossval_param = crossval_param)
            reg_ae = LinearRegressor(init = w_ae)
            print('interpolator approximated')

        # best possible linear predictor (theoretical device)
        c_mhalf = np.linalg.inv(sc.sqrtm(c)) # inverse square root of the covariance matrix
        w_b = c_mhalf.dot( np.linalg.lstsq( X.dot(c_mhalf),  xi, rcond=None)[0] ) + w_star # best possible predictor
        reg_b = LinearRegressor(init = w_b)

        # calculate the expected risks
        risk_2 = calculate_risk(w_star, c, reg_2.w ) + sigma2
        risk_c = calculate_risk(w_star, c, reg_c.w) + sigma2
        risk_ce = calculate_risk(w_star, c, reg_ce.w) + sigma2
        risk_a = calculate_risk(w_star, c, reg_a.w) + sigma2
        if include_best_achievable_empirical:
            risk_ae = calculate_risk(w_star, c, reg_ae.w) + sigma2
        risk_b = calculate_risk(w_star, c, reg_b.w) + sigma2

        # calculate the predicted asymptotic risks
        prediction_2 = risk_gd_strongweak(r2, sigma2, gamma, rho_1 = strong_feature, rho_2 = weak_feature,
                                        psi_1 = strong_feature_ratio, source_condition = source_condition) + sigma2
        prediction_c = risk_md(c, m, r2, sigma2, gamma) + sigma2
        prediction_b = variance_md(sigma2, gamma) + sigma2

        risks[i, :] = risk_2, risk_c, risk_ce, risk_a, risk_ae, risk_b
        predictions[i, :] = prediction_2, prediction_c, prediction_b


    # make plots
    fig, ax = plt.subplots()

    if include_gd:
        ax.plot(eval_ratios, risks[:, 0], 'bo', label = r'$w_{\ell_2}$')
    if include_md:
        ax.plot(eval_ratios, risks[:, 1], 'ro', label = r'$w_{V}$')
    if include_md_empirical:
        ax.plot(eval_ratios, risks[:, 2], 'mo',label = r'$w_{Ve}$', markersize = 4)
    if include_best_achievable:
        ax.plot(eval_ratios, risks[:, 3], 'co',label = r'$w_{O}$')
    if include_best_achievable_empirical:
        ax.plot(eval_ratios, risks[:, 4], 'yo',label = r'$w_{Oe}$', markersize = 4)
    ax.plot(eval_ratios, risks[:, 5], 'ko', label = r'$w_{b}$')


    if include_predictions:
        if include_gd:
            ax.plot(eval_ratios, predictions[:, 0], 'bx', markersize = 3)
        if include_md:
            ax.plot(eval_ratios, predictions[:, 1], 'rx', markersize = 3)
        ax.plot(eval_ratios, predictions[:, 2], 'kx', markersize = 3)


    ax.set_ylabel('Risk', fontsize = 'large')
    ax.set_xlabel(r'$\rho_1/\rho_2$',fontsize = 'large')
    ax.set_title('Comparison of interpolators')
    ax.legend()

    # save the plots
    if savefile:
        dtstamp = str(dt.now()).replace(' ', '_').replace(':','-').replace('.','_')
        fig.savefig(f'images/changing_rho2_n_{n}_r2_{r2}_sigma2_{sigma2}_gamma_{gamma}_strongfeature_{strong_feature}_strongfeatureratio_{strong_feature_ratio}_alpha_{alpha}_source_{source_condition}_final_{dtstamp}.pdf', format = 'pdf')

    return







if __name__ == '__main__':

    display_risks_rho_2(n = 100,
                            gamma = 2,
                            r2 = 2,
                            sigma2 = 1,

                            strong_feature_ratio = 1/2,
                            start_eval_ratio = 0.5,
                            end_eval_ratio = 5,
                            change_strong_feature = True,

                            source_condition = 'id',

                            empir = 'gl',
                            alpha = 0.25,

                            include_predictions = False,
                            include_gd = False,
                            include_md = False,
                            include_md_empirical = False,
                            include_best_achievable = True,
                            include_best_achievable_empirical = True,

                            snr_estimation = list(np.linspace(0.1,1,20))+list(np.linspace(1,10,20)),
                            crossval_param = 100,

                            savefile = False,
                            )
