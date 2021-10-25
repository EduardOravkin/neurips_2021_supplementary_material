import matplotlib.pyplot as plt
import scipy.linalg as sc
from sklearn.covariance import LedoitWolf, GraphicalLasso
from neurips_code.Regressor import LinearRegressor
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from numpy.random import normal
from functools import partial

import warnings


def sq_loss(y_pred, y):
    return np.linalg.norm(y_pred - y)**2


def generate_m(c,n,d,source_condition = 'id'):

    if source_condition == 'id':
        m = np.eye(c.shape[0])

    elif source_condition == 'easy':
        m = c

    elif source_condition == 'hard':
        m = np.linalg.inv(c)

    else:
        return generate_c(ro = 0.5,
                            regime = source_condition,
                            n = n,
                            d = d,
                            strong_feature = 1,
                            strong_feature_ratio = 1/2,
                            weak_feature = 1/5)

    return m






def generate_true_parameter(n=200, d = 600, r2=5, m = None):

    if m is None:
        m = np.eye(d)

    assert (m.shape[0] == d) & (m.shape[1] == d)

    w_star = np.random.multivariate_normal(np.zeros(d), r2/d*m)

    return w_star








def generate_c(ro=0.25,
              regime='id',
              n=200,
              d=600,
              strong_feature = 1,
              strong_feature_ratio = 1/2,
              weak_feature = 1):

    c = np.eye(d)

    if regime == 'id':
        pass

    elif regime == 'autoregressive':

        for i in range(d):
            for j in range(d):

                c[i,j] = ro**(abs(i-j))

    elif regime == 'strong_weak_features':

        s_1 = np.ones(int(d*strong_feature_ratio))*strong_feature
        s_2 = np.ones(d-int(d*strong_feature_ratio))*weak_feature

        c = np.diag(np.concatenate((s_1,s_2)))

    elif regime == 'exponential':

        s = np.linspace(0,1,d+1,endpoint = False)[1:]
        quantile = - np.log(1-s) # quantile function of the standard exponential distribution

        c = np.diag(quantile)

    else:
        raise AssertionError('wrong regime of covariance matrices')

    return c










def generate_c_empir(X,empir, alpha = 0.25):

    if empir == 'basic':
        c_e = X.transpose().dot(X)/len(X)

    elif empir == 'lw':

        lw = LedoitWolf(assume_centered = True).fit(X)
        c_e = lw.covariance_

    elif empir == 'gl':
        gl = GraphicalLasso(assume_centered = True, alpha = alpha, tol = 1e-4).fit(X)
        c_e = gl.covariance_

    else:
        raise AssertionError('specify regime of empirical approximation')

    return c_e






def generate_data(w_star,
                 c,
                 n = 200,
                 d = 600,
                 sigma2=1,
                 fix_norm_of_x = False):

    assert len(w_star) == d, 'dimensions error'

    # generate features
    X = np.zeros((n,d))

    X = np.random.multivariate_normal(mean = np.zeros(d),cov = c, size = n)

    if fix_norm_of_x:
        X = X*np.sqrt(d)/np.linalg.norm(X, axis = 1)[:,None]
        print(f'norm of 17th data point is {np.linalg.norm(X[17])}')

    # print warning if X is not on the sphere
    if (any( abs(np.linalg.norm(X, axis = 1) - np.sqrt(d)) > 1e-5)):
        warnings.warn('Warning, norms of datapoints are not sqrt(d)')

    # generate_noise
    xi = np.random.multivariate_normal(np.zeros(n),sigma2*np.eye(n))

    # generate response
    y = X.dot(w_star) + xi

    return X, y, xi




def calculate_risk(w_star,c,w=0):
    return (w-w_star).dot(c.dot(w-w_star))


def calculate_risk_rf(a, w_star, c, cov_z, cov_zx):
    return a.dot(cov_z.dot(a)) + w_star.dot(c.dot(w_star)) - 2*a.dot(cov_zx.dot(w_star))




def compute_best_achievable_interpolator(X, y, c, m, snr, crossval_param = 100):

    ''' If snr is passed as a list then for each entry in snr, this function splits
        X, y into a train-test split crossval_param times and calculates the
        average crossvalidated error on for the given entry in snr. The
        average of the first three entries which minimize the crossvalidated error
        is chosed as an estimate of the signal-to-noise ratio.'''

    c_inv = np.linalg.inv(c)
    d = X.shape[1]
    n = X.shape[0]


    if type(snr) == np.ndarray or type(snr) == list:

        # initialize dataframe where we save results
        df = pd.DataFrame([], columns = ['mu', 'error'])

        for mu in snr:

            error_crossvalidated = 0

            for j in range(crossval_param):

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) # random train test split

                n_train  = X_train.shape[0]
                n_test = X_test.shape[0]

                # calculate the best_achievable interpolator according to formula in paper
                auxi_matrix_train = np.linalg.inv(np.eye(n_train) + (mu/d)*X_train.dot(m.dot(X_train.T)))
                auxi_matrix_train_2 = ((mu/d)*m.dot(X_train.T) + (c_inv.dot(X_train.T)).dot( np.linalg.inv(X_train.dot(c_inv.dot(X_train.T))) ))
                w_e_train = auxi_matrix_train_2.dot(auxi_matrix_train.dot(y_train))

                y_test_pred = X_test.dot(w_e_train)

                error_crossvalidated += (np.linalg.norm(y_test - y_test_pred)**2)/n_test

            error_crossvalidated = error_crossvalidated/crossval_param

            df = df.append(pd.DataFrame(np.array([[mu, error_crossvalidated]]), columns = ['mu', 'error']))

        df = df.sort_values('error', ascending = True)

        snr = np.mean(df['mu'].iloc[:3].values)

    # calculate the best_achievable interpolator according to formula in paper
    auxi_matrix = np.linalg.inv(np.eye(n) + (snr/d)*X.dot(m.dot(X.T)))
    auxi_matrix_2 = ((snr/d)*m.dot(X.T) + (c_inv.dot(X.T)).dot( np.linalg.inv(X.dot(c_inv.dot(X.T))) ))
    w_e = auxi_matrix_2.dot(auxi_matrix.dot(y))


    return w_e






def compute_best_achievable_interpolator_multi(X, y, c, m, snr, crossval_param = 100):

    ''' If snr is passed as a list then for each entry in snr, this function splits
        X, y into a train-test split crossval_param times and calculates the
        average crossvalidated error on for the given entry in snr. The
        average of the first three entries which minimize the crossvalidated error
        is chosed as an estimate of the signal-to-noise ratio.'''

    c_inv = np.linalg.inv(c)
    d = X.shape[1]
    n = X.shape[0]


    def search_snr(mu):

        crossval_param = 10
        error_crossvalidated = 0

        for j in range(crossval_param):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) # random train test split

            n_train  = X_train.shape[0]
            n_test = X_test.shape[0]

            # calculate the best_achievable interpolator according to formula in paper
            auxi_matrix_train = np.linalg.inv(np.eye(n_train) + (mu/d)*X_train.dot(m.dot(X_train.T)))
            auxi_matrix_train_2 = ((mu/d)*m.dot(X_train.T) + (c_inv.dot(X_train.T)).dot( np.linalg.inv(X_train.dot(c_inv.dot(X_train.T))) ))
            w_e_train = auxi_matrix_train_2.dot(auxi_matrix_train.dot(y_train))

            y_test_pred = X_test.dot(w_e_train)
            error_crossvalidated += (np.linalg.norm(y_test - y_test_pred)**2)/n_test

        error_crossvalidated = error_crossvalidated/crossval_param
        pd.DataFrame(np.array([[mu, error_crossvalidated]]), columns = ['mu', 'error'])
        pd.to_csv(f'mu_{mu}')


    if type(snr) == np.ndarray or type(snr) == list:

        pool = Pool()
        pool.map(search_snr, snr)

        # initialize dataframe where we save results
        df = pd.DataFrame([], columns = ['mu', 'error'])
        for mu in snr:
            df_temp = pd.read_csv(f'mu_{mu}')
            df = df.append(df_temp)

        df = df.sort_values('error', ascending = True)
        snr = np.mean(df['mu'].iloc[:3].values)

    # calculate the best_achievable interpolator according to formula in paper
    auxi_matrix = np.linalg.inv(np.eye(n) + (snr/d)*X.dot(m.dot(X.T)))
    auxi_matrix_2 = ((snr/d)*m.dot(X.T) + (c_inv.dot(X.T)).dot( np.linalg.inv(X.dot(c_inv.dot(X.T))) ))
    w_e = auxi_matrix_2.dot(auxi_matrix.dot(y))


    return w_e






def compute_best_achievable_interpolator_rf(X, Z, y, cov_z, cov_zx, m, snr, crossval_param):

    ''' If snr is passed as a list then for each entry in snr, this function splits
        X, y into a train-test split crossval_param times and calculates the
        average crossvalidated error on for the given entry in snr. The
        average of the first three entries which minimize the crossvalidated error
        is chosed as an estimate of the signal-to-noise ratio.'''

    d = X.shape[1]
    n = X.shape[0]
    N = Z.shape[1]

    # calculate the best_achievable interpolator according to formula in paper
    m_1 = np.linalg.inv(cov_z)


    if type(snr) == np.ndarray or type(snr) == list:

        # initialize dataframe where we save results
        df = pd.DataFrame([], columns = ['mu', 'error'])

        for mu in snr:

            error_crossvalidated = 0

            for j in range(crossval_param):

                X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, Z, test_size=0.1) # random train test split

                n_train  = X_train.shape[0]
                n_test = X_test.shape[0]

                # calculate the best_achievable interpolator according to formula in paper

                m_21_train = cov_zx.dot(m.dot(X_train.T))
                m_22_train = Z_train.T.dot( np.linalg.inv(Z_train.dot(m_1.dot(Z_train.T))) )
                m_23_train = (d/mu)*np.eye(n_train) + X_train.dot(m.dot(X_train.T)) - Z_train.dot(m_1.dot(cov_zx.dot(m.dot(X_train.T))))

                m_2_train = m_21_train + m_22_train.dot(m_23_train)

                m_3_train = np.linalg.inv( (d/mu)*np.eye(n_train) + X_train.dot(m.dot(X_train.T)) )

                w_e_train = m_1.dot(m_2_train.dot(m_3_train.dot(y_train)))

                y_test_pred = Z_test.dot(w_e_train)

                error_crossvalidated += (np.linalg.norm(y_test - y_test_pred)**2)/n_test

            error_crossvalidated = error_crossvalidated/crossval_param

            df = df.append(pd.DataFrame(np.array([[mu, error_crossvalidated]]), columns = ['mu', 'error']))

        df = df.sort_values('error', ascending = True)

        snr = np.mean(df['mu'].iloc[:3].values)


    m_21 = cov_zx.dot(m.dot(X.T))
    m_22 = Z.T.dot( np.linalg.inv(Z.dot(m_1.dot(Z.T))) )
    m_23 = (d/snr)*np.eye(n) + X.dot(m.dot(X.T)) - Z.dot(m_1.dot(cov_zx.dot(m.dot(X.T))))

    m_2 = m_21 + m_22.dot(m_23)

    m_3 = np.linalg.inv( (d/snr)*np.eye(n) + X.dot(m.dot(X.T)) )

    w = m_1.dot(m_2.dot(m_3.dot(y)))

    return w
