import numpy as np
import scipy.linalg as sc

from neurips_code.utils import *
from datetime import datetime as dt


def v_0(gamma = 2, rho_1 = 1, rho_2 = 1, psi_1 = 1/2):
    ''' Formula of v(0), the companion Stieltjes transform evaluated at 0, for the strong_weak_features model.'''

    return ((rho_1 + rho_2 - gamma*psi_1*rho_1 - gamma*(1-psi_1)*rho_2)+\
    np.sqrt( (rho_1 + rho_2 - gamma*psi_1*rho_1 - gamma*(1-psi_1)*rho_2)**2 - 4*(1-gamma)*rho_1*rho_2))/\
    (2*(gamma-1)*rho_1*rho_2)



def delta(gamma = 2, rho_1 = 1, rho_2 = 1, psi_1 = 1/2):
    ''' Formula for a helpful expression.'''

    return psi_1*rho_1**2/((1+rho_1*v_0(gamma, rho_1, rho_2, psi_1))**2) +\
             (1-psi_1)*rho_2**2/((1+rho_2*v_0(gamma, rho_1, rho_2, psi_1))**2)


def dv0_v02(gamma=2, rho_1 = 1, rho_2 = 1, psi_1 = 1/2):
    ''' Formula of v'(0)/v(0)^2 for the strong_weak_features model.'''

    return 1/(1 - gamma*delta(gamma, rho_1, rho_2, psi_1)*(v_0(gamma, rho_1, rho_2, psi_1)**2))



def bias_gd_strongweak(r2=1, gamma=2, rho_1 = 1, rho_2 = 1, psi_1 = 1/2, source_condition = 'id'):
    ''' The bias term of the risk of gradient descent for the strong_weak_features model.'''

    if source_condition == 'hard':
        return r2/(gamma) * ((gamma - 1)*dv0_v02(gamma, rho_1, rho_2, psi_1) - 1)

    elif source_condition == 'easy':
        return r2/(gamma*(v_0(gamma, rho_1, rho_2, psi_1)**2)) * (dv0_v02(gamma, rho_1, rho_2, psi_1) - 1)

    else:
        return r2/(gamma*v_0(gamma, rho_1, rho_2, psi_1))



def variance_gd_strongweak(sigma2=1, gamma=2, rho_1 = 1, rho_2 = 1, psi_1 = 1/2):
    ''' The variance term of the risk of gradient descent.'''

    return sigma2*(dv0_v02(gamma, rho_1, rho_2, psi_1) - 1)



def risk_gd_strongweak(r2=1, sigma2=1, gamma=2, rho_1 = 1, rho_2 = 1, psi_1 = 1/2, source_condition = 'id'):
    ''' The risk of gradient descent.'''

    return bias_gd_strongweak(r2, gamma, rho_1, rho_2, psi_1, source_condition) + variance_gd_strongweak(sigma2, gamma, rho_1, rho_2, psi_1)



def bias_md(c, m, r2=1, gamma=2):

    return np.trace(c.dot(m))*r2*(gamma - 1)/(gamma*c.shape[0])



def variance_md(sigma2=1, gamma=2):

    return sigma2/(gamma - 1)



def risk_md(c, m, r2=1, sigma2=1, gamma=2):

    return bias_md(c, m, r2, gamma) + variance_md(sigma2, gamma)
