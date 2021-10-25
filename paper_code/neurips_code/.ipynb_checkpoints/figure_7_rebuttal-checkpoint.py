import numpy as np
from neurips_code.plot_covariance_changing_gamma import display_cov_errors_gamma


# This was meant to be figure 3 but coparison of covariance matrices.

if __name__ == '__main__':


    # for smaller run time (but noisier plot) decrease n
    display_cov_errors_gamma(n = 2000,
                        r2=1,
                        sigma2=1,

                        start_gamma = 1.4,
                        end_gamma = 4,

                        regime='autoregressive',
                        ro=0.5,

                        ord = 2,

                        empir = 'gl',
                        alpha = 0.1,

                        savefile = True,
                        )
