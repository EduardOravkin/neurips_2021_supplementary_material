import numpy as np
from neurips_code.plot_changing_gamma_random_features import display_risks_gamma_rf




if __name__ == '__main__':


    # for smaller run time (but noisier plot) decrease n
    display_risks_gamma_rf(n = 3000,
                        n_test = 6000,
                        r2=5,
                        sigma2=1,

                        start_gamma = 4,
                        end_gamma = 20,
                        gamma_2 = 3, # n/d

                        regime='id',
                        ro=0.5,

                        strong_feature_ratio = 1/2,
                        strong_feature = 1,
                        weak_feature = 1/5,

                        source_condition = 'id',

                        fix_norm_of_theta = True,
                        fix_norm_of_x = True,

                        empir = 'gl',
                        alpha = 0.2,

                        include_best_achievable = True,

                        snr_estimation = list(np.linspace(0.1,1,10))+list(np.linspace(1,10,10)),
                        crossval_param = 10,

                        savefile = True,
                        )

