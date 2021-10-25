import numpy as np
from neurips_code.plot_changing_rho2 import display_risks_rho_2
import time



if __name__ == '__main__':

    t1 = time.time()

    # for smaller run time (but noisier plot) decrease n
    display_risks_rho_2(n = 3000,
                        gamma = 2,
                        r2 = 1,
                        sigma2 = 1,

                        strong_feature_ratio = 1/2,
                        start_eval_ratio = 5,
                        end_eval_ratio = 100,
                        change_strong_feature = True,

                        source_condition = 'id',

                        empir = 'gl',
                        alpha = 0.25,

                        savefile = True,
                        include_predictions = True,

                        include_gd = True,
                        include_md = True,
                        include_md_empirical = False,
                        include_best_achievable = True,
                        include_best_achievable_empirical = True,

                        snr_estimation = list(np.linspace(0.1,1,10))+list(np.linspace(1,10,10)),
                        crossval_param = 10,
                        )

    t2 = time.time()
    print(t2-t1)
