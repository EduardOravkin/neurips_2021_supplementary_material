import numpy as np
from neurips_code.plot_changing_gamma import display_risks_gamma




if __name__ == '__main__':


    # for smaller run time (but noisier plot) decrease n
    display_risks_gamma(n = 2000,
                        r2=1,
                        sigma2=1,

                        start_gamma = 1.4,
                        end_gamma = 4,

                        regime='autoregressive',
                        ro=0.5,

                        source_condition = 'hard',

                        empir = 'gl',
                        alpha = 0.1,

                        include_gd = True,
                        include_md = False,
                        include_md_empirical = False,
                        include_best_achievable = True,
                        include_best_achievable_empirical = True,
                        include_best_achievable_empirical_phi = True,
                        include_best =  False,

                        snr_estimation = list(np.linspace(0.1,1,10))+list(np.linspace(1,10,10)),
                        crossval_param = 10,

                        savefile = True,
                        )
