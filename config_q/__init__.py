import pandas as pd

dT = 5
# Specify the dataset spatial bounds
dim_range = {
    'lat': slice(33, 43),
    'lon': slice(-65, -55),

}

# Specify the batch patch size
slice_win = {
    'time': 5,
    'lat': 200,
    'lon': 200,
}
# Specify the stride between two patches
strides = {
    'time': 1,
    'lat': 200,
    'lon': 200,
}

test_dates = pd.date_range('2013-01-03', "2013-01-27")
params = {
    'files_cfg' : dict(
                oi_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/oi/ssh_NATL60_4nadir.nc',
                oi_var='ssh_mod',
                obs_mask_path='/gpfsscratch/rech/yrf/ual82ir/4dvarnet-core/full_cal_obs.nc',
                obs_mask_var='nad_swot',
                gt_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc',
                gt_var='ssh',
        ),
    'splits':dict(
            train_slices=(slice('2012-10-01', "2012-11-20"), slice('2013-02-07', "2013-09-30")),
            test_slices=(slice(test_dates[0], test_dates[1]),),
            val_slices=(slice('2012-11-30', "2012-12-24"),),
    ),
    'test_dates': test_dates,
    'dataloading': 'new',
    'data_dir'        : '/gpfsscratch/rech/nlu/commun/large',
    'dir_save'        : '/gpfsscratch/rech/nlu/commun/large/results_maxime',

    'iter_update'     : [0, 20, 40, 60, 100, 150, 800],  # [0,2,4,6,9,15]
    'nb_grad_update'  : [5, 5, 10, 10, 15, 15, 20, 20, 20],  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
    'lr_update'       : [1e-3, 1e-4, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7],
    'k_batch'         : 1,
    'n_grad'          : 5,
    'dT'              : dT, ## Time window of each space-time patch
    'dx'              : 1,   ## subsampling step if > 1
    'W'               : 200, # width/height of each space-time patch
    'shape_state'       : [dT * 4, 200, 200],
    'shapeObs'       : [dT * 2, 200, 200],
    'dW'              : 3,
    'dW2'             : 1,
    'sS'              : 4,  # int(4/dx),
    'nbBlocks'        : 1,
    'Nbpatches'       : 1, #10#10#25 ## number of patches extracted from each time-step 

    # stochastic version
    'stochastic'      : False,

    # animation maps 
    'animate'         : False,

    # NN architectures and optimization parameters
    'batch_size'      : 2, #16#4#4#8#12#8#256#
    'DimAE'           : 50, #10#10#50
    'dim_grad_solver' : 150,
    'dropout'         : 0.25,
    'dropout_phi_r'   : 0.,

    'alpha_proj'      : 0.5,
    'alpha_sr'        : 0.5,
    'alpha_lr'        : 0.5,  # 1e4
    'alpha_mse_ssh'   : 10.,
    'alpha_mse_gssh'  : 1.,

    # data generation
    'sigNoise'        : 0.,## additive noise standard deviation
    'flagSWOTData'    : True, #False ## use SWOT data or not
    'Nbpatches'       : 1, #10#10#25 ## number of patches extracted from each time-step 
    'rnd1'            : 0, ## random seed for patch extraction (space sam)
    'rnd2'            : 100, ## random seed for patch extraction
    'dwscale'         : 1,

    'UsePriodicBoundary' : False,  # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
    'InterpFlag'         : False, # True :> force reconstructed field to observed data after each gradient-based update
    'flagSWOTData'       : True,
    'automatic_optimization' : True,

}
