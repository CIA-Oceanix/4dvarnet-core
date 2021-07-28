params = {
    'data_dir'        : '/gpfsscratch/rech/nlu/commun/large',
    'dir_save'        : '/gpfsscratch/rech/nlu/commun/large/results_maxime',

    'iter_update'     : [0, 20, 40, 60, 100, 150, 800],  # [0,2,4,6,9,15]
    'nb_grad_update'  : [5, 5, 10, 10, 15, 15, 20, 20, 20],  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
    'lr_update'       : [1e-3, 1e-4, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7],
    'k_batch'         : 1,
    'n_grad'          : 5,
    'dim_grad_solver' : 10,
    'dropout'         : 0.25,
    'dT'              : 5, ## Time window of each space-time patch
    'dx'              : 1,   ## subsampling step if > 1
    'W'               : 200, # width/height of each space-time patch
    'shape_state'       : [10, 200, 200],
    'dW'              : 3,
    'dW2'             : 1,
    'sS'              : 4,  # int(4/dx),
    'nbBlocks'        : 1,
    'Nbpatches'       : 1, #10#10#25 ## number of patches extracted from each time-step 

    # stochastic version
    'stochastic'      : True,
    'size_ensemble'   : 3,

    # animation maps 
    'animate'         : False,

    # NN architectures and optimization parameters
    'batch_size'      : 10, #16#4#4#8#12#8#256#
    'DimAE'           : 50, #10#10#50
    'dimGradSolver'   : 100, # dimension of the hidden state of the LSTM cell

    'alpha_MSE'       : 0.1,
    'alpha_Proj'      : 0.5,
    'alpha_SR'        : 0.5,
    'alpha_LR'        : 0.5,  # 1e4

    # data generation
    'sigNoise'        : 0.,## additive noise standard deviation
    'flagSWOTData'    : True, #False ## use SWOT data or not
    'Nbpatches'       : 1, #10#10#25 ## number of patches extracted from each time-step 
    'rnd1'            : 0, ## random seed for patch extraction (space sam)
    'rnd2'            : 100, ## random seed for patch extraction
    'dwscale'         : 1,

    'betaX'           : 42.20436766972647, #None
    'betagX'          : 77.99700321505073, #None

    'UsePriodicBoundary' : False,  # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
    'InterpFlag'         : False, # True :> force reconstructed field to observed data after each gradient-based update
    'flagSWOTData'       : True,
    'automatic_optimization' : True,

}
