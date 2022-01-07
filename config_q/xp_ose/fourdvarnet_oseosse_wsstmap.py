from . import *
import yaml

params = yaml.safe_load("""
files_cfg:
  oi_path: /gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/oi_OSE_OSSE.nc
  oi_var: ssh
  obs_mask_path: /gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/data_OSE_OSSE.nc
  obs_mask_var: ssh
  gt_path: /gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/ref_ssh_OSE_OSSE.nc
  gt_var: ssh
  sst_path: /gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/ref_sst_OSE_OSSE.nc
  sst_var: sst
dataloading: with_sst
data_dir: /gpfsscratch/rech/nlu/commun/large
dir_save: /gpfsscratch/rech/nlu/commun/large/results_maxime
iter_update:
- 0
- 20
- 40
- 60
- 100
- 150
- 800
nb_grad_update:
- 5
- 5
- 10
- 10
- 15
- 15
- 20
- 20
- 20
lr_update:
- 0.001
- 0.0001
- 0.001
- 0.0001
- 0.0001
- 1.0e-05
- 1.0e-05
- 1.0e-06
- 1.0e-07
k_batch: 1
n_grad: 5
dT: 5
dx: 1
W: 200
resize_factor: 1
shapeData:
- 10
- 200
- 200
dW: 3
dW2: 1
sS: 4
nbBlocks: 1
Nbpatches: 1
stochastic: false
size_ensemble: 3
animate: false
supervised: false
batch_size: 2
DimAE: 25
dim_grad_solver: 70
dropout: 0.25
dropout_phi_r: 0.0
alpha_proj: 0.5
alpha_sr: 0.5
alpha_lr: 0.5
alpha_mse_ssh: 10.0
alpha_mse_gssh: 1.0
sigNoise: 0.0
flagSWOTData: true
rnd1: 0
rnd2: 100
dwscale: 1
UsePriodicBoundary: false
InterpFlag: false
automatic_optimization: true
""")
