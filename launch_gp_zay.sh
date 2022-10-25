#!/bin/bash

gp_dataset=$1
max_epochs=$2

cd /gpfswork/rech/yrf/uba22to/4dvarnet-core

mkdir -p  oi/logs/${gp_dataset}

#sbatch main_gp.slurm ${gp_dataset} oi_gp_ref 1
sbatch main_gp.slurm ${gp_dataset} oi_gp_ref_gd 1
sbatch main_gp.slurm ${gp_dataset} oi_gp_unet_mseloss 10
sbatch main_gp.slurm ${gp_dataset} oi_gp_unet_oiloss 10
sbatch main_gp.slurm ${gp_dataset} oi_gp_unet_mseoiloss 10
sbatch main_gp.slurm ${gp_dataset} oi_gp_spde_wolp_mseloss ${max_epochs}
sbatch main_gp.slurm ${gp_dataset} oi_gp_spde_wolp_oiloss ${max_epochs}
sbatch main_gp.slurm ${gp_dataset} oi_gp_spde_wolp_mseoiloss ${max_epochs}
sbatch main_gp.slurm ${gp_dataset} oi_gp_4dvarnet_mseloss ${max_epochs}
sbatch main_gp.slurm ${gp_dataset} oi_gp_4dvarnet_oiloss ${max_epochs}
sbatch main_gp.slurm ${gp_dataset} oi_gp_4dvarnet_mseoiloss ${max_epochs}
sbatch main_gp.slurm ${gp_dataset} oi_gp_4dvarnet_fp_mseloss ${max_epochs}
sbatch main_gp.slurm ${gp_dataset} oi_gp_4dvarnet_fp_oiloss ${max_epochs}
sbatch main_gp.slurm ${gp_dataset} oi_gp_4dvarnet_fp_mseoiloss ${max_epochs}
sbatch main_gp.slurm ${gp_dataset} oi_gp_4dvarnet_wphi_linear_mseloss ${max_epochs}
sbatch main_gp.slurm ${gp_dataset} oi_gp_4dvarnet_wphi_linear_oiloss ${max_epochs}
sbatch main_gp.slurm ${gp_dataset} oi_gp_4dvarnet_wphi_linear_mseoiloss ${max_epochs}

sbatch main_gp.slurm ${gp_dataset} oi_gp_4dvarnet_mseloss_test_grad_lstm ${max_epochs}

