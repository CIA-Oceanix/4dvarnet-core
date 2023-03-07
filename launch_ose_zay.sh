#!/bin/bash

cd /gpfswork/rech/yrf/uba22to/4dvarnet-core

# Unet w/wosst

#  OSSE OI wosst
sbatch main_ose.slurm duacs_free_ose_gf_wosst 30 "../../ose/ckpt/4DVarNet_OI_OSSE_woSST.ckpt" false
#  OSSE OI wsst
sbatch main_ose.slurm duacs_free_ose_gf_wsst 30 "../../ose/ckpt/4DVarNet_OI_OSSE_wSST.ckpt" false

#  OSSE wDUACS wosst
sbatch main_ose.slurm duacs_anom_ose_gf_wosst 30 "../../ose/ckpt/4DVarNet_DUACS_anomaly_OSSE_woSST.ckpt" false
#  OSSE wDUACS wsst
sbatch main_ose.slurm duacs_anom_ose_gf_wsst 30 "../../ose/ckpt/4DVarNet_DUACS_anomaly_OSSE_wSST.ckpt" false

# fine-tuning OSSE OI wosst
sbatch main_ose.slurm duacs_free_ose_gf_wosst 30 "../../ose/ckpt/4DVarNet_OI_OSSE_woSST.ckpt" true
# fine-tuning OSSE OI wsst
sbatch main_ose.slurm duacs_free_ose_gf_wsst 30 "../../ose/ckpt/4DVarNet_OI_OSSE_wSST.ckpt" true

# fine-tuning OSSE wDUACS wosst
sbatch main_ose.slurm duacs_anom_ose_gf_wosst 30 "../../ose/ckpt/4DVarNet_DUACS_anomaly_OSSE_woSST.ckpt" true
# fine-tuning OSSE wDUACS wsst
sbatch main_ose.slurm duacs_anom_ose_gf_wsst 30 "../../ose/ckpt/4DVarNet_DUACS_anomaly_OSSE_wSST.ckpt" true

