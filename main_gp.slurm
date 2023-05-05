#!/bin/bash -l

##SBATCH --account=yrf@a100
#SBATCH --account=yrf@v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --ntasks-per-node=1         # nombre de coeurs à réserver (un quart du noeud)
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
##SBATCH -C a100
#SBATCH --qos=qos_gpu-t3
## SBATCH --partition=gpu_p4
#SBATCH --output=log%j.out
#SBATCH --error=log%j.err

# activate conda env
#source activate $1

module purge

# chargement des modules
eval "$(conda shell.bash hook)"
conda activate 4dvarnet
export PYTHONPATH=${WORK}/pacnet:${WORK}/4dvarnet-core:${PYTHONPATH}
export HYDRA_FULL_ERROR=1

gp_dataset=$1
xp_gp=$2
max_epochs=$3

cd /gpfswork/rech/yrf/uba22to/4dvarnet-core

mkdir -p  oi/logs/${gp_dataset}

# train the model
srun python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/${xp_gp} file_paths=jz_gp entrypoint=run entrypoint.max_epochs=${max_epochs} | tee oi/logs/${gp_dataset}/log_${xp_gp}
#srun python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/${xp_gp} file_paths=jz_gp entrypoint=test entrypoint.ckpt_path="/gpfswork/rech/yrf/uba22to/4dvarnet-core/dashboard/oi_gp_spde_wolp_oi_loss_pow_2_diff/lightning_logs/version_2047690/checkpoints/model.ckpt"
