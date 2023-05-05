#!/bin/bash -l

#SBATCH --account=yrf@v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --ntasks-per-node=1         # nombre de coeurs à réserver (un quart du noeud)
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-dev
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

cd /gpfswork/rech/yrf/uba22to/4dvarnet-core

srun launch_ose_ft.sh

