#!/bin/bash -l

#SBATCH --account=yrf@a100
##SBATCH --account=yrf@v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --ntasks-per-node=1         # nombre de coeurs à réserver (un quart du noeud)
#SBATCH --cpus-per-task=40
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread
##SBATCH -C v100-32g
#SBATCH -C a100
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

xp_ose=$1
max_epochs=$2
ckpt_path=$3
fine_tuning=$4

cd /gpfswork/rech/yrf/uba22to/4dvarnet-core


# train/test the model
if [ "$ckpt_path" = "" ] ; then
  srun python hydra_main.py xp=mbeaucha/xp_ose/${xp_ose} file_paths=jz_ose entrypoint=run entrypoint.max_epochs=${max_epochs}
else
  if [ "$fine_tuning" = true ] ; then
    srun python hydra_main.py xp=mbeaucha/xp_ose/${xp_ose} file_paths=jz_ose entrypoint=run entrypoint.max_epochs=${max_epochs} +entrypoint.ckpt_path=${ckpt_path}
  else
    srun python hydra_main.py xp=mbeaucha/xp_ose/${xp_ose} file_paths=jz_ose entrypoint=test +entrypoint.ckpt_path=${ckpt_path}
  fi
fi

