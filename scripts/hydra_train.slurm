#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --account=yrf@gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:30:00
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=log%j.out
#SBATCH --error=log%j.err

# activate conda env
#source activate $1

module purge

# chargement des modules
eval "$(conda shell.bash hook)"
conda activate 4dvarnet --stack
export PYTHONPATH=${WORK}/4dvarnet-core:${PYTHONPATH}
export HYDRA_FULL_ERROR=1


# srun python hydra_main.py xp=$1 entrypoint=train  file_paths=jz
python hydra_main.py xp=$1 entrypoint=train  file_paths=jz



