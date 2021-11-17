#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --account=yrf@gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
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
conda activate 4dvarnet
export PYTHONPATH=${WORK}/4dvarnet-core:${PYTHONPATH}


srun python -m icassp_code_bis.main --max_epochs=250 --config=$1 --progress_bar_refresh_rate=5 train



