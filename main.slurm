#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2
#SBATCH --account=yrf@gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH -C v100-16g
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=log%j.out
#SBATCH --error=log%j.err

# activate conda env
#source activate $1

module purge

# chargement des modules
eval "$(conda shell.bash hook)"
conda activate 4dvarnet
export PYTHONPATH=${WORK}/4dvarnet-core:${PYTHONPATH}

# run script from above
#srun python main.py --config=stochastic --dataloading=new --max_epochs=3 --progress_bar_refresh_rate=5 run
srun python main.py --dataloading=new --max_epochs=3 --progress_bar_refresh_rate=5 run

