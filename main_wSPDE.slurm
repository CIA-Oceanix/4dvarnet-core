#!/bin/bash -l

#SBATCH --account=yrf@gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4                # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --ntasks-per-node=4         # nombre de coeurs à réserver (un quart du noeud)
#SBATCH --cpus-per-task=10
#SBATCH --time=00:20:00
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
#conda activate 4dvarnet
conda activate spdenet2
export PYTHONPATH=${WORK}/4dvarnet-core:${PYTHONPATH}

# run script from above
# config files stored in config_q --> examples: --config=q.nad_swot

# train the model
srun python main.py --config=$1 --max_epochs=1 --progress_bar_refresh_rate=5 run
# fine-tuning with GF supervised training (nad+swot)
#srun python main.py --config=$1 --ckpt_path=checkpoints/GF_supervised.ckpt --max_epochs=30 --progress_bar_refresh_rate=5 run
# test with GF supervised training (nad+sst+swot)
#srun python main.py --config=$1 --ckpt_path=checkpoints/GF_supervised.ckpt --progress_bar_refresh_rate=5 test

