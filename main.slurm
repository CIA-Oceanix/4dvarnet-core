#!/bin/bash -l

#SBATCH --account=yrf@gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4                # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --ntasks-per-node=4         # nombre de coeurs à réserver (un quart du noeud)
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
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
export PYTHONPATH=${WORK}/pacnet:${WORK}/4dvarnet-core:${PYTHONPATH}

# run script from above
# config files stored in config_q --> examples: --config=q.nad_swot

# train the model
#srun python main.py --config=$1 --max_epochs=30 --progress_bar_refresh_rate=5 run
#srun python hydra_main.py xp=ose_gf_wsst file_paths=jz_ose entrypoint=run entrypoint.max_epochs=1
srun python hydra_main.py xp=stoch_En4DVar_osse_gf file_paths=jz_osse entrypoint=run entrypoint.max_epochs=3
#srun python hydra_main.py xp=sla_natl file_paths=jz_osse entrypoint=run entrypoint.max_epochs=50
# test the model
#srun python main.py --config=$1 --ckpt_path=checkpoints/GF_OSE_OSSE.ckpt --progress_bar_refresh_rate=5 test
#srun python hydra_main.py xp=sla_natl file_paths=jz_osse entrypoint=test entrypoint.ckpt_path="lightning_logs/version_1011512/checkpoints/modelSLAInterpGF-Exp3-epoch\=05-val_loss\=0.16.ckpt"

