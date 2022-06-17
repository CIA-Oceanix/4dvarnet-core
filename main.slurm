#!/bin/bash -l

#SBATCH --account=yrf@gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --ntasks-per-node=1         # nombre de coeurs à réserver (un quart du noeud)
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
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
export HYDRA_FULL_ERROR=1

# run script from above
# config files stored in config_q --> examples: --config=q.nad_swot

# train the model
#srun python hydra_main.py xp=mbeaucha/xp_stoch/stoch_En4DVar_osse_gf file_paths=jz_osse entrypoint=run entrypoint.max_epochs=3
#srun python hydra_main.py xp=mbeaucha/xp_ose/ose_osse_nadswot_gf_wsst file_paths=jz_ose entrypoint=run entrypoint.max_epochs=120
#srun python hydra_main.py xp=mbeaucha/xp_ose/ose_gf_wsst file_paths=jz_ose entrypoint=run entrypoint.max_epochs=120
# test the model
#srun python hydra_main.py xp=mbeaucha/xp_ose/osse_nadswot_gf_wosst file_paths=jz_ose entrypoint=test entrypoint.ckpt_path="/gpfsstore/rech/yrf/commun/repo_xp_new/4DVarNet_ckpt_GF_4nadirs1swot.ckpt"
srun python hydra_main.py xp=mbeaucha/xp_ose/osse_nadswot_gf_wsst file_paths=jz_ose entrypoint=test entrypoint.ckpt_path="/gpfsstore/rech/yrf/uba22to/4DVarNet_ckpt_GF_4nadirs1swot_wsst.ckpt"

