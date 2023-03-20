#!/bin/bash
#SBATCH --job-name=gpu_mono          # nom du job
# Il est possible d'utiliser une autre partition que celle par défaut
# en activant l'une des 5 directives suivantes :
##SBATCH -C v100-16g                 # decommenter pour reserver uniquement des GPU V100 16 Go
SBATCH -C v100-32g                 # decommenter pour reserver uniquement des GPU V100 32 Go
##SBATCH --partition=gpu_p2          # decommenter pour la partition gpu_p2 (GPU V100 32 Go)
##SBATCH --partition=gpu_p4          # decommenter pour la partition gpu_p4 (GPU A100 40 Go)
##SBATCH -C a100                     # decommenter pour la partition gpu_p5 (GPU A100 80 Go)
# Ici, reservation de 10 CPU (pour 1 tache) et d'un GPU sur un seul noeud :
SBATCH --nodes=1                    # on demande un noeud
SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
# Le nombre de CPU par tache doit etre adapte en fonction de la partition utilisee. Sachant
# qu'ici on ne reserve qu'un seul GPU (soit 1/4 ou 1/8 des GPU du noeud suivant la partition),
# l'ideal est de reserver 1/4 ou 1/8 des CPU du noeud pour la seule tache:
SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
##SBATCH --cpus-per-task=3           # nombre de CPU par tache pour gpu_p2 (1/8 des CPU du noeud 8-GPU)
##SBATCH --cpus-per-task=6           # nombre de CPU par tache pour gpu_p4 (1/8 des CPU du noeud 8-GPU)
##SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p5 (1/8 des CPU du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
SBATCH --hint=nomultithread         # hyperthreading desactive
SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=gpu_mono%j.out      # nom du fichier de sortie
#SBATCH --error=gpu_mono%j.out       # nom du fichier d'erreur (ici commun avec la sortie)
 
# Nettoyage des modules charges en interactif et herites par defaut
#module purge
 
# Decommenter la commande module suivante si vous utilisez la partition "gpu_p5"
# pour avoir acces aux modules compatibles avec cette partition
#module load cpuarch/amd
 
# Chargement des modules
#module load ...
 
# Echo des commandes lancees
#set -x
 
# Pour la partition "gpu_p5", le code doit etre compile avec les modules compatibles
# Execution du code
./test_4dvarnet.sh
