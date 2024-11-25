#!/bin/bash
# 
#SBATCH -p gpu # partition (queue)
#SBATCH --gpus=a100:1
#SBATCH -c 1 # number of cores
#SBATCH --mem 79G # memory pool for all cores
#SBATCH -t 29-00:00 # time (D-HH:MM)
#SBATCH --job-name=gnfrmr
#SBATCH -o /home/cranneyc/geneformer/scripts/slurmOutputs/slurm.%j.out # STDOUT
#SBATCH -e /home/cranneyc/geneformer/scripts/slurmOutputs/slurm.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=caleb.cranney@cshs.org


echo "geneformer start"
eval "$(conda shell.bash hook)"
export CUDA_LAUNCH_BLOCKING=1
conda activate geneformer
cd /home/cranneyc/geneformer/scripts
python 1_train_model.py
