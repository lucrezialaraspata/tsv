#!/bin/bash

#SBATCH -A IscrC_GEMEX
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH --time=24:00:00
#SBATCH -N 1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=2
#SBATCH --mem=123000
#SBATCH --job-name=tsv-train
#SBATCH --out=output.log
#SBATCH --err=error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it

source .venv/bin/activate

# Top 10 best layers for Adapter over all datasets
srun -u python tsv_main.py  --model_name llama3.1-8B  --dataset_name tqa --most_likely 1 --use_local