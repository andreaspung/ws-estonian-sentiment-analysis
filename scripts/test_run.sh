#!/bin/bash

#The name of the job is test_job
#SBATCH -J "Wrench-test"

#The job requires 1 compute node
#SBATCH -N 1

#The job requires 8 task per node
#SBATCH --ntasks-per-node=8

#The maximum walltime of the job is 96 hours
#SBATCH -t 96:00:00

#SBATCH --mem=32G

#If you keep the next two lines, you will get an e-mail notification
#whenever something happens to your job (it starts running, completes or fails)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andreas.pung@ut.ee

#Keep this line if you need a GPU for your job
#SBATCH --partition=gpu

#Indicates that you need one GPU node
#SBATCH --gres=gpu:tesla:1

#Commands to execute go below

#Load Python
module load miniconda3/4.8.2
source activate wrench2

export PYTHONPATH="${PYTHONPATH}:/gpfs/space/home/citius/thesis/wrench"
echo "$CONDA_PREFIX"

python --version

python apply_model.py
