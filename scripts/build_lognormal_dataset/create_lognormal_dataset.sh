#!/bin/bash

# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --array=0-99%100
#SBATCH --cpus-per-task=6 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=40G               # memory per node
#SBATCH --time=00-06:00         # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=
#SBATCH --output=jobout/%x_%A_%a.out

OUTPUT_DIR=/home/noedia/links/scratch/lognormal_dataset_dx500
SCRIPT_DIR=/home/noedia/links/projects/rrg-lplevass/noedia/wl_neurips/toy_notebooks
INPUT_DIR=/home/noedia/links/projects/rrg-lplevass/noedia/wl_neurips/training_data
source $HOME/wl_chall/bin/activate
cd $SCRIPT_DIR

python create_lognormal_dataset.py \
    --input_dir=$INPUT_DIR\
    --output_dir=$OUTPUT_DIR\
    --num_indep_sims=50\
    --nside=2048\
    --reso=2\
    --dx=500\
    --limber_approx=true\
    --num_patches=10\


