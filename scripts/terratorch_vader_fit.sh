#!/bin/sh
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=04:00:00

source ~/.bashrc
conda activate terramind

terratorch fit -c config/arctic_sea_ice/default.yaml --custom-modules-path="."