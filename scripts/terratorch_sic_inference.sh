#!/bin/sh
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH -o scripts/slurm/slurm-%j.out

source ~/.bashrc
conda activate terramind

terratorch test -c config/arctic_sea_ice/sic.yaml --custom_modules_path="/work/ka1176/caroline/gitlab/terramind-demo" --ckpt_path "output/wandb/sic/arctic-sea-ice/cxemzxfa/checkpoints/best-mca-epoch-epoch=13.ckpt" --predict_output_dir output/predictions/sic