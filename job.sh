#!/bin/sh
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --time=56:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=100GB
#SBATCH --job-name="clevr"
#SBATCH --output=clevr.out
#SBATCH --error=clevr.error
#SBATCH --nodelist=dy02
julia train.jl src/main.jl configs/config2.jl
