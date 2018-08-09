#!/bin/sh
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=96GB
#SBATCH --job-name="clevr"
#SBATCH --output=clevr.out
#SBATCH --error=clevr.error

julia train.jl
