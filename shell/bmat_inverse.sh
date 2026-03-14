#!/bin/bash
#SBATCH --job-name=alignn_ibmat
#SBATCH --partition=a100
#SBATCH --account=kchoudh2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=alignn.out
#SBATCH --error=alignn.err

# Load environment
source /home/jlee859/miniconda3/etc/profile.d/conda.sh
conda activate bmat

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run your training command
python /home/jlee859/alignn/alignn/train_alignn.py \
  --root_dir "/home/jlee859/scratchkchoudh2/jlee859/alignn_ibmat/Li/Li_250" \
  --config "/home/jlee859/scratchkchoudh2/jlee859/alignn_ibmat/Li/Li_250/config.json" \
  --target_key target \
  --id_key jid \
  --output_dir voltage

echo "Training complete"
