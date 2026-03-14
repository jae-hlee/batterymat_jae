#!/bin/bash
#SBATCH --job-name=bmat_single
#SBATCH --partition=a100
#SBATCH --account=kchoudh2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --output=logs/single_%j.out
#SBATCH --error=logs/single_%j.err

#Load environment
source /home/jlee859/miniconda3/etc/profile.d/conda.sh
conda activate bmat

echo "========== GPU Test Job =========="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# GPU check
echo "GPU Information:"
nvidia-smi
echo ""

# CUDA check
echo "PyTorch CUDA:"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"
echo ""

# Create directories
mkdir -p logs results

# Run analysis
echo "Running single material job..."
python /home/jlee859/batterymat/batterymat/jae_bmat/bmat_sign.py \
    --mode single \
    --jid JVASP-144791 \
    --output results/results.pkl \
    --log logs/bmat.log

echo ""
echo "Test completed: $(date)"
