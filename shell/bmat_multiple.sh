#!/bin/bash
#SBATCH --job-name=bmat_multiple
#SBATCH --partition=a100
#SBATCH --account=kchoudh2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/multiple_%j.out
#SBATCH --error=logs/multiple_%j.err

# Load environment
source /home/jlee859/miniconda3/etc/profile.d/conda.sh
conda activate bmat

# Print environment info
echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="
echo ""

# Create directories
mkdir -p logs results

# List of JIDs to process
JIDS=(
    "JVASP-119272"
    "JVASP-118981"
    "JVASP-118985"
    "JVASP-118986"
    "JVASP-120421"
    "JVASP-111889"
    "JVASP-121805"
    "JVASP-121905"
    "JVASP-122339"
)

# Process each material
for JID in "${JIDS[@]}"; do
    echo "=========================================="
    echo "Processing: ${JID}"
    echo "Time: $(date)"
    echo "=========================================="
    
    python /home/jlee859/batterymat/batterymat/jae_bmat/bmat_sign.py \
        --mode single \
        --jid ${JID} \
        --output results/${JID}_result.pkl \
        --log logs/${JID}.log
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✓ ${JID} completed successfully"
    else
        echo "✗ ${JID} failed"
    fi
    echo ""
done

echo "=========================================="
echo "All materials processed: $(date)"
echo "=========================================="
