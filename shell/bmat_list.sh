#!/bin/bash
#SBATCH --job-name=bmat_list
#SBATCH --partition=a100
#SBATCH --account=kchoudh2
#SBATCH --array=1-43               # Set to number of lines in jid_list.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/list_%A_%03a.out
#SBATCH --error=logs/list_%A_%03a.err

# Load environment
source /home/jlee859/miniconda3/etc/profile.d/conda.sh
conda activate bmat

# Path to JID list file
JID_FILE="jid_list.txt"

# Get JID for this array task (line number = array task ID)
JID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${JID_FILE})

echo "=========================================="
echo "Array task ${SLURM_ARRAY_TASK_ID}"
echo "Processing: ${JID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"
echo "=========================================="
echo ""

# Create directories
mkdir -p logs results

# Initialize CSV with header if it doesn't exist (thread-safe)
if [ ! -f results/summary.csv ] || [ ! -s results/summary.csv ]; then
    python -c "
import fcntl
import os

csv_file = 'results/summary.csv'
try:
    with open(csv_file, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        # Check file size after acquiring lock
        f.seek(0, 2)  # Seek to end
        if f.tell() == 0:
            f.write('jid,formula,avg_voltage_V,max_voltage_V,min_voltage_V,grav_capacity_mAh_g,vol_capacity_mAh_cm3,status\n')
        fcntl.flock(f, fcntl.LOCK_UN)
except Exception as e:
    pass  # Another task may have created it
"
fi

# Process this material
python /home/jlee859/batterymat/batterymat/jae_bmat/bmat_sign.py \
    --mode single \
    --jid ${JID} \
    --output results/${JID}_result.pkl \
    --log logs/${JID}.log

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ ${JID} completed successfully"

    # Extract summary and append to master summary log
    python -c "
import pickle
import sys
import os
from datetime import datetime
import fcntl

# Get variables from shell
jid = '${JID}'
array_id = '${SLURM_ARRAY_TASK_ID}'

try:
    with open(f'results/{jid}_result.pkl', 'rb') as f:
        result = pickle.load(f)

    # Format summary line
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary_line = (
        f'{timestamp:20s} | '
        f'Task {array_id:>3s} | '
        f'{result[\"jid\"]:15s} | '
        f'{result[\"formula\"]:15s} | '
        f'V_avg={result[\"avg_voltage\"]:5.3f}V | '
        f'V_max={result[\"max_voltage\"]:5.3f}V | '
        f'V_min={result[\"min_voltage\"]:5.3f}V | '
        f'Cap_g={result[\"gravimetric_capacity\"]:7.2f} mAh/g | '
        f'Cap_v={result[\"volumetric_capacity\"]:7.2f} mAh/cm³'
    )

    with open('logs/summary_all.log', 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(summary_line + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)

    # Also create CSV entry
    csv_line = f'{result[\"jid\"]},{result[\"formula\"]},{result[\"avg_voltage\"]:.3f},{result[\"max_voltage\"]:.3f},{result[\"min_voltage\"]:.3f},{result[\"gravimetric_capacity\"]:.2f},{result[\"volumetric_capacity\"]:.2f},Success'

    with open('results/summary.csv', 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(csv_line + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)

    # Print to stdout
    print(f'\nSummary: V_avg={result[\"avg_voltage\"]:.3f}V, Cap={result[\"gravimetric_capacity\"]:.2f} mAh/g')

except Exception as e:
    print(f'Error extracting summary: {e}', file=sys.stderr)

    # Log failure
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    failure_line = f'{timestamp:20s} | Task {array_id:>3s} | {jid:15s} | {\"FAILED\":15s} | Error: {str(e)}'

    with open('logs/summary_all.log', 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(failure_line + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)

    with open('results/summary.csv', 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(f'{jid},ERROR,N/A,N/A,N/A,N/A,N/A,Failed\n')
        fcntl.flock(f, fcntl.LOCK_UN)
"

else
    echo "✗ ${JID} failed with exit code ${EXIT_CODE}"

    # Log failure
    python -c "
import os
from datetime import datetime
import fcntl

jid = '${JID}'
array_id = '${SLURM_ARRAY_TASK_ID}'
exit_code = '${EXIT_CODE}'

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
failure_line = f'{timestamp:20s} | Task {array_id:>3s} | {jid:15s} | {\"FAILED\":15s} | Exit code: {exit_code}'

with open('logs/summary_all.log', 'a') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(failure_line + '\n')
    fcntl.flock(f, fcntl.LOCK_UN)

with open('results/summary.csv', 'a') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(f'{jid},FAILED,N/A,N/A,N/A,N/A,N/A,Failed\n')
    fcntl.flock(f, fcntl.LOCK_UN)
"

fi

echo "Finished: $(date)"
echo "=========================================="
