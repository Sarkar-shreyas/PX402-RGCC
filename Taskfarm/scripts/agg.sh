#!/bin/bash
#SBATCH --job-name=temp_agg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --kill-on-invalid-dep=yes

UPDATED_CONFIG="$1"
Q_BLOCK_SIZE="$4"
VARS="$6"
VERSIONSTR="$5"
NUM_SAMPLES="$2"
NUM_STEPS="$3"
TYPE="QP"

basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Our root directory
codedir="$basedir/code" # Where the code lives
joboutdir="$basedir/job_outputs/${VERSIONSTR}/$TYPE" # Where the output files will go
datadir="$joboutdir/data" # Where the data will live
# Libraries needed
module purge
module load GCC/13.3.0 SciPy-bundle/2024.05

# Make sure the system recognises the python path to ensure relative imports proceed without issue
export PYTHONPATH="$codedir:$PYTHONPATH"
cd "$codedir"
source "$basedir/.venv/bin/activate"
SRC_DIR="$codedir/source" # This is where the actual code lives
AGG_SCRIPT="$SRC_DIR/qshe_data_agg.py"

set -euo pipefail
export RG_CONFIG="$UPDATED_CONFIG"

python "$AGG_SCRIPT" \
    "$NUM_SAMPLES" \
    "$NUM_STEPS" \
    "$Q_BLOCK_SIZE" \
    "$datadir" \
    "$VARS"
