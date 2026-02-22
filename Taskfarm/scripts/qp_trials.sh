#!/bin/bash
#SBATCH --job-name=qp_trials

set -euo pipefail

basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Our root directory

module purge
module load GCC/13.3.0
source "$basedir/.venv/bin/activate"
export PYTHONPATH="$basedir/code:$PYTHONPATH" # Set pythonpath so we can define the function below
# Config parser
get_yaml(){
    local config="$1"
    local key="$2"
    local default="${3:-}"
    python - "$config" "$key" "$default" << PY

import sys
import json
from pathlib import Path

conf_path = sys.argv[1]
key = sys.argv[2]
default = sys.argv[3]
# Try existing functions
try:
    from source.config import load_yaml, get_nested_data
    config = load_yaml(conf_path)
    val = get_nested_data(config, key, None)
except Exception:
    # Fallback simple parser
    import yaml
    config = yaml.safe_load(Path(conf_path).read_text())
    data = config
    for part in key.split('.'):
        # print(f"Key {part} found.")
        if not isinstance(data, dict) or part not in data:
            print(default); sys.exit(0)
        data = data[part]
    val = data

if val is None:
    print(default)
elif isinstance(val, list):
    for x in val:
        print(x)
else:
    print(val)
PY
}

UPDATED_CONFIG="$1"
Q_BLOCK_SIZE="$2"
VARS="$3"

# Define the constants for this RG flow
TYPE="QP" # Mode of RG workflow
VERSION="$(get_yaml "$UPDATED_CONFIG" "main.version")"  # Version for tracking changes and matrix used
N="$(get_yaml "$UPDATED_CONFIG" "rg_settings.samples")" # Total number of samples
NUM_RG_ITERS="$(get_yaml "$UPDATED_CONFIG" "rg_settings.steps")" # Number of RG steps
SEED="$(get_yaml "$UPDATED_CONFIG" "rg_settings.seed")" # Starting seed
METHOD="$(get_yaml "$UPDATED_CONFIG" "engine.method")" # Flag to determine whether to use analytic or numerical methods
EXPR="$(get_yaml "$UPDATED_CONFIG" "engine.expr")" # Flag to determine which expression to use
Q_NUM="$(get_yaml "$UPDATED_CONFIG" "parameter_settings.q.num")" # Number of q values for trials
VERSIONSTR="${VERSION}_${METHOD}_${EXPR}"
PREV_GEN_JOB="" # Placeholder var for holding previous job ID when setting up dependency

# Sort out block number for array jobs
Q_BLOCKS=$(( (Q_NUM + Q_BLOCK_SIZE - 1) / Q_BLOCK_SIZE )) # No. of array jobs
MAX_ARRAY_VAL=$((Q_BLOCKS - 1))
# MAX_PARALLEL=$((MAX_ARRAY_VAL / 10))
MAX_PARALLEL=10
ARRAY_STR="0-${MAX_ARRAY_VAL}%${MAX_PARALLEL}" # Array str for sbatch command



# Set up folders
joboutdir="$basedir/job_outputs/${VERSIONSTR}/$TYPE" # Where the output files will go
datadir="$joboutdir/data" # Where the data will live
codedir="$basedir/code" # Where the code lives
scriptsdir="$basedir/scripts" # Where all shell scripts live
logsdir="$basedir/job_logs/${VERSIONSTR}/$TYPE" # Where log files will go
mkdir -p "$logsdir" "$joboutdir" "$datadir" # Make them in case they aren't already there

out_file="$joboutdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.out"
err_file="$logsdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.err"
exec >"$out_file" # Redirect outputs to be within their own folders, together with the data they produce
exec 2>"$err_file" # Redirect error logs to be within their own folders for easy grouping

echo "Redirecting output logs to $out_file"
echo "Redirecting error logs to $err_file"

echo "======================================================"
echo "                    SLURM JOB INFO "
echo "------------------------------------------------------"
echo " Job Name            : $SLURM_JOB_NAME"
echo " Job ID              : $SLURM_JOB_ID"
echo " Submitted from      : $SLURM_SUBMIT_DIR"
echo " Type                : $TYPE"
echo " Current dir         : $(pwd)"
echo " Date of job         : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "======================================================"
echo ""
echo "======================================================"
echo "                  QP WORKFLOW CONFIG "
echo "------------------------------------------------------"
echo " Version             : $VERSIONSTR "
echo " Type                : $TYPE "
echo " Number of samples   : $N "
echo " Number of RG steps  : $NUM_RG_ITERS "
echo " Number of q values  : $Q_NUM "
echo " Number of q blocks  : $Q_BLOCKS "
echo " Q block size        : $Q_BLOCK_SIZE "
echo " Agg vars            : $VARS "
echo " Starting seed       : $SEED "
echo " Solver Method       : $METHOD "
echo " Expression          : $EXPR "
echo " Date of job         : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "======================================================"
echo ""


# Make sure the system recognises the python path to ensure relative imports proceed without issue
export PYTHONPATH="$codedir:$PYTHONPATH"
cd "$codedir"
SRC_DIR="$codedir/source" # This is where the actual code lives

# Gen and Agg scripts
GEN_SCRIPT="$SRC_DIR/qshe_data_gen.py"
AGG_SCRIPT="$SRC_DIR/qshe_data_agg.py"

# Send gen jobs
gen_job=$(sbatch --parsable \
    --job-name="qp_gen" \
    --array="$ARRAY_STR" \
    --output="$joboutdir/qp_gen_%A_%a.out" \
    --error="$logsdir/qp_gen_%A_%a.err" \
    --export=ALL,GEN_SCRIPT="$GEN_SCRIPT",UPDATED_CONFIG="$UPDATED_CONFIG",NUM_SAMPLES="$N",NUM_STEPS="$NUM_RG_ITERS",Q_BLOCK_SIZE="$Q_BLOCK_SIZE",SEED="$SEED",OUTPUT_DIR="$datadir" \
    << 'GEN_EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@30
#SBATCH --kill-on-invalid-dep=yes

basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Our root directory
codedir="$basedir/code" # Where the code lives

# Libraries needed
module purge
module load GCC/13.3.0 SciPy-bundle/2024.05

# Make sure the system recognises the python path to ensure relative imports proceed without issue
export PYTHONPATH="$codedir:$PYTHONPATH"
cd "$codedir"
source "$basedir/.venv/bin/activate"
SRC_DIR="$codedir/source" # This is where the actual code lives

set -euo pipefail
export RG_CONFIG="$UPDATED_CONFIG"
GEN_SEED=$(( SEED + (SLURM_ARRAY_TASK_ID * 100003) + 1))
PHI_SEED=$(( SEED + 123456 ))
Q_BLOCK="${SLURM_ARRAY_TASK_ID}"
python "$GEN_SCRIPT" \
    "$NUM_SAMPLES" \
    "$NUM_STEPS" \
    "$Q_BLOCK" \
    "$Q_BLOCK_SIZE" \
    "$PHI_SEED" \
    "$GEN_SEED" \
    "$OUTPUT_DIR"
GEN_EOF
)

# Send agg job dependent on gen job completing
agg_job=$(sbatch --parsable \
    --job-name="qp-agg" \
    --dependency=afterok:"$gen_job" \
    --output="$joboutdir/qp_agg_%A.out" \
    --error="$logsdir/qp_agg_%A.err" \
    --export=ALL,AGG_SCRIPT="$AGG_SCRIPT",UPDATED_CONFIG="$UPDATED_CONFIG",NUM_SAMPLES="$N",NUM_STEPS="$NUM_RG_ITERS",Q_BLOCK_SIZE="$Q_BLOCK_SIZE",OUTPUT_DIR="$datadir",VARS="$VARS" \
    << 'AGG_EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --kill-on-invalid-dep=yes

basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Our root directory
codedir="$basedir/code" # Where the code lives

# Libraries needed
module purge
module load GCC/13.3.0 SciPy-bundle/2024.05

# Make sure the system recognises the python path to ensure relative imports proceed without issue
export PYTHONPATH="$codedir:$PYTHONPATH"
cd "$codedir"
source "$basedir/.venv/bin/activate"
SRC_DIR="$codedir/source" # This is where the actual code lives

set -euo pipefail
export RG_CONFIG="$UPDATED_CONFIG"

python "$AGG_SCRIPT" \
    "$NUM_SAMPLES" \
    "$NUM_STEPS" \
    "$Q_BLOCK_SIZE" \
    "$OUTPUT_DIR" \
    "$VARS"
AGG_EOF
)

echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted qp_gen array $gen_job "
echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted qp_agg job $agg_job "
echo " Output dir: $joboutdir "
