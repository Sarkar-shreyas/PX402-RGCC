#!/bin/bash
#SBATCH --job-name=shifted_rg_master

set -euo pipefail
basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Our root directory
# Libraries needed
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
SHIFT_INDEX="$2"
# Assign input vars
TYPE="EXP" # Mode of RG workflow
VERSION="$(get_yaml "$UPDATED_CONFIG" "main.version")" # Version for tracking changes and matrix used
N="$(get_yaml "$UPDATED_CONFIG" "rg_settings.samples")" # Total number of samples
NUM_RG_ITERS="$(get_yaml "$UPDATED_CONFIG" "rg_settings.steps")" # Number of RG steps
SEED="$(get_yaml "$UPDATED_CONFIG" "rg_settings.seed")" # Starting seed
FP_NUM=$(( NUM_RG_ITERS - 1 )) # Var to determine which FP distribution to use for generating the shifted dataset
METHOD="$(get_yaml "$UPDATED_CONFIG" "engine.method")" # Flag to determine whether to use analytic or numerical methods
EXPR="$(get_yaml "$UPDATED_CONFIG" "engine.expr")" # Flag to determine which expression to use
RESAMPLE="$(get_yaml "$UPDATED_CONFIG" "engine.resample")" # Flag to determine which type of resampler to use
SYMMETRISE="$(get_yaml "$UPDATED_CONFIG" "engine.symmetrise")" # Flag to determine whether to symmetrise data or not
VERSIONSTR="${VERSION}_${METHOD}_${EXPR}"
INITIAL=1 # Flag to generate starting distribution/histograms or not
EXISTING_T="" # Placeholder var to point to data file for non-initial RG steps
PREV_HIST_JOB="" # Placeholder var for holding previous job ID when setting up dependency

readarray -t SHIFTS < <(get_yaml "$UPDATED_CONFIG" "data_settings.shifts") # Store shifts list as a bash array
NUM_SHIFTS="${#SHIFTS[@]}" # Store the length of the shifts array

if (( NUM_SHIFTS == 0 )); then
    echo "No shifts found in config" >&2
    exit 2
fi

CURRENT_SHIFT="${SHIFTS[$SHIFT_INDEX]}" # Assigns the shift value to apply for this round.
export RG_CONFIG=$UPDATED_CONFIG

joboutdir="$basedir/job_outputs/${VERSIONSTR}/$TYPE/shift_${CURRENT_SHIFT}" # Where the output files will go
datadir="$joboutdir/data" # Where the data will live
scriptsdir="$basedir/scripts" # Where all shell scripts live
logsdir="$basedir/job_logs/${VERSIONSTR}/$TYPE/shift_${CURRENT_SHIFT}" # Where log files will go
mkdir -p "$logsdir" "$joboutdir" "$datadir" # Make them in case they aren't already there

out_file="$joboutdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.out"
err_file="$logsdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.err"
exec >"$out_file" # Redirect outputs to be within their own folders, together with the data they produce
exec 2>"$err_file" # Redirect error logs to be within their own folders for easy grouping

echo "Redirecting output logs to $out_file"
echo "Redirecting error logs to $err_file"

# General job information
echo "======================================================"
echo "                    SLURM JOB INFO "
echo "------------------------------------------------------"
echo " Job Name            : $SLURM_JOB_NAME"
echo " Job ID              : $SLURM_JOB_ID"
echo " Submitted from      : $SLURM_SUBMIT_DIR"
echo " Type                : $TYPE"
echo " Shift               : $CURRENT_SHIFT"
echo " Current dir         : $(pwd)"
echo " Date of job         : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "======================================================"
echo ""
echo "======================================================"
echo "                  RG WORKFLOW CONFIG "
echo "------------------------------------------------------"
echo " Version             : $VERSIONSTR "
echo " Type                : $TYPE "
echo " Number of samples   : $N "
echo " Number of RG steps  : $NUM_RG_ITERS "
echo " Starting seed       : $SEED "
echo " Prev FP step no.    : $FP_NUM "
echo " t' Method           : $METHOD "
echo " Expression          : $EXPR "
echo " Resample method     : $RESAMPLE "
echo " Symmetrising?       : $SYMMETRISE "
echo " Shifts to apply     : ${SHIFTS[*]} "
echo " Date of job         : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "======================================================"
echo ""
# Where the FP distribution we'll use to generate shifted data lives
FP_dist="$basedir/job_outputs/${VERSIONSTR}/FP/data/RG${FP_NUM}/hist/sym_z/sym_z_hist_RG${FP_NUM}.npz"


echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Generating initial data for shift ${CURRENT_SHIFT} "
# Run the job to generate initial shifted data
shift_job=$(sbatch --parsable \
    --output="$basedir/job_outputs/bootstrap/gen_shift_${CURRENT_SHIFT}_%A_%a.out" \
    --error="$basedir/job_logs/bootstrap/gen_shift_${CURRENT_SHIFT}_%A_%a.err" \
    "$scriptsdir/gen_shifted_data.sh" \
        "$UPDATED_CONFIG" "$VERSIONSTR" "$N" "$FP_dist" "Initial" "$SEED" "$CURRENT_SHIFT")

echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted shift job ${shift_job} for shift ${CURRENT_SHIFT}"
# For each RG, we queue data generation and then histogram jobs, dependencies ensure they run in sequence
for step in $(seq 0 $(( NUM_RG_ITERS - 1 ))); do
    echo "================================================================================================================="
    echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Proceeding with RG step $step of type $TYPE"
    laundereddir="$datadir/RG${step}/laundered"
    # If its the first step, we need to use the shifted data, later steps go back to using data from the previous RG step
    if [ "$step" -eq 0 ]; then
        INITIAL=0
        gen_job_dep="$shift_job"
        EXISTING_T="$datadir/Initial"
    else
        prev_step=$(( $step - 1 ))
        gen_job_dep="$PREV_HIST_JOB"
        EXISTING_T="$datadir/RG${prev_step}/laundered"
    fi

    gen_job=$(sbatch --parsable \
    --dependency=afterok:${gen_job_dep} \
    --output="$basedir/job_outputs/bootstrap/rg_gen_RG${step}_%A_%a.out" \
    --error="$basedir/job_logs/bootstrap/rg_gen_RG${step}_%A_%a.err" \
    "$scriptsdir/rg_gen_batch.sh" \
        "$UPDATED_CONFIG" "$TYPE" "$VERSIONSTR" "$N" "$step" "$SEED" "$METHOD" "$EXPR" "$INITIAL" "$EXISTING_T" "$CURRENT_SHIFT")

    echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted generation job for RG step $step : $gen_job (after ${gen_job_dep}) "

    echo "-----------------------------------------------------------------------------------------------------------------"
    hist_job=$(sbatch --parsable \
        --dependency=afterok:${gen_job} \
        --output="$basedir/job_outputs/bootstrap/rg_hist_RG${step}_%A.out" \
        --error="$basedir/job_logs/bootstrap/rg_hist_RG${step}_%A.err" \
        "$scriptsdir/rg_hist_manager.sh" \
        "$UPDATED_CONFIG" "$TYPE" "$VERSIONSTR" "$N" "$step" "$SEED" "$SYMMETRISE" "$CURRENT_SHIFT")

    echo "-----------------------------------------------------------------------------------------------------------------"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted histogram job for RG step $step  : $hist_job (after ${gen_job}) "
    # Keep track of the job ID for setting up dependencies in order
    PREV_HIST_JOB="$hist_job"
    INITIAL=0

done
echo "================================================================================================================="
echo " All ${NUM_RG_ITERS} RG jobs for shift ${CURRENT_SHIFT} submitted. Final dependency ends at job ${PREV_HIST_JOB} "
echo "================================================================================================================="

NEXT_SHIFT_INDEX=$(( SHIFT_INDEX + 1 ))
# Check if there are any shifts remaining, then send in jobs for the next
if (( NEXT_SHIFT_INDEX < NUM_SHIFTS )); then
    # Assign the next shift to run, and store any remaining
    next_shift="${SHIFTS[$NEXT_SHIFT_INDEX]}"

    echo "================================================================================================================="
    echo " Queueing job for shift $next_shift to run after job $PREV_HIST_JOB ends "

    # Submit the shift job for the next shift
    next_shift_job=$(sbatch --parsable \
            --dependency=afterok:${PREV_HIST_JOB} \
            "$scriptsdir/shifted_rg.sh" \
            "$UPDATED_CONFIG" "$NEXT_SHIFT_INDEX")

    echo " Submitted next shift job $next_shift_job"
    echo "================================================================================================================="
else
    echo " All shifts complete"
fi
