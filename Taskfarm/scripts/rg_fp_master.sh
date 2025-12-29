#!/bin/bash
#SBATCH --job-name=rg_fp_master
#SBATCH --output=../job_outputs/bootstrap/%x_%A.out
#SBATCH --error=../job_logs/bootstrap/%x_%A.err

set -euo pipefail

basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Our root directory
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

# Define the constants for this RG flow
TYPE="FP" # Mode of RG workflow
VERSION="$(get_yaml "$UPDATED_CONFIG" "main.version")"  # Version for tracking changes and matrix used
N="$(get_yaml "$UPDATED_CONFIG" "rg_settings.samples")" # Total number of samples
NUM_RG_ITERS="$(get_yaml "$UPDATED_CONFIG" "rg_settings.steps")" # Number of RG steps
SEED="$(get_yaml "$UPDATED_CONFIG" "rg_settings.seed")" # Starting seed
METHOD="$(get_yaml "$UPDATED_CONFIG" "engine.method")" # Flag to determine whether to use analytic or numerical methods
EXPR="$(get_yaml "$UPDATED_CONFIG" "engine.expr")" # Flag to determine which expression to use
RESAMPLE="$(get_yaml "$UPDATED_CONFIG" "engine.resample")" # Flag to determine which type of resample to use
SYMMETRISE="$(get_yaml "$UPDATED_CONFIG" "engine.symmetrise")" # Flag to determine whether to symmetrise data or not
VERSIONSTR="${VERSION}_${METHOD}_${EXPR}"
INITIAL=1 # Flag to generate starting distribution/histograms or not
EXISTING_T="" # Placeholder var to point to data file for non-initial RG steps
PREV_HIST_JOB="" # Placeholder var for holding previous job ID when setting up dependency
export RG_CONFIG=$UPDATED_CONFIG

joboutdir="$basedir/job_outputs/${VERSIONSTR}/$TYPE" # Where the output files will go
datadir="$joboutdir/data" # Where the data will live
scriptsdir="$basedir/scripts" # Where all shell scripts live
logsdir="$basedir/job_logs/${VERSIONSTR}/$TYPE" # Where log files will go
mkdir -p "$logsdir" "$joboutdir" # Make them in case they aren't already there

exec >"$joboutdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.out" # Redirect outputs to be within their own folders, together with the data they produce
exec 2>"$logsdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.err" # Redirect error logs to be within their own folders for easy grouping


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
echo "                  RG WORKFLOW CONFIG "
echo "------------------------------------------------------"
echo " Version             : $VERSIONSTR "
echo " Type                : $TYPE "
echo " Number of samples   : $N "
echo " Number of RG steps  : $NUM_RG_ITERS "
echo " Starting seed       : $SEED "
echo " t' Method           : $METHOD "
echo " Expression          : $EXPR "
echo " Resample method     : $RESAMPLE "
echo " Symmetrising?       : $SYMMETRISE "
echo " Date of job         : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "======================================================"
echo ""
# For each RG, we queue data generation and then histogram jobs, dependencies ensure they run in sequence
for step in $(seq 0 $(( NUM_RG_ITERS - 1 ))); do
    echo "================================================================================================================"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Proceeding with RG step $step "
    laundereddir="$datadir/RG${step}/laundered"
    # If its the first step, we need pass in an empty string so it generates the initial t distribution
    # later steps go back to using data from the previous RG step
    if [ "$step" -eq 0 ]; then
        EXISTING_T=""
    else
        prev_step=$(( $step - 1 ))
        EXISTING_T="$datadir/RG${prev_step}/laundered"
    fi
    # Run the first step without any dependency, then set the last histogram job as its dependency
    if [ "$step" -eq 0 ]; then
        gen_job=$(sbatch --parsable \
        --output=../job_outputs/bootstrap/rg_gen_RG${step}_%A_%a.out \
        --error=../job_logs/bootstrap/rg_gen_RG${step}_%A_%a.err \
        "$scriptsdir/rg_gen_batch.sh"\
            "$UPDATED_CONFIG" "$TYPE" "$VERSIONSTR" "$N" "$step" "$SEED" "$METHOD" "$EXPR" "$INITIAL" "$EXISTING_T" )

        echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted generation job for RG step $step : $gen_job "
    else
        gen_job=$(sbatch --parsable \
        --dependency=afterok:${PREV_HIST_JOB} \
        --output=../job_outputs/bootstrap/rg_gen_RG${step}_%A_%a.out \
        --error=../job_logs/bootstrap/rg_gen_RG${step}_%A_%a.err \
        "$scriptsdir/rg_gen_batch.sh"\
            "$UPDATED_CONFIG" "$TYPE" "$VERSIONSTR" "$N" "$step" "$SEED" "$METHOD" "$EXPR" "$INITIAL" "$EXISTING_T" )

        echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted generation job for RG step $step : $gen_job (after ${PREV_HIST_JOB}) "
    fi
    #sleep 1
    echo "----------------------------------------------------------------------------------------------------------------"
    hist_job=$(sbatch --parsable \
        --dependency=afterok:${gen_job} \
        --output=../job_outputs/bootstrap/rg_hist_RG${step}_%A.out \
        --error=../job_logs/bootstrap/rg_hist_RG${step}_%A.err \
        "$scriptsdir/rg_hist_manager.sh" \
        "$UPDATED_CONFIG" "$TYPE" "$VERSIONSTR" "$N" "$step" "$SEED" "$RESAMPLE" "$SYMMETRISE")
    echo "----------------------------------------------------------------------------------------------------------------"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted histogram job for RG step $step  : $hist_job (after ${gen_job}) "
    # Keep track of the job ID for setting up dependencies in order
    PREV_HIST_JOB="$hist_job"
    INITIAL=0

    #sleep 1
done
echo "================================================================================================================"
echo " All ${NUM_RG_ITERS} RG jobs submitted. Final dependency ends at JOB${PREV_HIST_JOB} "
