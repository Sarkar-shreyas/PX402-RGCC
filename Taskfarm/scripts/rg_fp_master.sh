#!/bin/bash
#SBATCH --job-name=rg_fp_master
#SBATCH --output=../job_outputs/bootstrap/%x_%A.out
#SBATCH --error=../job_logs/bootstrap/%x_%A.err

# Define the constants for this RG flow
N=800000 # Total number of samples
NUM_RG_ITERS=10 # Number of RG steps
VERSION=1.8S  # Version for tracking changes and matrix used
TYPE="FP" # Type flag to toggle symmetrisation/launder target
INITIAL=1 # Flag to generate starting distribution/histograms or not
EXISTING_T="" # Placeholder var to point to data file for non-initial RG steps
prev_hist_job="" # Placeholder var for holding previous job ID when setting up dependency

basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Our root directory
joboutdir="$basedir/job_outputs/v${VERSION}/$TYPE" # Where the output files will go
datadir="$joboutdir/data" # Where the data will live
scriptsdir="$basedir/scripts" # Where all shell scripts live
logsdir="$basedir/job_logs/v${VERSION}/$TYPE" # Where log files will go
mkdir -p "$logsdir" "$joboutdir" # Make them in case they aren't already there

exec >"$joboutdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.out" # Redirect outputs to be within their own folders, together with the data they produce
exec 2>"$logsdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.err" # Redirect error logs to be within their own folders for easy grouping


echo "==================================================="
echo "                  SLURM JOB INFO "
echo "---------------------------------------------------"
echo " Job Name         : $SLURM_JOB_NAME"
echo " Job ID           : $SLURM_JOB_ID"
echo " Submitted from   : $SLURM_SUBMIT_DIR"
echo " Type             : $TYPE"
echo " Current dir      : $(pwd)"
echo " Date of job      : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "==================================================="
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
        --output=../job_outputs/bootstrap/rg_gen_RG${step}_%x_%A_%a.out \
        --error=../job_logs/bootstrap/rg_gen_RG${step}_%x_%A_%a.err \
        "$scriptsdir/rg_gen_batch.sh"\
            "$N" "$INITIAL" "$EXISTING_T" "$step" "$VERSION" "$TYPE")

        echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted generation job for RG step $step : $gen_job "
    else
        gen_job=$(sbatch --parsable \
        --dependency=afterok:${prev_hist_job} \
        --output=../job_outputs/bootstrap/rg_gen_RG${step}_%A_%a.out \
        --error=../job_logs/bootstrap/rg_gen_RG${step}_%A_%a.err \
        "$scriptsdir/rg_gen_batch.sh"\
            "$N" "$INITIAL" "$EXISTING_T" "$step" "$VERSION" "$TYPE")

        echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted generation job for RG step $step : $gen_job (after ${prev_hist_job}) "
    fi
    sleep 1
    echo "----------------------------------------------------------------------------------------------------------------"
    hist_job=$(sbatch --parsable \
        --dependency=afterok:${gen_job} \
        --output=../job_outputs/bootstrap/rg_hist_RG${step}_%A.out \
        --error=../job_logs/bootstrap/rg_hist_RG${step}_%A.err \
        "$scriptsdir/rg_hist_manager.sh" \
        "$N" "$step" "$TYPE" "$VERSION")
    echo "----------------------------------------------------------------------------------------------------------------"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted histogram job for RG step $step  : $hist_job (after ${gen_job}) "
    # Keep track of the job ID for setting up dependencies in order
    prev_hist_job="$hist_job"
    INITIAL=0

    EXISTING_T="$laundereddir"
    sleep 1
done
echo "================================================================================================================"
echo " All ${NUM_RG_ITERS} RG jobs submitted. Final dependency ends at JOB${prev_hist_job} "
