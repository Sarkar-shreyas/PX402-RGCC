#!/bin/bash
#SBATCH --job-name=shifted_rg_master
#SBATCH --output=../job_outputs/bootstrap/%x_%A.out
#SBATCH --error=../job_logs/bootstrap/%x_%A.err

# Define the constants for this RG flow
N=200000000 # Total number of samples
NUM_RG_ITERS=8 # Number of RG steps
NUM_BATCHES=16 # Number of batches to generate/process data over
VERSION=1.63S # Version for tracking changes and matrix used
TYPE="EXP" # Type flag to toggle symmetrisation/launder target
INITIAL=1 # Flag to generate starting distribution/histograms or not
EXISTING_T="" # Placeholder var to point to data file for non-initial RG steps
prev_hist_job="" # Placeholder var for holding previous job ID when setting up dependency
last_step=$((NUM_RG_ITERS - 1)) # Var to determine which FP distribution to use for generating the shifted dataset, for now we use the latest one
SAMPLE_SIZE=$(( N / NUM_BATCHES )) # How many samples should be calculated per batch

CURRENT_SHIFT="$1" # Takes in the shift value to apply for this round.
shift # Move onto the next input
REMAINING_SHIFTS=("$@") # Store the remaining shifts from the array

basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Our root directory
joboutdir="$basedir/job_outputs/v${VERSION}/$TYPE/shift_${CURRENT_SHIFT}" # Where the output files will go
datadir="$joboutdir/data" # Where the data will live
scriptsdir="$basedir/scripts" # Where all shell scripts live
logsdir="$basedir/job_logs/v${VERSION}/$TYPE/shift_${CURRENT_SHIFT}" # Where log files will go
mkdir -p "$logsdir" "$joboutdir" "$datadir" # Make them in case they aren't already there

exec >"$joboutdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.out" # Redirect outputs to be within their own folders, together with the data they produce
exec 2>"$logsdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.err" # Redirect error logs to be within their own folders for easy grouping

# General job information
echo "==================================================="
echo "                  SLURM JOB INFO "
echo "---------------------------------------------------"
echo " Job Name         : $SLURM_JOB_NAME"
echo " Job ID           : $SLURM_JOB_ID"
echo " Submitted from   : $SLURM_SUBMIT_DIR"
echo " Type             : $TYPE"
echo " Shift            : $CURRENT_SHIFT"
echo " Current dir      : $(pwd)"
echo " Date of job      : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "==================================================="
echo ""

# Where the FP distribution we'll use to generate shifted data lives
FP_dist="$basedir/job_outputs/v${VERSION}/FP/data/RG${last_step}/hist/sym_z/sym_z_hist_RG${last_step}.npz"


echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Generating initial data for shift ${CURRENT_SHIFT} "
# Run the job to generate initial shifted data
shift_job=$(sbatch --parsable \
    --output=../job_outputs/bootstrap/gen_shift_${CURRENT_SHIFT}_%A_%a.out \
    --error=../job_logs/bootstrap/gen_shift_${CURRENT_SHIFT}_%A_%a.out \
    "$scriptsdir/gen_shifted_data.sh" \
        "$SAMPLE_SIZE" "$VERSION" "$FP_dist" "Initial" "$CURRENT_SHIFT")

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
        EXISTING_T="$datadir/Initial/perturbed_t_shift_${CURRENT_SHIFT}.npy"
    else
        prev_step=$(( $step - 1 ))
        gen_job_dep="$prev_hist_job"
        EXISTING_T="$datadir/RG${prev_step}/laundered"
    fi

    gen_job=$(sbatch --parsable \
    --dependency=afterok:${gen_job_dep} \
    --output=../job_outputs/bootstrap/rg_gen_RG${step}_%A_%a.out \
    --error=../job_logs/bootstrap/rg_gen_RG${step}_%A_%a.err \
    "$scriptsdir/rg_gen_batch.sh" \
        "$N" "$INITIAL" "$EXISTING_T" "$step" "$VERSION" "$TYPE" "$CURRENT_SHIFT")

    echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted generation job for RG step $step : $gen_job (after ${gen_job_dep}) "

    echo "-----------------------------------------------------------------------------------------------------------------"
    hist_job=$(sbatch --parsable \
        --dependency=afterok:${gen_job} \
        --output=../job_outputs/bootstrap/rg_hist_RG${step}_%A.out \
        --error=../job_logs/bootstrap/rg_hist_RG${step}_%A.err \
        "$scriptsdir/rg_hist_manager.sh" \
        "$N" "$step" "$TYPE" "$VERSION" "$CURRENT_SHIFT")

    echo "-----------------------------------------------------------------------------------------------------------------"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted histogram job for RG step $step  : $hist_job (after ${gen_job}) "
    # Keep track of the job ID for setting up dependencies in order
    prev_hist_job="$hist_job"
    INITIAL=0

    EXISTING_T="$laundereddir"
    sleep 5
done
echo "================================================================================================================="
echo " All ${NUM_RG_ITERS} RG jobs for shift ${CURRENT_SHIFT} submitted. Final dependency ends at job ${prev_hist_job} "
echo "================================================================================================================="

# Check if there are any shifts remaining, then send in jobs for the next
if [[ "${#REMAINING_SHIFTS[@]}" -gt 0 ]]; then
    # Assign the next shift to run, and store any remaining
    next_shift="${REMAINING_SHIFTS[0]}"
    new_remaining=("${REMAINING_SHIFTS[@]:1}")

    echo "================================================================================================================="
    echo " Queueing job for shift $next_shift to run after job $prev_hist_job ends "
    echo " Remaining shifts: ${new_remaining[*]:-(None)} "

    # Submit the shift job for the next shift with remaining shifts if there are any. Dependency of prev hist jobs makes sure 1 shift is done at a time
    if [[ "${#new_remaining[@]}" -gt 0 ]]; then
        next_shift_job=$(sbatch --parsable \
            --dependency=afterok:${prev_hist_job} \
            "$scriptsdir/shifted_rg.sh" \
            "$next_shift" "${new_remaining[@]}")
    else
        next_shift_job=$(sbatch --parsable \
            --dependency=afterok:${prev_hist_job} \
            "$scriptsdir/shifted_rg.sh" \
            "$next_shift")
    fi
    echo " Shift job chain queued for all shifts "
    echo "================================================================================================================="
fi
