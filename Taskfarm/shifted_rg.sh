#!/bin/bash
#SBATCH --job-name=shifted_rg_master
#SBATCH --output=../job_outputs/bootstrap/%x_%A.out
#SBATCH --error=../job_logs/bootstrap/%x_%A.err

N=120000000
NUM_RG_ITERS=8
VERSION=1.32
TYPE="EXP"
INITIAL=1
EXISTING_T=""
prev_hist_job=""
last_step=$((NUM_RG_ITERS - 1))
SAMPLE_SIZE=$(( N / NUM_RG_ITERS ))

shift="$1"

basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Root, fyp for now
joboutdir="$basedir/job_outputs/v${VERSION}/$TYPE/shift_${shift}"
datadir="$joboutdir/data"
scriptsdir="$basedir/scripts"
logsdir="$basedir/job_logs/v${VERSION}/$TYPE/shift_${shift}"
mkdir -p "$logsdir" "$joboutdir" "$datadir"

exec > >(tee -a "$joboutdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$logsdir/${SLURM_JOB_NAME}_JOB${SLURM_JOB_ID}.err" >&2)

echo "==================================================="
echo "                  SLURM JOB INFO "
echo "---------------------------------------------------"
echo " Job Name         : $SLURM_JOB_NAME"
echo " Job ID           : $SLURM_JOB_ID"
echo " Submitted from   : $SLURM_SUBMIT_DIR"
echo " Type             : $TYPE"
echo " Shift            : $shift"
echo " Current dir      : $(pwd)"
echo " Date of job      : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "==================================================="
echo ""

FP_dist="$basedir/job_outputs/v${VERSION}/FP/data/RG${last_step}/hist/sym_z/sym_z_hist_RG${last_step}.npz"


echo "============================================================================"
echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Generating initial data for shift ${shift} "

shift_job=$(sbatch --parsable \
    --output=../job_outputs/bootstrap/gen_shift_${shift}_%A_%a.out \
    --error=../job_logs/bootstrap/gen_shift_${shift}_%A_%a.out \
    "$scriptsdir/gen_shifted_data.sh" \
        "$SAMPLE_SIZE" "$VERSION" "$FP_dist" "Initial" "$shift")

echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted shift job ${shift_job} for shift ${shift}"

for step in $(seq 0 $(( NUM_RG_ITERS - 1 ))); do
    echo "================================================================================================================"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Proceeding with RG step $step of type $TYPE"
    laundereddir="$datadir/RG${step}/laundered"

    if [ "$step" -eq 0 ]; then
        INITIAL=0
        gen_job_dep="$shift_job"
        EXISTING_T="$datadir/Initial/perturbed_t_shift_${shift}.txt"
    else
        prev_step=$(( $step - 1 ))
        gen_job_dep="$prev_hist_job"
        EXISTING_T="$datadir/RG${prev_step}/laundered"
    fi

    gen_job=$(sbatch --parsable \
    --dependency=afterok:${gen_job_dep} \
    --output=../job_outputs/bootstrap/rg_gen_RG${step}_%A_%a.out \
    --error=../job_logs/bootstrap/rg_gen_RG${step}_%A_%a.err \
    "$scriptsdir/rg_gen_batch.sh"\
        "$N" "$INITIAL" "$EXISTING_T" "$step" "$VERSION" "$TYPE" "$shift")

    echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted generation job for RG step $step : $gen_job (after ${gen_job_dep}) "

    echo "----------------------------------------------------------------------------------------------------------------"
    hist_job=$(sbatch --parsable \
        --dependency=afterok:${gen_job} \
        --output=../job_outputs/bootstrap/rg_hist_RG${step}_%A.out \
        --error=../job_logs/bootstrap/rg_hist_RG${step}_%A.err \
        "$scriptsdir/rg_hist_manager.sh" \
        "$N" "$step" "$TYPE" "$VERSION" "$shift")
    echo "----------------------------------------------------------------------------------------------------------------"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted histogram job for RG step $step  : $hist_job (after ${gen_job}) "

    prev_hist_job="$hist_job"
    INITIAL=0

    EXISTING_T="$laundereddir"

done
echo "================================================================================================================"
echo " All ${NUM_RG_ITERS} RG jobs submitted. Final dependency ends at JOB${prev_hist_job} "
