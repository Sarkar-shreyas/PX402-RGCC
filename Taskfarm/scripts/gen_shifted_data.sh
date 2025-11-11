#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --job-name=gen_shift
#SBATCH --output=../job_outputs/bootstrap/%x_%A_%a.out
#SBATCH --error=../job_logs/bootstrap/%x_%A_%a.err

# Config
N="$1"
VERSION="$2"
INPUT_FILE="$3"
STEP="$4"
shift="$5"
TYPE="EXP"

set -euo pipefail

# Libraries needed
module purge
module load GCC/13.3.0 SciPy-bundle/2024.05

basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)" # Root, fyp for now
codedir="$basedir/code" # Where the code lives
logsdir="$basedir/job_logs/v${VERSION}/$TYPE/shift_${shift}/${SLURM_JOB_NAME}" # Where logs will be sent
outputdir="$basedir/job_outputs/v${VERSION}/$TYPE/shift_${shift}" # Where the outputs will live
joboutdir="$outputdir/output/${SLURM_JOB_NAME}/${STEP}" # General output directory
jobdatadir="$outputdir/data" # Where the data will go
stepdir="$jobdatadir/${STEP}"
mkdir -p "$outputdir" "$logsdir" # Make these now so that it does it every time we run this job
mkdir -p "$joboutdir" "$jobdatadir" "$stepdir"

exec > >(tee -a "$joboutdir/JOB${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$logsdir/JOB${SLURM_JOB_ID}.err" >&2)

echo "==================================================="
echo "                  SLURM JOB INFO "
echo "---------------------------------------------------"
echo " Job Name         : $SLURM_JOB_NAME"
echo " Job ID           : $SLURM_JOB_ID"
echo " Submitted from   : $SLURM_SUBMIT_DIR"
echo " Current dir      : $(pwd)"
echo " Date of job      : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "==================================================="
echo ""


echo "====================================================================="
echo "      Config for shifting samples from the FP for shift $shift "
echo "---------------------------------------------------------------------"
echo " Total samples         : $N"
echo " Input FP distribution : $INPUT_FILE"
echo " Output data directory : $stepdir"
echo "====================================================================="
echo ""


export PYTHONPATH="$codedir:$PYTHONPATH"
cd "$codedir"
SRC_DIR="$codedir/source" # This is where the actual code lives

OUTPUT_FILE="$jobdatadir/${STEP}/perturbed_t_shift_${shift}.npy"

python -m "source.shift_z" \
    "$N" \
    "$INPUT_FILE" \
    "$OUTPUT_FILE" \
    "$shift"

echo "======================================================================================================="
echo " Data shift job ${SLURM_JOB_ID} for Shift ${shift} completed on : [$(date '+%Y-%m-%d %H:%M:%S')] "
echo "======================================================================================================="
echo ""
