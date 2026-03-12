#!/bin/bash

# Set default params
SETS=()
OUT=""
JOB="temp_gen.sh"
TYPE="QP"
Q_BLOCK_SIZE=10
Q_BLOCK=0
VARS="p,q"
VERSIONSTR=""

set -euo pipefail

# Read command line input
while [[ $# -gt 0 ]]; do
    case "$1" in
        --job)
            JOB="$2";
            shift 2;;
        -o|--out)
            OUT="$2";
            shift 2;;
        --q-block-size)
            Q_BLOCK_SIZE="$2";
            shift 2;;
        --q-block)
            Q_BLOCK="$2";
            shift 2;;
        --vars)
            VARS="$2";
            shift 2;;
        --version)
            VERSIONSTR="$2";
            shift 2;;
        --)
            shift; break;;
        *)
            echo "Unknown arg $1"
            exit 2 ;;
    esac
done




basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)" # Our root directory
codedir="$basedir/code" # Where the code lives
scriptsdir="$basedir/scripts" # Where all shell scripts live
CONFIG="$basedir/job_outputs/$VERSIONSTR/$TYPE/config/updated_config.yaml"

# Libraries needed
module purge
module load GCC/13.3.0 SciPy-bundle/2024.05
source "$basedir/.venv/bin/activate"

export PYTHONPATH="$codedir:$PYTHONPATH"

if (( ${#SETS[@]} )); then
    UPDATED_CONFIG="$(
        python "$codedir/source/parse_config.py" \
        --config "$CONFIG" \
        --type "$TYPE" \
        --set "${SETS[@]}" \
        ${OUT:+--out "$OUT"}
    )"
else
    UPDATED_CONFIG="$(
        python "$codedir/source/parse_config.py" \
        --config "$CONFIG" \
        --type "$TYPE" \
        ${OUT:+--out "$OUT"}
    )"
fi

qp_job=$(sbatch --parsable \
        --output="$basedir/job_outputs/$VERSIONSTR/$TYPE/output/temp_gen_%A.out" \
        --error="$basedir/job_logs/$VERSIONSTR/$TYPE/temp_gen_%A.err" \
        "$scriptsdir/$JOB" \
        "$UPDATED_CONFIG" "$Q_BLOCK_SIZE" "$VERSIONSTR" "$Q_BLOCK" )

echo " [$(date '+%Y-%m-%d %H:%M:%S')]: Submitted QP job with id $qp_job "
echo " Gen job of block $Q_BLOCK with size $Q_BLOCK_SIZE for version $VERSIONSTR "
