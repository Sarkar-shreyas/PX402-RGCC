#!/bin/bash
#SBATCH --job-name=temp_gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@30
#SBATCH --kill-on-invalid-dep=yes
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
VERSIONSTR="$3"
NUM_SAMPLES="$(get_yaml "$UPDATED_CONFIG" "rg_settings.samples")"
NUM_STEPS="$(get_yaml "$UPDATED_CONFIG" "rg_settings.steps")"
SEED="$(get_yaml "$UPDATED_CONFIG" "rg_settings.seed")"
Q_BLOCK="$4"
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
GEN_SCRIPT="$SRC_DIR/qshe_data_gen.py"

set -euo pipefail
export RG_CONFIG="$UPDATED_CONFIG"

GEN_SEED=$(( SEED + (Q_BLOCK * 100003) + 1))
PHI_SEED=$(( SEED + 123456 ))


python "$GEN_SCRIPT" \
    "$NUM_SAMPLES" \
    "$NUM_STEPS" \
    "$Q_BLOCK" \
    "$Q_BLOCK_SIZE" \
    "$PHI_SEED" \
    "$GEN_SEED" \
    "$datadir"
