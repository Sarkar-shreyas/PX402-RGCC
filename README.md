# RG Monte Carlo Pipeline for Quantum Phase Transitions (IQHE & QSHE)

A Renormalization Group (RG) Monte Carlo pipeline for extracting the critical exponent ν at quantum phase transitions in the Integer Quantum Hall Effect (IQHE) and Quantum Spin Hall Effect (QSHE).

---

## Physics Background

The Integer Quantum Hall Effect (IQHE) describes two-dimensional electron systems under strong perpendicular magnetic fields, where plateaus in the Hall conductance are separated by quantum phase transitions between topologically distinct insulating states. The Quantum Spin Hall Effect (QSHE) is a time-reversal-invariant analogue driven by spin-orbit coupling, which supports topological edge states without breaking time-reversal symmetry. At the critical point separating these phases, the correlation length diverges as ξ ~ |δ|^{−ν}, where δ is the detuning from criticality and ν is the critical exponent characterising the universality class of the transition. Monte Carlo RG is used here because it allows direct numerical iteration of the RG transformation on large sample populations, providing a controlled route to the fixed-point distribution and enabling precise extraction of ν from the growth rate of relevant perturbations.

---

## Methodology

The IQHE pipeline implements the real-space renormalisation group (RSRG) approach
to the Chalker-Coddington (CC) network model described in:

> S. Shaw and R. A. Römer, "Real-space renormalisation approach to the
> Chalker-Coddington model revisited: improved statistics,"
> *Physica E*, 2024. [arXiv:2404.00660](https://arxiv.org/abs/2404.00660)

The goal is an independent reproduction of their ν extraction pipeline at production scale (320M samples), using the analytic and numerical RG unit variants described therein.

The QSHE pipeline seeks to apply the RSRG approach to the ℤ₂ network model
and adopts the (p, q) parametrisation of scattering parameters described in:

> K. Kobayashi, T. Ohtsuki, H. Obuse, and K. Slevin,
> "Conductance distributions in disordered quantum spin-Hall systems,"
> *Phys. Rev. B* 82, 165301 (2010). arXiv:[1007.4073](https://arxiv.org/abs/1007.4073)

An estimate of ν is obtained using a finite-size scaling (FSS) analysis of the RG flow
for the constructed ℤ₂ RG cell.

---

## Repository Structure

```
Project Code/
├── source/               # Core RG engine (authoritative)
├── Local/                # Local testing drivers & configs
├── Taskfarm/             # HPC orchestration scripts & configs
│   ├── scripts/          # Slurm shell scripts
│   └── configs/          # Production YAML configs
├── analysis/             # Post-processing, plotting, ν extraction
├── QSHE/                 # Experimental QSHE code
├── testing/              # Solver validation tests
├── docs/                 # Pipeline docs, runbooks, config reference
├── Data from taskfarm/   # HPC outputs (read-only, not in context)
├── Local data/           # Local run outputs (not in context)
├── report/               # Thesis report (not in context)
├── test_qshe.ipynb       # Core analysis performer for QSHE data
├── constants.py          # Global constants and paths (loads .env)
├── file_management.py    # scp/rsync transfer utility
├── requirements.txt      # numpy>=1.26, scipy>=1.11, matplotlib>=3.8, PyYAML>=6.0
└── .env                  # Local path config (not committed)
```

---

## Installation

**Python requirement:** Python ≥ 3.9 (required by numpy ≥ 1.26).

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repository root with the following variables:

```
DATA_DIR     = "...\Data from taskfarm"    # HPC output destination
LOCAL_DIR    = "...\Local data"            # Local run output
ROOT_DIR     = "...\Project Code"          # Repo root
TASKFARM_DIR = "...\Taskfarm"
QSHE_DIR     = "...\QSHE"
CONFIG_FILE  = "...\Taskfarm\configs\iqhe.yaml"
HOST         = # HPC hostname
USERNAME     = # HPC username
REMOTE_DIR   = # HPC remote directory
```

---

## Quick-start (Local IQHE)

Run a local fixed-point (FP) test with reduced sample count:

```bash
python -m Local.run_local_iqhe \
  --config Local/configs/local_iqhe \
  --set "rg_settings.steps=3" "rg_settings.samples=10000000" \
  --type FP
# Output: Local data/{version}/FP/
```

Pull histograms from the cluster:

```bash
python file_management.py --action pull --pull hist \
  --version fp_iqhe_numerical_shaw --type FP --sys linux
```

Extract the critical exponent:

```bash
python -m analysis.critical_exponent \
  --version fp_iqhe_numerical_shaw --mode EXP --steps 9
```

---

## HPC Workflow

1. **Push** code, scripts, and config to the cluster:

   ```bash
   python file_management.py --action push --push code scripts config \
     --version fp_iqhe_numerical_shaw --sys linux
   ```

2. **Submit FP run** on the cluster:

   ```bash
   bash Taskfarm/scripts/run_rg.sh \
     --config Taskfarm/configs/iqhe.yaml \
     --set "engine.method=numerical" \
     --out /tmp/configs
   ```

3. **Submit EXP (shifted) run** using the FP output:

   ```bash
   bash Taskfarm/scripts/run_shifts.sh \
     --config Taskfarm/configs/iqhe.yaml \
     --index 0 --out /tmp/configs
   ```

4. **Pull** histograms from the cluster:

   ```bash
   python file_management.py --action pull --pull hist \
     --version fp_iqhe_numerical_shaw --type FP --sys linux
   ```

5. **Analyse** — extract ν:

   ```bash
   python -m analysis.critical_exponent \
     --version fp_iqhe_numerical_shaw --mode EXP --steps 9
   ```

---

## Configuration

Run configuration follows a three-level hierarchy: a **YAML config file** (e.g. `Taskfarm/configs/iqhe.yaml` or `Local/configs/local_iqhe.yaml`) provides all base settings; **CLI overrides** via `--set "key.nested.path=value"` (parsed by `source/parse_config.py`) allow per-invocation mutations without editing the file; the merged dictionary is then validated and promoted into a typed **dataclass** (`IQHEConfig` or `QSHEConfig`, constructed by `source/config.py`) that all downstream modules consume. See [docs/Config.md](docs/Config.md) for the full key reference.

---

## Documentation

- [docs/Pipeline.md](docs/Pipeline.md) — end-to-end workflow description
- [docs/Config.md](docs/Config.md) — all configuration keys documented
- [docs/Artifacts.md](docs/Artifacts.md) — output file formats and directory layout
- [docs/Runbook.md](docs/Runbook.md) — operational procedures for common tasks
- [docs/Troubleshooting.md](docs/Troubleshooting.md) — common issues and fixes

A general README.md exists inside the `source/`, `Taskfarm/` and `analysis/` folders for further detail of each section.

---

## License / Academic Use

TODO
