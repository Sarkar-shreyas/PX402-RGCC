# Analysis Overview

This folder contains post-processing scripts for RG Monte Carlo pipeline outputs. Analysis scripts estimate physical quantities (e.g., critical exponent ν), generate diagnostic and publication-quality plots, and summarize simulation results. These scripts are not part of the authoritative pipeline and are intended for local use only.

## Compatibility
- **Config-based runs:** Analysis scripts read YAML config files for run parameters and metadata.
- **Legacy runs:** If no config file is present, scripts fall back to heuristics and folder/filename parsing.
- **Detection:** Scripts automatically detect config presence and adapt logic; see individual docstrings for details.

## Usage
Run analysis scripts locally after pulling data from the cluster. Example:

```bash
python -m analysis.critical_exponent --version fp_iqhe_numerical_shaw --mode EXP
```

## Backwards Compatibility
- Old-format data: No config file; analysis uses legacy inference.
- New-format data: Includes config file; analysis uses config-based logic.
- Limitations: Legacy inference may be less accurate; see DATA_FORMATS.md for details.

---
For data formats and workflow details, see DATA_FORMATS.md and WORKFLOWS.md.
