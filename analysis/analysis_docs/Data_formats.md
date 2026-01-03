# Analysis Data Formats

This document describes the input and output formats used by analysis scripts in this folder.

## Input Data
- **Histogram files:** NPZ format containing arrays (`histval`, `binedges`, `bincenters`).
- **Stats files:** JSON format summarizing RG step results (e.g., `overall_stats.json`).
- **Config files:** YAML format (new-format runs only) specifying run parameters.

## Output Data
- **Plots:** PNG files summarizing distributions and extracted quantities.
- **Stats:** JSON and PNG files written to stats/ and plots/ subfolders.

## Detection Logic
- **Config-based runs:** If a config YAML is present, scripts parse it for run parameters.
- **Legacy runs:** If no config is found, scripts infer parameters from folder names and filenames.
- **Automatic detection:** Scripts check for config presence and select logic accordingly.

## Backwards Compatibility
- **Old-format data:** No config file; analysis uses heuristics and may be less accurate.
- **New-format data:** Includes config file; analysis uses explicit parameters.
- **Limitations:** Legacy inference may misinterpret folder structure or filenames. Assumption: Data layout and naming follow conventions described in repo docs.

## Example
- NPZ histogram: `Data from taskfarm/fp_iqhe_numerical_shaw/FP/hist/z/z_sym_hist_RG1.npz`
- Config YAML: `Taskfarm/configs/iqhe.yaml`
- Stats JSON: `Data from taskfarm/fp_iqhe_numerical_shaw/FP/overall_stats.json`

See individual script docstrings for details on detection and assumptions.