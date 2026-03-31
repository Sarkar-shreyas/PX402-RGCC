# source/ — Authoritative RG Engine

This directory is the single source of truth for the entire RG Monte Carlo pipeline. It implements all core physics: MC sample generation, RG transformation logic (analytic and numerical), variable conversions, histogram construction, peak fitting, and ν extraction. Every other pipeline component — `Local/`, `Taskfarm/`, and `analysis/` — imports from here and must not duplicate any of its logic. The code in this directory runs identically in local test runs and on the Slurm HPC cluster.

---

> **Do not modify** the physics logic, public function signatures, or RNG seeding behaviour in this directory without first re-validating the pipeline against known ν values. The correctness of the critical exponent extraction depends on these implementations remaining stable and internally consistent across FP and EXP runs.

---

## File reference

| File | Role | Key exports |
|------|------|-------------|
| [utilities.py](utilities.py) | Core RG math: MC sampling, RG transformations (analytic & numerical), variable conversions, resampling, convergence statistics | `generate_t_prime`, `numerical_t_prime`, `solve_matrix_eq`, `qshe_numerical_solver`, `rg_data_workflow`, `launder`, `inverse_cdf_sampler`, `rejection_sampler`, `convert_t_to_z`, `convert_z_to_t`, `convert_g_to_z`, `convert_z_to_g`, `convert_t_to_g`, `mean_squared_distance`, `hist_moments`, `calculate_nu`, `build_rng`, `save_data` |
| [config.py](config.py) | Config loading, YAML I/O, CLI override parsing, validated `IQHEConfig`/`QSHEConfig` dataclasses | `build_config`, `handle_config`, `load_yaml`, `dump_yaml`, `parse_overrides`, `update_config`, `get_nested_data`, `check_required_info`, `get_rg_config`, `save_updated_config` |
| [data_generation.py](data_generation.py) | CLI script: generate one batch of `t'` amplitude samples for a single RG step (fresh start or continuation from prior step) | CLI only — `generate_initial_t_distribution`, `rg_data_workflow` (via `utilities`) |
| [histogram_manager.py](histogram_manager.py) | Build or append NPZ histogram files from raw sample arrays | `construct_initial_histogram`, `append_to_histogram` |
| [helpers.py](helpers.py) | CLI wrappers for laundering (z→t, t), symmetrising z-histograms, and converting t arrays to z arrays | CLI only — wraps `launder`, `inverse_cdf_sampler`, `save_data` (via `utilities`) |
| [fitters.py](fitters.py) | Peak estimation, Gaussian fitting, and linear fit for ν extraction from EXP runs | `estimate_z_peak`, `fit_z_peaks`, `std_derivative` |
| [shift_z.py](shift_z.py) | CLI script: draw samples from FP z-histogram, apply a constant shift δz, convert to t; produces the initial condition for an EXP run | CLI only — wraps `launder`, `save_data` (via `utilities`) |
| [parse_config.py](parse_config.py) | CLI entry point: load YAML, apply `--set` overrides, write `updated_config.yaml` to the run output directory | CLI only — wraps `handle_config`, `save_updated_config` (via `config`) |
| [qshe_data_gen.py](qshe_data_gen.py) | HPC script: scan a q-block of the (q, p) parameter grid, run MC trials for each pair, write per-block `.npy` arrays | CLI only — wraps `qp_trials`, `run_qp_trials` (via `utilities`) |
| [qshe_data_agg.py](qshe_data_agg.py) | HPC script: concatenate per-q-block `.npy` files into a single aggregated array per observable; write `trial_state.json` | CLI only |

---

## Variable conventions

These names and definitions are used consistently across every file in this directory.

| Name | Domain | Physical meaning |
|------|--------|-----------------|
| `t` | [0, 1] | Transmission amplitude — the fundamental RG variable. Represents the probability amplitude for an electron to transmit through a scattering region. |
| `z` | ℝ | Log-ratio RG flow parameter: `z = ln((1 − t²) / t²)`. At the RG critical fixed point the z-distribution is symmetric about z = 0. EXP runs measure how a shift in z grows under repeated RG iterations. |
| `tprime` | [0, 1] | Renormalised transmission amplitude after one RG step — the output of `generate_t_prime` or `numerical_t_prime`. Becomes the `t` input for the next step. |
| `shifts` | ℝ (small) | Constant perturbations added to every z-sample to initialise EXP runs (e.g. `0.003`, `0.005`). Must be small enough to remain in the linear regime near the fixed point. The growth rate of the shift over RG steps yields the relevant RG eigenvalue λ = 1/ν. |
