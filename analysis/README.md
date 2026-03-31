# analysis/ — Post-processing, ν Extraction, and Plotting

This directory contains all post-processing scripts for IQHE RG Monte Carlo pipeline outputs. It is not part of the authoritative RG engine — it runs locally after histograms have been pulled from the HPC cluster, and reads `source/` and `constants.py` for shared logic and paths. Scripts here do not generate samples or modify histograms; they only read, fit, and plot.

> **QSHE analysis is not handled here.** All QSHE post-processing, visualisation, and ν extraction is performed in [`test_qshe.ipynb`](../test_qshe.ipynb) at the repository root. There is no CLI equivalent.

---

## Scripts

| File | Role |
|------|------|
| [critical_exponent.py](critical_exponent.py) | Main ν extraction pipeline — loads FP and EXP z-histograms, fits Gaussian peaks, performs log-log linear fit, and writes `peaks.json`, `overall_stats.json`, and plots |
| [data_plotting.py](data_plotting.py) | Shared helpers: histogram loading, moment computation, overlay plot generation, shared CLI parser (`build_plot_parser`) |
| [plot_stats.py](plot_stats.py) | Plot ν vs RG step from saved `overall_stats.json`; supports averaging over a range of steps via `--start`/`--end` |
| [report_plots.py](report_plots.py) | Publication-quality figure generation from processed outputs; reads saved JSON stats and histogram NPZ files |

---

## CLI usage

### Extract critical exponent ν

```bash
python -m analysis.critical_exponent \
    --version fp_iqhe_numerical_shaw \
    --mode EXP \
    --steps 9 \
    [--loc local|taskfarm]
```

`--loc` controls which data root is used (`local_dir` or `data_dir` from `.env`); defaults to taskfarm data. `--steps` must match the number of RG iterations used in the run.

### Plot ν statistics

```bash
python -m analysis.plot_stats \
    --version fp_iqhe_numerical_shaw \
    --mode EXP \
    --steps 9 \
    --start 4 \
    --end 8
```

---

## Expected input layout

Both `FP/` and `EXP/` outputs must be present under the same version directory:

```
{data_folder}/{version}/
├── FP/
│   ├── hist/
│   │   ├── t/t_hist_RG{i}.npz
│   │   └── sym_z/sym_z_hist_RG{steps-1}.npz   ← reference distribution (shift = 0)
│   ├── stats/{statistic}_moments.json
│   └── config/updated_config.yaml
└── EXP/
    └── shift_{shift}/
        └── hist/
            ├── z/z_hist_unsym_RG{i}.npz         ← per-step EXP z-histograms
            └── t/t_hist_RG{i}.npz
```

The `sym_z_hist_RG{steps-1}.npz` from the FP run serves as the zero-shift reference; all EXP peak displacements are measured relative to it.

---

## Outputs produced

`critical_exponent.py` writes the following to `{data_folder}/{version}/`:

```
peaks.json               — bootstrapped Gaussian peak estimates per (RG step, shift)
overall_stats.json       — ν, slope, and R² per RG step (from peak and mean fits)
z_peaks.png              — peak displacement vs shift scatter/line plot
Nu_{N}_shifts.png        — ν vs system size (2^n) with error bars
EXP/
├── stats/{shift}/       — per-shift moment statistics
└── plots/{shift}/       — per-shift histogram overlay PNGs
```

---

## Further documentation

- [analysis_docs/Data_formats.md](analysis_docs/Data_formats.md) — histogram NPZ schema and legacy vs config-based data formats
- [analysis_docs/Workflows.md](analysis_docs/Workflows.md) — step-by-step analysis workflow
