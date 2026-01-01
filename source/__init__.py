"""Taskfarm utilities package.

This package groups helper scripts and utilities used for the project's
renormalization-group (RG) data generation and analysis. Modules provide
functions to generate RG samples, launder histograms, build/append
histograms, convert between parameterizations (t, g, z), and compute
statistics used in convergence and critical-exponent analysis.

Submodules
----------
- :mod:`data_generation`  -- batch data generation script for RG steps
- :mod:`fitters`          -- fitting helpers and peak estimators
- :mod:`helpers`          -- CLI helper scripts for laundering and conversion
- :mod:`histogram_manager`-- building and appending histograms
- :mod:`rg`               -- histogram comparison / convergence utilities
- :mod:`shift_z`          -- create shifted (perturbed) samples from symmetrized Q(z)
- :mod:`t_laundered_hist_manager` -- t-specific histogram helpers
- :mod:`utilities`       -- low-level RNG, transformations, and IO

This file intentionally keeps imports lazy; import the specific
submodules you need to avoid heavy startup costs in batch scripts.
"""
