"""Configuration parameters for RG flow analysis.

This module defines the key numerical parameters used throughout the analysis:

Simulation Parameters:
- N: Number of samples per distribution (default 100,000)
- K: Number of RG steps to perform (default 9)

Distribution Parameters:
- T_BINS: Number of bins for t-space histograms (default 1000)
- Z_BINS: Number of bins for z-space histograms (default 2000)
- T_RANGE: Range for t-space binning (default (0,1))
- Z_RANGE: Range for z-space binning (default (-25,25))

RG Parameters:
- Z_PERTURBATION: Size of perturbation around fixed point (default 0.007)
- DIST_TOLERANCE: L2 distance threshold for convergence (default 0.001)
- STD_TOLERANCE: Standard deviation change threshold (default 0.0005)

Expression Selection:
- EXPRESSION: Which RG transformation to use ("Shaw", "Shreyas", "Cain", "Jack")
"""

N: int = 5 * (10**7)
K: int = 9
T_BINS: int = 1000
Z_BINS: int = 50000
Z_RANGE: tuple = (-25.0, 25.0)
Z_PERTURBATION: float = 0.007
DIST_TOLERANCE: float = 0.001
STD_TOLERANCE: float = 0.0005
T_RANGE: tuple = (0, 1.0)
EXPRESSION: str = "Shaw"
# EXPRESSION = "Shreyas"
# EXPRESSION = "Cain"
# EXPRESSION = "Jack"
