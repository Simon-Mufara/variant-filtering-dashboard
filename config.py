# Default filter thresholds
DEFAULT_MIN_QUAL = 30
DEFAULT_MIN_DP = 10
DEFAULT_MAX_AF = 1.0
DEFAULT_MIN_AF = 0.0

# Variant types
VARIANT_TYPES = ["SNP", "INDEL", "MNP", "ALL"]

# Chromosome list (human)
CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]

# Plot settings
PLOT_HEIGHT = 450
PLOT_WIDTH = 700
COLOR_PALETTE = {
    "SNP": "#1f77b4",
    "INDEL": "#ff7f0e",
    "MNP": "#2ca02c",
}
