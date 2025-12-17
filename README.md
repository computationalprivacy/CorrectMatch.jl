# CorrectMatch

[![Build status](https://github.com/computationalprivacy/CorrectMatch.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/computationalprivacy/CorrectMatch.jl/actions/workflows/CI.yml)
[![version](https://juliahub.com/docs/General/CorrectMatch/stable/version.svg)](https://juliahub.com/ui/Packages/General/CorrectMatch)

## Installation

CorrectMatch requires gfortran to precompile the mvndst routines.

On macOS, using Homebrew, run:
```brew install gcc```

On GNU/Linux, install the gfortran package with your preferred package manager:
```
sudo apt-get install gfortran  # on Debian-based systems
sudo pacman -S gcc-gfortran    # on Archlinux-based systems
```


## Usage

CorrectMatch works directly with DataFrames, automatically handling categorical variables:

```julia
using CorrectMatch
using DataFrames

# Load your data as a DataFrame
df = DataFrame(
    age = [25, 30, 35, 25, 30],
    gender = ["M", "F", "X", "M", "M"],
    city = ["NYC", "LA", "NYC", "NYC", "SF"]
)

# Compute population metrics directly on DataFrame
uniqueness(df)   # 0.60 (fraction of unique records)
correctness(df)  # 0.80 (fraction of correctly re-identifiable records)

# Fit a Gaussian copula model
G = fit_mle(GaussianCopula, df)

# Generate synthetic data
d_sim = rand(G, 100)
uniqueness(d_sim)  # e.g., 0.04
```

Individual uniqueness and correctness can be computed for any record:

```julia
indiv = df[1, :]
individual_uniqueness(G, indiv, 100)  # e.g., 0.20
individual_correctness(G, indiv, 100)  # e.g., 0.50

# Or pass raw values
individual_uniqueness(G, [35, "M", "NYC"], 100)  # e.g., 0.35
```

### Working with integer matrices

The codebase also supports working directly with integer matrices, where each column represents a categorical variable encoded as integers starting from 1. This allows for using the `exact_marginal=false` option for better fitting distributions in small sparse datasets.

```julia
# Create a simple dataset of 1000 records and 3 columns
d = rand(1:10, 1000, 3)
uniqueness(d)

G = fit_mle(GaussianCopula, d)
d_sim = rand(G, 1000)
uniqueness(d_sim)
correctness(d_sim)

# Individual metrics (values must be 1-indexed for matrix API)
individual_uniqueness(G, [5, 5, 5], 1000)
individual_correctness(G, [5, 5, 5], 1000)
```

See the [examples](https://github.com/computationalprivacy/CorrectMatch.jl/tree/master/examples) folder to learn how to load a CSV file and estimate the metrics from a small sample.

## License
GNU General Public License v3.0

See LICENSE to see the full text.

Patent-pending code. Additional support and details are available for commercial uses.
