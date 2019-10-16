# CorrectMatch

[![Build Status](https://travis-ci.org/computationalprivacy/CorrectMatch.jl.svg)](https://travis-ci.org/computationalprivacy/CorrectMatch.jl)


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

CorrectMatch contains functions to fit a copula model and estimate the population uniqueness:

```julia
using CorrectMatch
# Create a simple dataset of 100 records and 4 independent columns, and compute the true uniqueness
d = rand(1:10, 1000, 3)
uniqueness(d)  # 0.377

# The first precompilation takes a few seconds
G = fit_mle(GaussianCopula, d)
d_sim = rand(G, 1000)
uniqueness(d_sim)  # 0.361
```

but also the likelihood of uniqueness for a single individual:
```julia
# Uniqueness of record (5, 5, 5)
individual_uniqueness(G, [5, 5, 5], 1000)  # 0.34276768289842735
```

See the [examples](https://github.com/computationalprivacy/CorrectMatch.jl/tree/master/examples) folder to learn how to load a CSV file and estimate the uniqueness from a small sample.

## License
GNU General Public License v3.0

See LICENSE to see the full text.

Patent-pending code. Additional support and details are available for commercial uses.
