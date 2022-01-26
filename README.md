# RP21_RE

Repeatibility evaluation for "Reachability of weakly nonlinear systems using Carleman linearization" (RP'21) by Marcelo Forets and Christian Schilling.

The article was accepted at the [15th International Conference on Reachability Problems](https://rp2021.csc.liv.ac.uk/index.html), which took place in Liverpool, UK between October 25-27, 2021.

An arXiv version is available [here](https://arxiv.org/abs/2108.10390).

![](https://rp2021.csc.liv.ac.uk/images/LivVGM-McCoy_Wynne-5880.png)

## Installation

- Install Julia (download [here](https://julialang.org/downloads/)).

- Run the script `startup.jl` either within the Julia REPL with `include("startup.jl")` or from the terminal:

```bash
$ julia startup.jl
```

The startup script adds the package implementing our method plus some additional dependencies for simulation and plotting. Resulting plots (in pdf format) and runtime table (in CSV format) are stored in a new folder named `results`.


