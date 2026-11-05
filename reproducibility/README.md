# Numerical experiments: code and data

This directory contains the current experiment code and saved numerical data.
Its `code/` directory includes all files needed to reproduce the tables and
figures. The repository root retains an earlier set of experiments.

From this directory, install the dependencies and run:

```sh
python -m pip install -r code/requirements.txt
python code/run_numerics.py
python code/run_validation.py --workers 4
python code/render_results.py
python code/compare_target_outcomes.py
```

The first two scripts recompute the numerical experiments; the third renders
tables and figures, and the fourth calculates vaccination and infection
outcomes for the two stopping targets. The finite-grid search does not certify
a global minimum of the continuous switching-time objective. The total-infection
target comparison is restricted to single-switch policies. The computations
evaluate ODE trajectories and do not numerically solve the Hamilton--Jacobi--
Bellman equation.

To render from saved data without repeating the main numerical search, run the
first two scripts with `--render-only`, followed by the third and fourth
commands. The fourth command recomputes the target-outcome comparison. Tables
and figures are written to `code/output/rendered/`. See
[`code/README.md`](code/README.md) for details.
