# Erlang vaccination experiments: code and data

The experiments use deterministic event-driven ODE integration. Switching
times are searched using endpoint checks, a uniform grid and refinement of
every sampled local minimum, for both the weighted and total-infection targets.
The finite-grid search does not certify a global minimum of the continuous
scalar objective. Optimization for the total-infection target is restricted to
single-switch controls.

From the folder containing `code/`, run:

```sh
python -m pip install -r code/requirements.txt
python code/run_numerics.py
python code/run_validation.py --workers 4
python code/render_results.py
python code/compare_target_outcomes.py
```

The first script calculates the parameter sweep, selected cases, switching
curves and adjoints. The second calculates grid and tolerance refinements,
initial stage profiles and a rebound example. The third renders tables and
figures, and the last compares vaccination and infection outcomes for the two
targets. The calculations use ODE trajectories; they do not numerically solve
the Hamilton--Jacobi--Bellman equation.

To render from the supplied data without repeating the main numerical search,
run the first two scripts with `--render-only`, followed by the third and fourth
commands above. The fourth command recomputes the target-outcome comparison.
Tables and figures are written to `code/output/rendered/`.

`results.json` and `validation_results.json` contain the saved calculations;
`target_outcomes.json` records the target comparison. The files record
settings and software versions. Calculations are CPU-only and may take several
minutes; small numerical differences can occur across environments.
