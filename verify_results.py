"""Audit the supplied ODE data, including fresh endpoint re-evaluation.

Run after both numerical scripts. This is a numerical consistency check,
not a proof of global optimality or a uniform PDE residual certificate.
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_numerics as num


def main():
    data = json.loads((num.OUT / "validation_results.json").read_text())
    original = json.loads((num.OUT / "results.json").read_text())
    num.ODE = data["setting"]["tight_ode"]
    endpoints = []
    baseline_differences = []
    for row in data["convergence"]:
        model = num.Model(row["n"], row["beta"], row["gamma"])
        z0 = np.array(row["z0"])
        ref = row["tight"]["estimates"][0]
        pol = model.policy(ref["tau"], z0)
        err = abs(pol["T"] - ref["V"])
        assert err < 1e-9 and abs(model.margin(pol["zT"])) < 1e-10
        assert 0 <= ref["tau"] <= ref["W"] and 0 < ref["V"] <= ref["W"]
        # Stability of the actual reported three-decimal values, not a
        # bound on an unknown exact value.
        for estimate in row["standard"]["estimates"]:
            assert f"{estimate['V']:.3f}" == f"{ref['V']:.3f}"
            assert f"{estimate['tau']:.3f}" == f"{ref['tau']:.3f}"
        group = "sweep" if row["case"].startswith("fixed") else "cases"
        prior = next(r for r in original[group] if r["n"] == row["n"])
        assert f"{prior['V']:.3f}" == f"{ref['V']:.3f}"
        assert f"{prior['tau']:.3f}" == f"{ref['tau']:.3f}"
        baseline_differences.append(dict(case=row["case"],
                                        V=abs(prior["V"] - ref["V"]),
                                        tau=abs(prior["tau"] - ref["tau"])))
        endpoints.append(dict(case=row["case"], time_error=err,
                              target_margin=model.margin(pol["zT"])))

    profile_checks = []
    for row in data["initial_profiles"]:
        model = num.Model(row["n"], num.BETA_FIX, num.GAMMA_FIX)
        z0 = np.array(row["z0"])
        assert abs(z0[0] - num.S0) < 1e-14 and abs(model.Y(z0) - num.Y0) < 1e-14
        ref = row["estimates"][0]
        pol = model.policy(ref["tau"], z0)
        assert abs(pol["T"] - ref["V"]) < 1e-8
        if ref["tau"] > 1e-6:
            assert pol["pre"].sol(ref["tau"])[0] > model.Sc
        profile_checks.append(dict(n=row["n"], profile=row["profile"],
                                   time_error=abs(pol["T"] - ref["V"])))

    row = data["suppression"]
    t = np.array(row["t"])
    post = t >= row["Q_entry_time"]
    assert row["initial_Y_derivative"] > 0 and row["off_peak_Y"] > num.EPS
    assert np.max(np.array(row["full_then_off_Y"])[post]) <= num.EPS + 1e-11
    assert np.max(np.array(row["full_then_off_Q"])[post]) <= num.EPS + 1e-11
    assert abs(np.max(row["off_Y"]) - row["off_peak_Y"]) < 1e-12
    report = dict(status="passed",
                  displayed_three_decimal_values_stable=True,
                  convergence_endpoint_checks=endpoints,
                  differences_from_original_illustrations=baseline_differences,
                  initial_profile_checks=profile_checks,
                  suppression_check=dict(off_peak_Y=row["off_peak_Y"],
                                         Q_entry_time=row["Q_entry_time"],
                                         after_entry_max_Y=row["after_Q_entry_max_Y"]),
                  maximum_grid_time_change=max(r["grid_delta_tau"] for r in data["convergence"]),
                  maximum_ode_time_change=max(r["tolerance_delta_tau"] for r in data["convergence"]))
    (num.OUT / "verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
