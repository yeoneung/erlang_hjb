#!/usr/bin/env python
"""Grid refinement, initial-stage sensitivity, and post-entry suppression.

Run: python -X utf8 run_validation.py --workers 4
Render saved data: python -X utf8 run_validation.py --render-only
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import platform
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

import run_numerics as num

STANDARD = dict(num.ODE)
TIGHT = dict(method="DOP853", rtol=1e-12, atol=1e-14, max_step=0.05)
GRIDS = (161, 321, 641)


def search(model, z0, grids=(641,)):
    """Refine sampled local-minimum brackets and retain both interval endpoints.

    Nested grids reuse objective samples, not prior optimization results.
    This remains a numerical search, not a proof of global minimization.
    """
    W = model.T(0.0, z0)
    largest = max(grids)
    grid = np.linspace(0.0, W, largest)
    vals = np.array([model.T(float(t), z0) for t in grid])
    estimates = []
    for size in grids:
        stride = (largest - 1) // (size - 1)
        assert stride * (size - 1) == largest - 1
        g, v = grid[::stride], vals[::stride]
        minima = [i for i in range(1, size - 1)
                  if v[i] <= v[i - 1] and v[i] <= v[i + 1]
                  and (v[i] < v[i - 1] or v[i] < v[i + 1])]
        # A constant tail is possible after uncontrolled target entry.
        # Retaining the best grid node covers a flat minimum without
        # redundantly refining every interior node of the plateau.
        # Include the best grid node even in a flat or endpoint case.
        candidates = [(float(v[0]), float(g[0])), (float(v[-1]), float(g[-1])),
                      (float(v.min()), float(g[v.argmin()]))]
        for i in minima:
            result = minimize_scalar(lambda t: model.T(float(t), z0),
                                     bounds=(g[i - 1], g[i + 1]), method="bounded",
                                     options={"xatol": 1e-9})
            if not result.success:
                raise RuntimeError(result.message)
            candidates.append((float(result.fun), float(result.x)))
        V, tau = min(candidates)
        policy = model.policy(tau, z0)
        estimates.append(dict(n_grid=size, W=W, V=V, tau=tau,
                              gain=W - V, rel_gain=(W - V) / W,
                              sampled_local_minima=len(minima),
                              terminal_margin=model.margin(policy["zT"])))
    return dict(estimates=estimates, grid=grid.tolist(), values=vals.tolist())


def convergence_job(case):
    case_id, n, beta, gamma = case
    model = num.Model(n, beta, gamma)
    z0 = model.uniform_z0()
    num.ODE = dict(STANDARD)
    standard = search(model, z0, GRIDS)
    num.ODE = dict(TIGHT)
    tight = search(model, z0)
    estimates, ref = standard["estimates"], standard["estimates"][-1]
    tightened = tight["estimates"][0]
    return dict(case=case_id, n=n, beta=beta, gamma=gamma, z0=z0.tolist(),
                standard=standard, tight=tight,
                grid_delta_V=max(abs(e["V"] - ref["V"]) for e in estimates[:-1]),
                grid_delta_tau=max(abs(e["tau"] - ref["tau"]) for e in estimates[:-1]),
                tolerance_delta_V=abs(tightened["V"] - ref["V"]),
                tolerance_delta_tau=abs(tightened["tau"] - ref["tau"]))


def profile_job(case):
    n, profile = case
    num.ODE = dict(STANDARD)
    model = num.Model(n, num.BETA_FIX, num.GAMMA_FIX)
    z0 = np.zeros(n + 1)
    z0[0] = num.S0
    if profile == "early":
        z0[1] = num.Y0
    elif profile == "late":
        z0[-1] = num.Y0
    else:
        z0[1:] = num.Y0 / n
    result = search(model, z0, (321,))
    return dict(n=n, profile=profile, z0=z0.tolist(), Q0=model.Q(z0),
                remaining_infectious_time=model.Q(z0) / model.mu, **result)


def suppression_example():
    num.ODE = dict(TIGHT)
    model = num.Model(4, num.BETA_FIX, num.GAMMA_FIX)
    z0 = np.zeros(model.n + 1)
    z0[0], z0[1] = 0.9 * model.Sc, num.EPS
    full = model.policy(0.0, z0)
    T, zT = full["T"], full["zT"]
    end = 80.0

    def peak(t, z):
        return model.beta * z[0] * model.Y(z) - model.mu * z[-1]

    peak.direction = -1
    off = solve_ivp(model.rhs(0.0), (0.0, end), z0, dense_output=True,
                    events=peak, **TIGHT)
    after = solve_ivp(model.rhs(0.0), (T, end), zT, dense_output=True,
                      events=peak, **TIGHT)
    if not off.success or not after.success or not 0 < T < end:
        raise RuntimeError("Suppression comparison integration failed")
    ts = np.unique(np.r_[np.linspace(0, end, 1601), T, off.t_events[0], after.t_events[0]])
    Zoff = off.sol(ts)
    Zsafe = np.empty_like(Zoff)
    before = ts <= T
    Zsafe[:, before] = full["post"].sol(ts[before])
    Zsafe[:, ~before] = after.sol(ts[~before])
    post_times = np.r_[T, after.t_events[0], end]
    return dict(n=model.n, beta=model.beta, gamma=model.gamma, Sc=model.Sc,
                z0=z0.tolist(), initial_Y_derivative=peak(0, z0),
                Q0=model.Q(z0), Q_entry_time=T, z_Q_entry=zT.tolist(), horizon=end,
                off_peak_Y=float(Zoff[1:].sum(axis=0).max()),
                off_peak_time=float(ts[Zoff[1:].sum(axis=0).argmax()]),
                after_Q_entry_max_Y=float(after.sol(post_times)[1:].sum(axis=0).max()),
                t=ts.tolist(), off_Y=Zoff[1:].sum(axis=0).tolist(),
                off_Q=(model.w @ Zoff[1:]).tolist(),
                full_then_off_Y=Zsafe[1:].sum(axis=0).tolist(),
                full_then_off_Q=(model.w @ Zsafe[1:]).tolist())


def render(data):
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Configuration & $\Delta_g V$ & $\Delta_g\tau$ & $\Delta_o V$ & $\Delta_o\tau$ \\",
             r"\midrule"]
    for row in data["convergence"]:
        label = ("Fixed" if row["case"].startswith("fixed") else "Selected") + rf", $n={row['n']}$"
        cells = [num.sci(row[k]) for k in ("grid_delta_V", "grid_delta_tau",
                                         "tolerance_delta_V", "tolerance_delta_tau")]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    num.write(num.OUT / "table_convergence.tex", "\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{rlrrrr}", r"\toprule",
             r"$n$ & Initial stages & $Q(z_0)$ & $\tau^*$ & $V(z_0)$ & $(W-V)/W$ \\", r"\midrule"]
    labels = dict(early="First only", uniform="Uniform", late="Last only")
    for row in data["initial_profiles"]:
        e = row["estimates"][0]
        lines.append(f"{row['n']} & {labels[row['profile']]} & {row['Q0']:.3f} & "
                     f"{e['tau']:.3f} & {e['V']:.3f} & {100*e['rel_gain']:.1f}\\%" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    num.write(num.OUT / "table_initial_profiles.tex", "\n".join(lines) + "\n")

    row = data["suppression"]
    num.plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, axes = num.plt.subplots(1, 2, figsize=(6.5, 2.65), constrained_layout=True)
    for ax, quantity, tag in zip(axes, ("Y", "Q"), ("(a)", "(b)")):
        ax.plot(row["t"], np.array(row[f"off_{quantity}"]) / num.EPS,
                color=num.ORANGE, label="Stop at initial Y-target entry", linewidth=1.5)
        ax.plot(row["t"], np.array(row[f"full_then_off_{quantity}"]) / num.EPS,
                color=num.BLUE, label="Full vaccination until Q-target", linewidth=1.5)
        ax.axhline(1, color=num.TEXT2_C, linewidth=0.7, linestyle="--")
        ax.axvline(row["Q_entry_time"], color=num.TEXT2_C, linewidth=0.8, linestyle=":")
        ax.set_xlabel("time")
        ax.set_ylabel(rf"${quantity}/\varepsilon$")
        ax.set_title(rf"{tag} $n=4$, $S_0=0.9 S_c$, $I_1(0)=\varepsilon$", fontsize=8)
        ax.set_xlim(0, row["horizon"])
        num.style_axis(ax)
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper right")
    for ext in ("pdf", "png"):
        fig.savefig(num.OUT / f"fig_target_suppression.{ext}", dpi=200)
    num.plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    path = num.OUT / "validation_results.json"
    if args.render_only:
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        started = time.time()
        fixed = [(f"fixed_n{n}", n, num.BETA_FIX, num.GAMMA_FIX) for n in (1, 2, 19)]
        selected = [(f"selected_n{n}", n, b, g) for n, b, g in num.SEARCH_CASES if n in (4, 19)]
        profiles = [(n, p) for n in (2, 9, 19) for p in ("early", "uniform", "late")]
        convergence, initial_profiles = [], []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(convergence_job, c): ("convergence", c) for c in fixed + selected}
            jobs.update({pool.submit(profile_job, c): ("profile", c) for c in profiles})
            for future in as_completed(jobs):
                kind, case = jobs[future]
                result = future.result()
                (convergence if kind == "convergence" else initial_profiles).append(result)
                print(f"completed {kind}: {case}", flush=True)
        convergence.sort(key=lambda r: [c[0] for c in fixed + selected].index(r["case"]))
        initial_profiles.sort(key=lambda r: profiles.index((r["n"], r["profile"])))
        data = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    environment=dict(python=platform.python_version(), numpy=np.__version__,
                                     scipy=scipy.__version__, matplotlib=num.matplotlib.__version__),
                    setting=dict(r_max=num.R_MAX, eps=num.EPS, S0=num.S0, Y0=num.Y0,
                                 beta_fixed=num.BETA_FIX, gamma_fixed=num.GAMMA_FIX,
                                 standard_ode=STANDARD, tight_ode=TIGHT, grids=list(GRIDS),
                                 profile_grid=321, optimizer_xatol=1e-9,
                                 post_switch_horizon=num.HORIZON),
                    convergence=convergence, initial_profiles=initial_profiles,
                    suppression=suppression_example(), seconds=time.time() - started)
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    render(data)
    print("All additional experiments rendered; saved results.json is unchanged.")


if __name__ == "__main__":
    main()
