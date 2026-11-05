#!/usr/bin/env python
"""Numerical illustration of the single-switch theory for the sustained-suppression
target G_eps = {S <= S_c, Q <= eps}, Q = sum_j (n-j+1) I_j.

Calculations use single-switch ODE evaluation.
Before uncontrolled target entry, T(tau) = tau + W(Phi^0_tau(z0)); after
that entry, the stopped policy cost is the uncontrolled hitting time.
The minimum over 0 <= tau <= W(z0) is V(z0) in either representation.
The script writes JSON results, LaTeX tables and PDF figures to output/.

Run from anywhere:   python -X utf8 run_numerics.py
Runtime: a few minutes on a laptop (CPU only; numpy, scipy, matplotlib).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- setting
R_MAX, EPS, S0, Y0 = 0.03, 0.01, 0.99, 0.01
HORIZON = 600.0
N_GRID = 161
ODE = dict(rtol=1e-10, atol=1e-12, max_step=0.1, method="DOP853")
ADJ = dict(rtol=1e-11, atol=1e-13, max_step=0.05, method="DOP853")

# (n, beta, gamma): selected parameter sets with uniform initial stages.
SEARCH_CASES = [
    (2, 0.4163625240325928, 0.11758241057395935),
    (4, 0.2922666668891907, 0.06820093095302582),
    (9, 0.2720777988433838, 0.05967894196510315),
    (19, 0.26834726333618164, 0.05442754924297333),
]
BETA_FIX, GAMMA_FIX = SEARCH_CASES[0][1], SEARCH_CASES[0][2]
SWEEP_N = [1, 2, 3, 4, 6, 9, 14, 19]
SWEEP_PLOT_N = [1, 2, 4, 9, 19]

# Plot colors.
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
BLUE, ORANGE = "#2a78d6", "#eb6834"
GRID_C, TEXT_C, TEXT2_C = "#e6e5e1", "#0b0b0b", "#52514e"


class Model:
    def __init__(self, n: int, beta: float, gamma: float):
        self.n, self.beta, self.gamma = n, beta, gamma
        self.mu = n * gamma
        self.Sc = gamma / beta
        self.w = np.arange(n, 0, -1, dtype=float)

    def rhs(self, r: float):
        n, beta, mu = self.n, self.beta, self.mu

        def f(t, z):
            S, I = z[0], z[1:]
            Y = I.sum()
            dz = np.empty(n + 1)
            dz[0] = -beta * S * Y - r * S
            dz[1] = beta * S * Y - mu * I[0]
            dz[2:] = mu * (I[:-1] - I[1:])
            return dz

        return f

    def Q(self, z) -> float:
        return float(self.w @ z[1:])

    def Y(self, z) -> float:
        return float(z[1:].sum())

    def margin(self, z, kind: str = "Q") -> float:
        burden = self.Q(z) if kind == "Q" else self.Y(z)
        return max(z[0] - self.Sc, burden - EPS)

    def event(self, kind: str = "Q"):
        def g(t, z):
            return self.margin(z, kind)

        g.terminal = True
        g.direction = -1
        return g

    def uniform_z0(self):
        return np.concatenate([[S0], np.full(self.n, Y0 / self.n)])

    def policy(self, tau: float, z0, kind: str = "Q") -> dict:
        """Trajectory of the single-switch policy r_tau (0 before tau, R_MAX after)."""
        ev = self.event(kind)
        if self.margin(z0, kind) <= 0:
            return dict(T=0.0, pre=None, post=None, zT=np.array(z0))
        pre, z, t0 = None, np.array(z0), 0.0
        if tau > 0:
            pre = solve_ivp(self.rhs(0.0), (0.0, tau), z0, events=ev, dense_output=True, **ODE)
            if not pre.success:
                raise RuntimeError(pre.message)
            if pre.t_events[0].size:  # target reached before the switch
                return dict(T=float(pre.t_events[0][0]), pre=pre, post=None, zT=pre.y_events[0][0])
            z, t0 = pre.y[:, -1], tau
        post = solve_ivp(self.rhs(R_MAX), (t0, t0 + HORIZON), z, events=ev, dense_output=True, **ODE)
        if not post.success:
            raise RuntimeError(post.message)
        if not post.t_events[0].size:
            raise RuntimeError("full vaccination did not reach the target within the horizon")
        return dict(T=float(post.t_events[0][0]), pre=pre, post=post, zT=post.y_events[0][0])

    def T(self, tau: float, z0, kind: str = "Q") -> float:
        return self.policy(tau, z0, kind)["T"]


def minimize_switch(model: Model, z0, kind: str = "Q") -> dict:
    """Scalar minimization of T(tau) on [0, W(z0)]."""
    W = model.T(0.0, z0, kind)
    grid = np.linspace(0.0, W, N_GRID)
    vals = np.array([model.T(t, z0, kind) for t in grid])
    k = int(np.argmin(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, N_GRID - 1)]
    res = minimize_scalar(lambda t: model.T(t, z0, kind), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-8})
    if res.fun <= vals[k]:
        tau, V = float(res.x), float(res.fun)
    else:
        tau, V = float(grid[k]), float(vals[k])
    interior = [i for i in range(1, N_GRID - 1) if vals[i] < vals[i - 1] and vals[i] < vals[i + 1]]
    return dict(W=W, tau=tau, V=V, gain=W - V, rel_gain=(W - V) / W,
                grid=grid.tolist(), vals=vals.tolist(), n_interior_grid_minima=len(interior))


def after_entry(model: Model, pol: dict, horizon: float = 300.0) -> dict:
    """Stop vaccination at target entry and follow the uncontrolled flow."""
    T, zT = pol["T"], pol["zT"]
    sol = solve_ivp(model.rhs(0.0), (T, T + horizon), zT, rtol=1e-9, atol=1e-12, max_step=0.05,
                    method="DOP853")
    Y = sol.y[1:].sum(axis=0)
    Q = model.w @ sol.y[1:]
    return dict(max_Y=float(Y.max()), max_Q=float(Q.max()))


def pmp_check(model: Model, tau: float, z0, n_samples: int = 20001) -> dict:
    """Integrate the adjoint system backward from the Q-face terminal condition
    (lambda_S = 0, lambda_Q = 1) along the optimal single-switch trajectory and
    check the adjoint signs and switching identity."""
    pol = model.policy(tau, z0, "Q")
    T, pre, post = pol["T"], pol["pre"], pol["post"]
    n, beta, mu, w = model.n, model.beta, model.mu, model.w

    def state(t):
        if pre is not None and t <= tau:
            return pre.sol(t)
        return post.sol(min(max(t, tau), T))

    def rate(t):
        return 0.0 if t < tau else R_MAX

    def adj(t, p):
        z = state(t)
        S, I = z[0], z[1:]
        Y = I.sum()
        r = rate(t)
        pS, pI = p[0], p[1:]
        dp = np.empty(n + 1)
        dp[0] = (beta * Y + r) * pS - beta * Y * pI[0]
        dp[1:] = beta * S * (pS - pI[0]) + mu * (pI - np.append(pI[1:], 0.0))
        return dp

    pT = np.concatenate([[0.0], w])
    sol = solve_ivp(adj, (T, 0.0), pT, dense_output=True, **ADJ)
    ts = np.linspace(0.0, T, n_samples)
    P = sol.sol(ts)
    Z = np.array([state(t) for t in ts]).T
    psi = -Z[0] * P[0]
    p1 = P[1]
    zeros = []
    for k in np.where(np.diff(np.sign(psi)) != 0)[0]:
        zeros.append(float(brentq(lambda t: -state(t)[0] * sol.sol(t)[0], ts[k], ts[k + 1], xtol=1e-12)))
    p1_zeros = []
    for k in np.where(np.diff(np.sign(p1)) != 0)[0]:
        p1_zeros.append(float(brentq(lambda t: sol.sol(t)[1], ts[k], ts[k + 1], xtol=1e-12)))
    p0 = np.array([-(sol.sol(t) @ model.rhs(rate(t))(t, state(t))) for t in ts[::200]])
    info = dict(T=T, S_T_over_Sc=float(pol["zT"][0] / model.Sc), Q_T=model.Q(pol["zT"]),
                psi_zeros=zeros, p1_zeros=p1_zeros,
                psi_positive_before_tau=float(np.mean(psi[ts < tau] > 0)),
                psi_negative_after_tau=float(np.mean(psi[ts > tau] < 0)),
                p0_mean=float(p0.mean()), p0_rel_spread=float((p0.max() - p0.min()) / abs(p0.mean())),
                series=dict(t=ts[::10].tolist(), psi=psi[::10].tolist(), p1=p1[::10].tolist()))
    interior = [z for z in zeros if z < T - 1e-6]
    if interior:
        tz = interior[0]
        p, z = sol.sol(tz), state(tz)
        d = adj(tz, p)[1:]
        info.update(zero_minus_tau=tz - tau, p1_at_switch=float(p[1]), min_dpj_at_switch=float(d.min()),
                    all_dpj_positive=bool(np.all(d > 0)), S_switch_over_Sc=float(z[0] / model.Sc),
                    identity_residual=float(d.sum() - n * (model.gamma - beta * z[0]) * p[1]))
    return info


# ----------------------------------------------------------------------------- formatting helpers
def f3(x):
    return f"{x:.3f}"


def sci(x, digits=1):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return rf"${m:.{digits}f}\times10^{{{e}}}$"


def write(path: pathlib.Path, text: str):
    path.write_text(text, encoding="utf-8")
    print("wrote", path.relative_to(HERE.parent))


# ----------------------------------------------------------------------------- experiments
def run_sweep() -> list[dict]:
    rows = []
    for n in SWEEP_N:
        t0 = time.time()
        m = Model(n, BETA_FIX, GAMMA_FIX)
        z0 = m.uniform_z0()
        r = minimize_switch(m, z0, "Q")
        pol = m.policy(r["tau"], z0, "Q")
        z_tau = pol["pre"].y[:, -1] if pol["pre"] is not None else z0
        r.update(n=n, Q0=m.Q(z0), S_switch_over_Sc=float(z_tau[0] / m.Sc), after=after_entry(m, pol),
                 pmp=pmp_check(m, r["tau"], z0) if n in SWEEP_PLOT_N else None)
        rows.append(r)
        print(f"sweep n={n:2d}: W={r['W']:.4f} tau*={r['tau']:.4f} V={r['V']:.4f} "
              f"gain={r['gain']:.4f} ({100*r['rel_gain']:.2f}%) S/Sc={r['S_switch_over_Sc']:.3f} "
              f"[{time.time()-t0:.0f}s]")
    return rows


def run_cases() -> list[dict]:
    rows = []
    for n, beta, gamma in SEARCH_CASES:
        t0 = time.time()
        m = Model(n, beta, gamma)
        z0 = m.uniform_z0()
        r = minimize_switch(m, z0, "Q")
        pol = m.policy(r["tau"], z0, "Q")
        z_tau = pol["pre"].y[:, -1] if pol["pre"] is not None else z0
        r.update(n=n, beta=beta, gamma=gamma, R0=beta * S0 / gamma, Q0=m.Q(z0),
                 S_switch_over_Sc=float(z_tau[0] / m.Sc), after=after_entry(m, pol),
                 pmp=pmp_check(m, r["tau"], z0))
        # the non-invariant set {S <= S_c, Y <= eps} of Example ex:old_target
        ry = minimize_switch(m, z0, "Y")
        poly = m.policy(ry["tau"], z0, "Y")
        ry.update(Q_at_entry=m.Q(poly["zT"]), after=after_entry(m, poly))
        r["Ytarget"] = ry
        rows.append(r)
        print(f"case n={n:2d}: W={r['W']:.4f} tau*={r['tau']:.4f} V={r['V']:.4f} gain={r['gain']:.4f} "
              f"| Y-set: W={ry['W']:.4f} tau={ry['tau']:.4f} V={ry['V']:.4f} Q_entry={ry['Q_at_entry']:.4f} "
              f"[{time.time()-t0:.0f}s]")
    return rows


# ----------------------------------------------------------------------------- tables
def table_sweep(rows):
    lines = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
             r"$n$ & $Q(z_0)$ & $W(z_0)$ & $\tau^*$ & $V(z_0)$ & $W-V$ & $(W-V)/W$ & "
             r"$S(\tau^*)/S_c$ & $\max Y$ after entry \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['n']} & {r['Q0']:.3f} & {f3(r['W'])} & {f3(r['tau'])} & {f3(r['V'])} & "
                     f"{f3(r['gain'])} & {100*r['rel_gain']:.1f}\\% & {r['S_switch_over_Sc']:.2f} & "
                     f"{r['after']['max_Y']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_cases(rows):
    lines = [r"\begin{tabular}{rrrrrrrrrr}", r"\toprule",
             r"$n$ & $\beta$ & $\gamma$ & $\beta S_0/\gamma$ & $W(z_0)$ & $\tau^*$ & $V(z_0)$ & "
             r"$W-V$ & $(W-V)/W$ & $S(\tau^*)/S_c$ \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['n']} & {r['beta']:.4f} & {r['gamma']:.4f} & {r['R0']:.2f} & {f3(r['W'])} & "
                     f"{f3(r['tau'])} & {f3(r['V'])} & {f3(r['gain'])} & {100*r['rel_gain']:.1f}\\% & "
                     f"{r['S_switch_over_Sc']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_pmp(rows):
    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$n$ & $|t_\psi-\tau^*|$ & $p_1(t_\psi)$ & $\min_j\dot p_j(t_\psi)$ & "
             r"identity residual & $p_0$ spread & $S(T)/S_c$ \\", r"\midrule"]
    for r in rows:
        q = r["pmp"]
        lines.append(f"{r['n']} & {sci(abs(q['zero_minus_tau']))} & {sci(q['p1_at_switch'], 2)} & "
                     f"{sci(q['min_dpj_at_switch'], 2)} & {sci(abs(q['identity_residual']))} & "
                     f"{sci(q['p0_rel_spread'])} & {q['S_T_over_Sc']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_ytarget(rows):
    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$n$ & $\widetilde W(z_0)$ & $\widetilde\tau$ & $\widetilde V_{01}(z_0)$ & "
             r"$Q$ at entry & $\max Y$ after entry & $W(z_0)$ \\", r"\midrule"]
    for r in rows:
        y = r["Ytarget"]
        lines.append(f"{r['n']} & {f3(y['W'])} & {f3(y['tau'])} & {f3(y['V'])} & {y['Q_at_entry']:.4f} & "
                     f"{y['after']['max_Y']:.4f} & {f3(r['W'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- figures
def style_axis(ax):
    ax.grid(True, color=GRID_C, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TEXT2_C)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=TEXT2_C, labelsize=8, width=0.6)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)


def fig_switch_curves(sweep, cases):
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), constrained_layout=True)
    ax = axes[0]
    for c, n in zip(BLUE_RAMP, SWEEP_PLOT_N):
        r = next(x for x in sweep if x["n"] == n)
        g, v = np.array(r["grid"]), np.array(r["vals"]) - r["W"]
        ax.plot(g, v, color=c, linewidth=1.5, label=rf"$n={n}$")
        ax.plot([r["tau"]], [r["V"] - r["W"]], "o", color=c, markersize=4, markeredgecolor="white",
                markeredgewidth=0.6)
    ax.axhline(0, color=TEXT2_C, linewidth=0.6)
    ax.set_xlabel(r"switching time $\tau$")
    ax.set_ylabel(r"$T(\tau)-W(z_0)$")
    ax.set_title(rf"(a) fixed $\beta={BETA_FIX:.3f}$, $\gamma={GAMMA_FIX:.4f}$", fontsize=8, color=TEXT_C)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    style_axis(ax)
    ax = axes[1]
    ramp4 = [BLUE_RAMP[0], BLUE_RAMP[2], BLUE_RAMP[3], BLUE_RAMP[4]]
    for c, r in zip(ramp4, cases):
        g, v = np.array(r["grid"]), np.array(r["vals"]) - r["W"]
        ax.plot(g, v, color=c, linewidth=1.5, label=rf"$n={r['n']}$")
        ax.plot([r["tau"]], [r["V"] - r["W"]], "o", color=c, markersize=4, markeredgecolor="white",
                markeredgewidth=0.6)
    ax.axhline(0, color=TEXT2_C, linewidth=0.6)
    ax.set_xlabel(r"switching time $\tau$")
    ax.set_title(r"(b) selected parameter sets", fontsize=8, color=TEXT_C)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    style_axis(ax)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_switch_curves.{ext}", dpi=200)
    plt.close(fig)


def fig_pmp(cases):
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
    sel = [cases[0], cases[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), constrained_layout=True)
    for ax, r, tag in zip(axes, sel, ("(a)", "(b)")):
        q = r["pmp"]
        t = np.array(q["series"]["t"])
        psi = np.array(q["series"]["psi"])
        p1 = np.array(q["series"]["p1"])
        ax.plot(t, psi / np.abs(psi).max(), color=BLUE, linewidth=1.5, label=r"$\psi=-Sp_S$")
        ax.plot(t, p1 / np.abs(p1).max(), color=ORANGE, linewidth=1.5, label=r"$p_1$")
        ax.axhline(0, color=TEXT2_C, linewidth=0.6)
        ax.axvline(r["tau"], color=TEXT2_C, linewidth=0.8, linestyle=":")
        ax.text(r["tau"] + 0.012 * q["T"], 1.0, r"$\tau^*$", ha="left", va="bottom", fontsize=8,
                color=TEXT_C)
        ax.set_xlabel(r"$t$")
        ax.set_title(rf"{tag} $n={r['n']}$, $T={q['T']:.2f}$", fontsize=8, color=TEXT_C)
        ax.set_ylim(-1.15, 1.15)
        style_axis(ax)
    axes[0].set_ylabel("normalized adjoint quantities")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_pmp.{ext}", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only', action='store_true',
                        help='regenerate tables and figures from the supplied results.json')
    args = parser.parse_args()
    t0 = time.time()
    if args.render_only:
        results = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
        sweep, cases = results['sweep'], results['cases']
        print('Rendering supplied numerical data; ODE experiments are not rerun.')
    else:
        sweep = run_sweep()
        cases = run_cases()
        results = dict(setting=dict(r_max=R_MAX, eps=EPS, S0=S0, Y0=Y0, ode=ODE, adjoint=ADJ, n_grid=N_GRID,
                                    horizon=HORIZON, beta_fixed=BETA_FIX, gamma_fixed=GAMMA_FIX),
                       sweep=sweep, cases=cases, seconds=time.time() - t0)
        write(OUT / "results.json", json.dumps(results, indent=1))
    write(OUT / "table_sweep.tex", table_sweep(sweep))
    write(OUT / "table_cases.tex", table_cases(cases))
    write(OUT / "table_pmp.tex", table_pmp(cases))
    write(OUT / "table_ytarget.tex", table_ytarget(cases))
    fig_switch_curves(sweep, cases)
    fig_pmp(cases)
    print(f"done in {time.time()-t0:.0f}s")
    # consistency summary
    for r in sweep + cases:
        assert r["n_interior_grid_minima"] <= 1, ("multiple grid minima", r["n"])
    for r in cases + [x for x in sweep if x["pmp"]]:
        q = r["pmp"]
        assert len([z for z in q["psi_zeros"] if z < q["T"] - 1e-6]) == 1, ("psi zeros", r["n"], q["psi_zeros"])
        assert q["all_dpj_positive"] and q["p1_at_switch"] < 0 and q["S_switch_over_Sc"] > 1
    print("all sign checks passed")


if __name__ == "__main__":
    main()
