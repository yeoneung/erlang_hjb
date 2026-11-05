#!/usr/bin/env python3
"""Compare the two reported single-switch policies at identical initial states.

After the relevant target is first reached, vaccination is stopped. The script
integrates cumulative incidence and vaccination through absolute time 1000 and
records the maximum total infection on the following 300 time units. The Y-target
search is restricted to the single-switch policies and is not a global optimum.
"""
from pathlib import Path
import json, platform
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import run_numerics as num

OUT=num.OUT
SOURCE=OUT/'results.json'
data=json.loads(SOURCE.read_text())

def evaluate(row, kind, tight=False):
    model=num.Model(row['n'],row['beta'],row['gamma'])
    policy=row if kind=='Q' else row['Ytarget']
    tau=float(policy['tau'])
    hit=model.policy(tau, model.uniform_z0(), kind)
    T=float(hit['T'])
    if T <= tau: raise RuntimeError('Unexpected first entry before the switch')
    s=np.r_[model.uniform_z0(),0.,0.]
    opts=dict(method='DOP853',rtol=1e-12 if tight else 1e-10,
              atol=1e-14 if tight else 1e-12,max_step=.1 if tight else .25)
    def solve(a,b,r,y,track_peaks=False):
        base=model.rhs(r)
        def rhs(t,w):
            state=w[:-2]; return np.r_[base(t,state),model.beta*state[0]*model.Y(state),r*state[0]]
        def peak(t,w):
            state=w[:-2]; return model.beta*state[0]*model.Y(state)-model.mu*state[-1]
        peak.direction=-1
        peak.terminal=False
        return solve_ivp(rhs,(a,b),y,events=peak if track_peaks else None,**opts)
    for a,b,r in [(0.,tau,0.),(tau,T,num.R_MAX)]:
        if b>a:
            sol=solve(a,b,r,s)
            if not sol.success: raise RuntimeError(sol.message)
            s=sol.y[:,-1]
    target_state=s.copy()
    endpoint_error=np.max(np.abs(target_state[:-2]-hit['zT']))
    post=solve(T,T+300.,0.,s,True)
    if not post.success: raise RuntimeError(post.message)
    peaks=[model.Y(target_state[:-2]),model.Y(post.y[:-2,-1])]
    peaks += [model.Y(p[:-2]) for p in post.y_events[0]]
    final=solve(T+300.,1000.,0.,post.y[:,-1])
    if not final.success: raise RuntimeError(final.message)
    mass_balance_error=abs(num.S0-final.y[0,-1]-final.y[-2,-1]-final.y[-1,-1])
    if mass_balance_error > 1e-9 or endpoint_error > 1e-9:
        raise RuntimeError('Mass balance or target endpoint check failed')
    if kind == 'Q' and max(peaks) > num.EPS+1e-9:
        raise RuntimeError('Weighted-target continuation exceeded infection threshold')
    return dict(n=row['n'], beta=row['beta'], gamma=row['gamma'], target=kind, tau=tau,
                entry_time=T,Q_entry=model.Q(target_state[:-2]), Y_entry=model.Y(target_state[:-2]),
                doses_at_entry=target_state[-1], cumulative_incidence_t1000=final.y[-2,-1],
                Y_t1000=model.Y(final.y[:-2,-1]), post_entry_peak_300=max(peaks),
                integration_state_error=endpoint_error, mass_balance_error=mass_balance_error)

if __name__=='__main__':
    rows=[]
    for row in data['cases']:
        for kind in ('Q','Y'):
            a=evaluate(row,kind); b=evaluate(row,kind,True)
            a['tight_difference']={k:abs(a[k]-b[k]) for k in ('entry_time','doses_at_entry','cumulative_incidence_t1000','post_entry_peak_300')}
            rows.append(a)
            print('n=',a['n'],'target=',kind,'T=',a['entry_time'],'doses=',a['doses_at_entry'], 'incidence=',a['cumulative_incidence_t1000'],'peak=',a['post_entry_peak_300'],flush=True)
    result={'description':'Selected single-switch policies; numerical estimates, not independent global optimality proofs.',
            'absolute_final_time':1000.,'post_entry_peak_window':300.,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'rows':rows}
    (OUT/'target_outcomes.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[r'\begin{tabular}{rlrrrrr}',r'\toprule',r'$n$ & Target & $\tau$ & $T$ & $D(T)$ & $C(1000)$ & $Y_{\max}$ \\',r'\midrule']
    for r in rows:
        lines.append(f"{r['n']} & ${r['target']}$ & {r['tau']:.3f} & {r['entry_time']:.3f} & {r['doses_at_entry']:.3f} & {r['cumulative_incidence_t1000']:.3f} & {r['post_entry_peak_300']:.4f} \\\\")
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (OUT/'rendered').mkdir(parents=True, exist_ok=True)
    (OUT/'rendered'/'table_target_outcomes.tex').write_text('\n'.join(lines)+'\n')
    print('maximum tight discrepancy',max(v for r in rows for v in r['tight_difference'].values()))
