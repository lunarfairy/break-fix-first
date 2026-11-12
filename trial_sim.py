"""
trial_sim.py (revision 2) - Monte Carlo design analysis for the two-arm cluster-randomised trial
(break-fix-first productive failure, PS-I, vs instruction-first, I-PS, with the same faults).

Data-generating model (per replicate)
  J tutorial groups in strata of four (two per arm; 7 strata for J = 28); group sizes m_j equal (25) or
  gamma-distributed with coefficient of variation cv (mean 25, clipped to 8..45).
  stratum effect s_k ~ N(0, tau_s);  cluster effect u_j ~ N(0, rho);  individual e_ij ~ N(0, 1 - rho)
  untreated latent skill  th0 = s_k + u_j + e_ij ;  theta = th0 + delta*T_j + gamma*T_j*(Z_ij - 0.5)
  pre-test x_ij = r*(b*u_j + e_ij)/sqrt(b^2*rho + 1 - rho) + sqrt(1 - r^2)*N(0,1); b = 1 gives the same
  predictive R^2 = r^2 at both levels, b = 0 a pre-test with no between-group information.
  practical: K items per student drawn as a balanced form from a pool of 13 targeted and 6 untargeted items
  (K=8: 5 targeted + 3 untargeted), item difficulties b_i ~ N(0,1) fixed per replicate, discrimination a,
  optional lower asymptote c (lenient, resource-level scoring);
  P(correct) = c + (1-c)*logistic(a*(theta + eta*T_j*targeted_i - b_i)).  score = mean of K responses.
  missingness: MCAR at rate attr, or MNAR with P(missing) = logistic(a0 + a1*(-th0)) calibrated to attr,
  optionally differential (slope and rate higher in the PS-I arm); optional loss of whole groups.
Primary analysis (pre-specified): ITT, cluster-average estimand. Pooled within-group slope of score on
  pre-test; adjusted group means regressed on arm and stratum indicators (OLS, df = J - strata - 1);
  two-sided alpha = .05. Alternatives reported for comparison: unstratified two-sample t (df = J - 2) and
  the same model weighted by analysed group size.
H2: arm difference in the adjusted within-student contrast (targeted minus untargeted proportion), same model.
H3: within-group difference (prerequisite Z=1 minus Z=0, adjusted) on arm with stratum indicators;
  reported at alpha = .05 and at the Holm first-step level .05/3.
H5 (non-inferiority of project grade): separate outcome with ICC rho_g, pre-test R^2 0.3; one-sided 97.5%
  lower confidence bound of the adjusted difference (in student-level SD units) compared with -margin.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)
N_TARG, N_UNT = 13, 6


def _strata_fit(yc, T, strat, w=None):
    """OLS of cluster-level y on arm + stratum dummies; returns (estimate, se, df)."""
    J = len(yc)
    ks = np.unique(strat)
    X = np.column_stack([T] + [(strat == k).astype(float) for k in ks])
    if w is None:
        w = np.ones(J)
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], yc * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    res = yw - Xw @ beta
    df = J - X.shape[1]
    s2 = res @ res / df
    cov = s2 * np.linalg.inv(Xw.T @ Xw)
    return beta[0], np.sqrt(cov[0, 0]), df


def simulate(J=28, m=25, cv=0.0, rho=0.10, tau_s=0.02, delta=0.3, r2=0.3, b_between=1.0, K=8, a=1.2, c=0.0,
             attr=0.15, mnar=0.0, diff_attr=False, lose=0, gamma=0.0, eta=0.0, reps=1000, seed=1,
             rel=False, h5=False, rho_g=0.10, true_g=0.0, margin=0.2):
    rng = np.random.default_rng(seed)
    r = np.sqrt(r2)
    nstr = J // 4
    strat = np.repeat(np.arange(nstr), 4)
    k_t = int(round(K * N_TARG / (N_TARG + N_UNT)))
    k_u = K - k_t
    acc = dict(rej=0, rej_t=0, rej_w=0, rej_h2=0, rej_h3=0, rej_h3_holm=0, ni=0)
    d_obs, est, rels, iccs = [], [], [], []
    for rep in range(reps):
        T = np.zeros(J)
        for k in range(nstr):
            idx = np.where(strat == k)[0]
            T[rng.permutation(idx)[:2]] = 1
        if cv > 0:
            mj = np.clip(np.round(m * rng.gamma(1 / cv ** 2, cv ** 2, J)), 8, 45).astype(int)
        else:
            mj = np.full(J, m)
        M = mj.max()
        valid = np.arange(M)[None, :] < mj[:, None]
        s_eff = rng.normal(0, np.sqrt(tau_s), nstr)[strat] if tau_s > 0 else np.zeros(J)
        u = rng.normal(0, np.sqrt(rho), J)
        e = rng.normal(0, np.sqrt(1 - rho), (J, M))
        th0 = s_eff[:, None] + u[:, None] + e
        Z = (rng.random((J, M)) < 0.5).astype(float)
        theta = th0 + delta * T[:, None] + gamma * T[:, None] * (Z - 0.5)
        xs = (b_between * u[:, None] + e) / np.sqrt(b_between ** 2 * rho + 1 - rho)
        x = r * xs + np.sqrt(1 - r2) * rng.normal(size=(J, M))
        bt = rng.normal(0, 1, N_TARG)
        bu = rng.normal(0, 1, N_UNT)
        it = np.argsort(rng.random((J, M, N_TARG)), axis=2)[:, :, :k_t]
        iu = np.argsort(rng.random((J, M, N_UNT)), axis=2)[:, :, :k_u]
        pt = c + (1 - c) / (1 + np.exp(-a * (theta[:, :, None] + eta * T[:, None, None] - bt[it])))
        pu = c + (1 - c) / (1 + np.exp(-a * (theta[:, :, None] - bu[iu])))
        rt = (rng.random(pt.shape) < pt).mean(axis=2)
        ru = (rng.random(pu.shape) < pu).mean(axis=2)
        y = (rt * k_t + ru * k_u) / K
        ydiff = rt - ru
        # missingness
        if mnar > 0:
            a1 = np.where(T[:, None] == 1, mnar * (1.5 if diff_attr else 1.0), mnar)
            rate = np.where(T[:, None] == 1, attr + (0.05 if diff_attr else 0.0), attr)
            z = -(th0 - th0[valid].mean()) / th0[valid].std()
            lin = a1 * z
            # calibrate intercept per arm so that the expected missing share equals `rate`
            a0 = np.zeros((J, 1))
            for arm in (0, 1):
                sel = (T == arm)
                lo, hi = -10.0, 10.0
                target = rate[sel][0, 0] if sel.any() else attr
                for _ in range(40):
                    mid = (lo + hi) / 2
                    pm = 1 / (1 + np.exp(-(mid + lin[sel][valid[sel]])))
                    lo, hi = (mid, hi) if pm.mean() < target else (lo, mid)
                a0[sel] = (lo + hi) / 2
            miss = rng.random((J, M)) < 1 / (1 + np.exp(-(a0 + lin)))
        else:
            miss = rng.random((J, M)) < attr
        keep = valid & ~miss
        keepc = np.ones(J, bool)
        if lose > 0:
            for arm in (0, 1):
                keepc[rng.choice(np.where(T == arm)[0], lose, replace=False)] = False
        keep &= keepc[:, None]
        cl = np.where(keepc & (keep.sum(axis=1) >= 2))[0]
        xm = np.array([x[j, keep[j]].mean() for j in cl])
        ym = np.array([y[j, keep[j]].mean() for j in cl])
        xc = np.concatenate([x[j, keep[j]] - xm[i] for i, j in enumerate(cl)])
        yc_ = np.concatenate([y[j, keep[j]] - ym[i] for i, j in enumerate(cl)])
        beta = (xc @ yc_) / (xc @ xc) if r2 > 0 else 0.0
        adj = ym - beta * xm
        Tc, sc = T[cl], strat[cl]
        b1, se1, df1 = _strata_fit(adj, Tc, sc)
        p1 = 2 * stats.t.sf(abs(b1 / se1), df1)
        acc['rej'] += p1 < 0.05
        tt = stats.ttest_ind(adj[Tc == 1], adj[Tc == 0])
        acc['rej_t'] += tt.pvalue < 0.05
        bw, sew, dfw = _strata_fit(adj, Tc, sc, w=keep[cl].sum(axis=1).astype(float))
        acc['rej_w'] += 2 * stats.t.sf(abs(bw / sew), dfw) < 0.05
        sd_y = np.concatenate([y[j, keep[j]] for j in cl if T[j] == 0]).std(ddof=1)
        d_obs.append(b1 / sd_y)
        est.append(b1)
        # H2
        dm = np.array([ydiff[j, keep[j]].mean() for j in cl])
        b2, se2, df2 = _strata_fit(dm, Tc, sc)
        p2 = 2 * stats.t.sf(abs(b2 / se2), df2)
        acc['rej_h2'] += p2 < 0.05
        # H3
        if gamma != 0:
            dif = []
            for j in cl:
                k1 = keep[j] & (Z[j] == 1)
                k0 = keep[j] & (Z[j] == 0)
                dif.append((y[j, k1] - beta * x[j, k1]).mean() - (y[j, k0] - beta * x[j, k0]).mean())
            b3, se3, df3 = _strata_fit(np.array(dif), Tc, sc)
            p3 = 2 * stats.t.sf(abs(b3 / se3), df3)
            acc['rej_h3'] += p3 < 0.05
            acc['rej_h3_holm'] += p3 < 0.05 / 3
        if rel and rep < 50:
            ctrl = (T[:, None] == 0) & valid
            rels.append(np.corrcoef(theta[ctrl], y[ctrl])[0, 1] ** 2)
            # observed-score ICC (one-way ANOVA estimator) in the control arm
            jj = np.where(T == 0)[0]
            g = [y[j, valid[j]] for j in jj]
            n0 = np.mean([len(v) for v in g])
            msb = n0 * np.var([v.mean() for v in g], ddof=1)
            msw = np.mean([v.var(ddof=1) for v in g])
            iccs.append((msb - msw) / (msb + (n0 - 1) * msw))
        if h5:
            ug = rng.normal(0, np.sqrt(rho_g), J)
            eg = rng.normal(0, np.sqrt(1 - rho_g), (J, M))
            g0 = ug[:, None] + eg
            gx = r * g0 + np.sqrt(1 - r2) * rng.normal(size=(J, M))
            G = g0 + true_g * T[:, None]
            gm = np.array([G[j, keep[j]].mean() for j in cl])
            gxm = np.array([gx[j, keep[j]].mean() for j in cl])
            gxc = np.concatenate([gx[j, keep[j]] - gxm[i] for i, j in enumerate(cl)])
            gyc = np.concatenate([G[j, keep[j]] - gm[i] for i, j in enumerate(cl)])
            bg = (gxc @ gyc) / (gxc @ gxc)
            b5, se5, df5 = _strata_fit(gm - bg * gxm, Tc, sc)
            sdg = np.concatenate([G[j, keep[j]] for j in cl if T[j] == 0]).std(ddof=1)
            lcb = (b5 - stats.t.ppf(0.975, df5) * se5) / sdg
            acc['ni'] += lcb > -margin
    out = dict(power=acc['rej'] / reps, power_unstratified_t=acc['rej_t'] / reps, power_size_weighted=acc['rej_w'] / reps,
               power_h2=acc['rej_h2'] / reps, d_observed=float(np.mean(d_obs)), est_mean=float(np.mean(est)),
               mc_se=float(np.sqrt(max(acc['rej'], 1) / reps * (1 - acc['rej'] / reps) / reps)))
    if gamma != 0:
        out['power_h3'] = acc['rej_h3'] / reps
        out['power_h3_holm_first_step'] = acc['rej_h3_holm'] / reps
    if rel:
        out['rel'] = float(np.mean(rels))
        out['icc_observed'] = float(np.mean(iccs))
    if h5:
        out['power_noninferiority'] = acc['ni'] / reps
    return out


def mdes_analytic(J, m, rho, r2, rel=1.0, n_strata=7, P=0.5, alpha=0.05, power=0.8):
    """MDES on the latent scale. rel < 1 adds measurement error of the practical (latent-SD units:
    error variance (1-rel)/rel per student, not explained by the pre-test)."""
    df = J - n_strata - 1
    Mlt = stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)
    within = (1 - rho) * (1 - r2) + (1 - rel) / rel
    return Mlt * np.sqrt(rho * (1 - r2) / (P * (1 - P) * J) + within / (P * (1 - P) * J * m))


def jobs():
    J = []
    sd = 1000

    def add(kind, **kw):
        nonlocal sd
        sd += 1
        kw.setdefault('reps', 1000)
        J.append((kind, dict(kw, seed=sd)))

    for rho in [0.05, 0.10, 0.20]:
        for r2 in [0.0, 0.3, 0.5]:
            for K in [4, 8, 12]:
                for delta in [0.0, 0.2, 0.3, 0.4, 0.5]:
                    add('power', rho=rho, r2=r2, K=K, delta=delta)
    for delta in [0.0, 0.2, 0.3, 0.4, 0.5]:
        add('headline', rho=0.10, r2=0.3, K=8, delta=delta, reps=4000)
    for delta in [0.3, 0.4]:
        base = dict(rho=0.10, r2=0.3, K=8, delta=delta, reps=2000)
        for lab, kw in [('unequal sizes (cv 0.3)', dict(cv=0.3)), ('pre-test without between-group information', dict(b_between=0.0)),
                        ('MNAR attrition, non-differential', dict(mnar=1.0)), ('MNAR attrition, differential', dict(mnar=1.0, diff_attr=True)),
                        ('one group lost per arm', dict(lose=1)), ('discrimination a = 0.8', dict(a=0.8)), ('discrimination a = 1.6', dict(a=1.6)),
                        ('lenient scoring (c = 0.15)', dict(c=0.15)), ('no attrition', dict(attr=0.0)), ('30% attrition', dict(attr=0.30)),
                        ('larger stratum variance (0.05)', dict(tau_s=0.05))]:
            J.append(('robust', dict(base, label=lab, seed=5000 + int(delta * 10), **kw)))
    for Jn in [20, 28, 40, 56]:
        for delta in [0.2, 0.3, 0.4]:
            add('clusters', J=Jn, rho=0.10, r2=0.3, K=8, delta=delta, reps=2000)
    for eta in [0.0, 0.2, 0.3, 0.4, 0.5]:
        add('h2', rho=0.10, r2=0.3, K=8, delta=0.3, eta=eta, reps=2000)
    for rho in [0.05, 0.10, 0.20]:
        for gamma in [0.2, 0.3, 0.4, 0.5]:
            add('moderator', rho=rho, r2=0.3, K=8, delta=0.3, gamma=gamma, reps=2000)
    for rho_g in [0.05, 0.10, 0.20]:
        for tg in [0.0, -0.1]:
            add('noninferiority', rho=0.10, r2=0.3, K=8, delta=0.3, h5=True, rho_g=rho_g, true_g=tg, reps=2000)
    for K in [2, 4, 6, 8, 10, 12, 16]:
        for a in [0.8, 1.2, 1.6]:
            add('reliability', K=K, a=a, delta=0.0, reps=50, rel=True)
    return J


def _cell(job):
    kind, kw = job
    lab = kw.pop('label', '')
    res = simulate(**kw)
    out = {k: v for k, v in kw.items() if k not in ('reps', 'seed', 'rel', 'h5')}
    out.update({k: float(v) for k, v in res.items()})
    out['kind'], out['label'], out['reps'] = kind, lab, kw['reps']
    return out


if __name__ == '__main__':
    JOBS = jobs()
    if len(sys.argv) >= 3:
        i, n = int(sys.argv[1]), int(sys.argv[2])
        pd.DataFrame([_cell(j) for j in JOBS[i::n]]).to_csv(os.path.join(OUT, 'trial_part_%02d.csv' % i), index=False)
        sys.exit(0)
    import glob
    if len(sys.argv) == 2 and sys.argv[1] == 'merge':
        R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, 'trial_part_*.csv')))])
    else:
        R = pd.DataFrame([_cell(j) for j in JOBS])
    for kind in R.kind.unique():
        R[R.kind == kind].dropna(axis=1, how='all').to_csv(os.path.join(OUT, 'trial_%s.csv' % kind), index=False)
    rel8 = float(R[(R.kind == 'reliability') & (R.K == 8) & (R.a == 1.2)].rel.iloc[0])
    an = []
    for rho in [0.05, 0.10, 0.20]:
        for r2 in [0.0, 0.3, 0.5]:
            an.append(dict(rho=rho, r2=r2, rel=rel8, mdes_latent_perfect_measure=mdes_analytic(28, 25 * 0.85, rho, r2),
                           mdes_latent_with_measurement_error=mdes_analytic(28, 25 * 0.85, rho, r2, rel=rel8)))
    pd.DataFrame(an).to_csv(os.path.join(OUT, 'trial_mdes_analytic.csv'), index=False)
    print('done', len(R))
