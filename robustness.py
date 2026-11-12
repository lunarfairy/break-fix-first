"""
robustness.py - revision-2 robustness and sensitivity analyses of the model-based results.
Deterministic (seeded). Reads nothing from results/ except what run_analyses.py wrote; writes
results/robust_*.csv and results/robust_summary.json.
Analyses
  R1 ambiguity distribution by evidence tier (median, quartiles, maximum of class size per mutant)
  R2 cost breakdown by action type; costs excluding ticket-identifiable faults; time-weighted costs
  R3 resource-balanced prior (each resource equally likely, mutants equally likely within resource)
  R4 symptom-conditioned procedural baseline (outside-in tracing restricted to failing paths)
  R5 distance-cost association under 1,000 random fixed action orders
  R6 learner-model sensitivity: impasse timing; no repair verification; perception masks removed;
     observation noise; open hypothesis space (extra two-fault hypotheses); pairs of misconceptions
  R7 two-fault incidents diagnosed end to end by a single-fault learner with real re-observation
"""
import itertools
import json
import math
import os
import random
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import cloudsim as cs
import diagnosis as dg
import universe as uv

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
SEED = 20261106
S = {}

UNI, CUR = uv.generate()
HYP = {m['mid']: m['cfg'] for m in UNI}
MRES = {m['mid']: m['resource'] for m in UNI}
INV = {v: k for k, v in CUR.items()}
W = dg.World('none', hyp_cfgs=HYP, inject={m: m for m in HYP})
nomT = {a: cs.observe(cs.base_config(), a) for a in cs.MONITORS + cs.ACTIONS}
SYM = [m for m in HYP if any(W.true[m][x] != nomT[x] for x in cs.MONITORS)]
HS = {m: HYP[m] for m in SYM}
CURM = [CUR[f] for f in cs.FIDS]
FB = pd.read_csv(os.path.join(OUT, 'fault_bank.csv')).set_index('fault')
SD = pd.read_csv(os.path.join(OUT, 'strategy_costs_universe.csv')).set_index('mid')
WS = {mc: dg.World(mc, hyp_cfgs=HS, inject={m: m for m in SYM}) for mc in dg.MISCONCEPTIONS}
W0 = WS['none']
MC27 = [m for m in dg.MISCONCEPTIONS if m not in ('none', 'MC1')]

# ------------------------------------------------------------- R1 ambiguity distribution
TIERS = [('T0', cs.MONITORS), ('T1', cs.MONITORS + cs.E_ACTIONS), ('T2', cs.MONITORS + cs.E_ACTIONS + cs.M_ACTIONS),
         ('T3', cs.MONITORS + cs.E_ACTIONS + cs.M_ACTIONS + cs.S_ACTIONS), ('T4', cs.MONITORS + cs.ACTIONS)]
rows = []
for name, acts in TIERS:
    g = defaultdict(list)
    for m in SYM:
        g[tuple(W.true[m][a] for a in acts)].append(m)
    sz = np.array([len(c) for c in g.values() for _ in c])
    szc = np.array([len(g[tuple(W.true[m][a] for a in acts)]) for m in CURM if m in SYM])
    rows.append(dict(tier=name, universe_mean=sz.mean(), universe_median=np.median(sz), universe_q1=np.percentile(sz, 25),
                     universe_q3=np.percentile(sz, 75), universe_max=sz.max(), universe_share_singleton=(sz == 1).mean(),
                     curated_mean_in_universe=szc.mean(), curated_median_in_universe=np.median(szc)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'robust_ambiguity.csv'), index=False)
S['ambiguity'] = rows

# ------------------------------------------------------------- R2 action types, zero-action faults, time weights
def atype(a):
    return {'E': 'probe', 'M': 'telemetry', 'S': 'shell', 'C': 'inspection', 'R': 'path analysis'}[cs.TIER[a]]


pf_u, _ = dg.optimal_policy(W0, 'unit')
br = []
for m in SYM:
    tr = dg.run_episode(W0, m, strategy='order', order=dg.TRACE_ORDER, budget=80)
    rec = dict(mid=m, fault=INV.get(m, ''), optimal=pf_u[m]['cost'])
    for t in ['probe', 'telemetry', 'shell', 'inspection']:
        rec['opt_' + t] = sum(atype(a) == t for a in pf_u[m]['path'])
        rec['trace_' + t] = sum(atype(a) == t for a in tr['trace'] if not a.startswith('REPAIR'))
    br.append(rec)
BR = pd.DataFrame(br)
BR.to_csv(os.path.join(OUT, 'robust_action_types.csv'), index=False)
S['action_types_universe'] = {c: float(BR[c].mean()) for c in BR.columns if c.startswith(('opt_', 'trace_'))}
S['action_types_curated'] = {c: float(BR[BR.fault != ''][c].mean()) for c in BR.columns if c.startswith(('opt_', 'trace_'))}
nz_c = FB[FB.optimal >= 1]
nz_u = SD[SD.optimal >= 1]
S['nonzero'] = dict(curated_n=int(len(nz_c)), universe_n=int(len(nz_u)),
                    curated={s: float(nz_c[s].mean()) for s in ['optimal', 'greedy', 'trace', 'sweep', 'random']},
                    universe={s: float(nz_u[s].mean()) for s in ['optimal', 'greedy', 'trace', 'sweep', 'random']})
tw = []
for lab, D in [('curated', SD.loc[CURM]), ('universe', SD)]:
    tw.append(dict(set=lab, **{s: float(D[s + '_min'].mean()) for s in ['optimal', 'greedy', 'trace', 'sweep']}))
pd.DataFrame(tw).to_csv(os.path.join(OUT, 'robust_time_weighted.csv'), index=False)
S['time_weighted_minutes'] = tw

# ------------------------------------------------------------- R3 resource-balanced prior
cnt = defaultdict(int)
for m in SYM:
    cnt[MRES[m]] += 1
WT = {m: 1.0 / cnt[MRES[m]] for m in SYM}


def optimal_weighted(world, w):
    P, acts = world.pred, list(world.actions)

    @lru_cache(maxsize=None)
    def V(Sx):
        if len(Sx) <= 1:
            return 0.0, None
        tot = sum(w[h] for h in Sx)
        best, ba = math.inf, None
        for a in acts:
            g = defaultdict(list)
            for h in Sx:
                g[P[h][a]].append(h)
            if len(g) == 1:
                continue
            v = 1.0 + sum(sum(w[h] for h in B) / tot * V(tuple(sorted(B)))[0] for B in g.values())
            if v < best - 1e-12:
                best, ba = v, a
        return (0.0, None) if ba is None else (best, ba)

    out = {}
    for f in world.true:
        Sx = tuple(sorted(world.ticket_consistent(f)))
        n = 0
        while len(Sx) > 1:
            v, a = V(Sx)
            if a is None:
                break
            n += 1
            Sx = tuple(sorted(h for h in Sx if P[h][a] == P[f][a]))
        out[f] = n
    return out


pw = optimal_weighted(W0, WT)
PR = pd.DataFrame(dict(mid=SYM, fault=[INV.get(m, '') for m in SYM], weight=[WT[m] for m in SYM],
                       optimal_uniform=[pf_u[m]['cost'] for m in SYM], optimal_resource_prior=[pw[m] for m in SYM]))
PR.to_csv(os.path.join(OUT, 'robust_prior.csv'), index=False)
wsum = sum(WT.values())
S['prior'] = dict(universe_unweighted_mean_uniform=float(PR.optimal_uniform.mean()),
                  universe_unweighted_mean_resource=float(PR.optimal_resource_prior.mean()),
                  universe_resource_weighted_mean_resource=float(sum(WT[m] * pw[m] for m in SYM) / wsum),
                  universe_resource_weighted_mean_uniform=float(sum(WT[m] * pf_u[m]['cost'] for m in SYM) / wsum),
                  curated_mean_uniform=float(PR[PR.fault != ''].optimal_uniform.mean()),
                  curated_mean_resource=float(PR[PR.fault != ''].optimal_resource_prior.mean()),
                  max_abs_change=int((PR.optimal_resource_prior - PR.optimal_uniform).abs().max()))

# ------------------------------------------------------------- R4 symptom-conditioned procedural baseline
RDB = {'rds', 'rds-config', 'sg-db', 'nacl-db', 'appcfg'}
DEPS2 = {k: list(v) for k, v in cs.DEPS.items()}


def reach(n, drop=()):
    seen, st = {n}, [n]
    while st:
        u = st.pop()
        for v in DEPS2.get(u, []):
            if v not in seen and v not in drop:
                seen.add(v)
                st.append(v)
    return seen


MON_NODES = {'cf_static': {'cf', 'cf-config', 'bucket'}, 'cf_root': reach('cf', RDB), 'cf_items': reach('cf'),
             'cf_upload': reach('cf', RDB), 'az_test_a': reach('cf', RDB), 'az_test_b': reach('cf', RDB),
             'alarm_unhealthy': reach('tg-app', RDB), 'ssm_available': reach('instance', RDB),
             'logs_present': reach('instance', RDB)}
ACT_NODE = {'alb_get_root': 'alb', 'ping_alb': 'alb', 'target_health': 'tg-app', 'asg_activity': 'asg', 'console_a': 'instance',
            'console_b': 'instance', 'read_app_logs': 'app', 'flow_logs': 'instance', 'c_rt_pub': 'rt-pub', 'c_rt_app_a': 'rt-app-a',
            'c_rt_app_b': 'rt-app-b', 'c_igw': 'igw', 'c_nat_a': 'nat-a', 'c_nat_b': 'nat-b', 'c_nacl_pub': 'nacl-pub',
            'c_nacl_app': 'nacl-app', 'c_nacl_db': 'nacl-db', 'c_sg_alb': 'sg-alb', 'c_sg_app': 'sg-app', 'c_sg_db': 'sg-db',
            'c_alb': 'alb-config', 'c_tg': 'tg-config', 'c_asg': 'asg-config', 'c_lt': 'lt', 'c_role': 'role', 'c_bucket': 'bucket',
            'c_rds': 'rds-config', 'c_appcfg': 'appcfg', 'c_cf': 'cf-config'}
for a in cs.S_ACTIONS:
    k = a.split('@')[0]
    ACT_NODE[a] = {'curl_localhost_health': 'app', 'curl_internet': 'nat-a' if a.endswith('a') else 'nat-b',
                   'ping_internet': 'nat-a' if a.endswith('a') else 'nat-b', 'nc_rds_3306': 'rds', 'ping_rds': 'rds',
                   's3_put_test': 'bucket', 'sts_identity': 'role'}[k]


def conditioned_order(m):
    fail = [x for x in cs.MONITORS if W0.true[m][x] != nomT[x]]
    R = set().union(*[MON_NODES[x] for x in fail]) if fail else set()
    first = [a for a in dg.TRACE_ORDER if ACT_NODE.get(a) in R]
    return first + [a for a in dg.TRACE_ORDER if a not in first]


cond = {}
for m in SYM:
    r = dg.run_episode(W0, m, strategy='order', order=conditioned_order(m), budget=80)
    assert r['outcome'] == 'correct'
    cond[m] = r['n_actions']
SD['conditioned'] = pd.Series(cond)
S['conditioned_trace'] = dict(curated_mean=float(SD.loc[CURM].conditioned.mean()), curated_median=float(SD.loc[CURM].conditioned.median()),
                              universe_mean=float(SD.loc[SYM].conditioned.mean()),
                              curated_trace_mean=float(SD.loc[CURM].trace.mean()), curated_optimal_mean=float(SD.loc[CURM].optimal.mean()),
                              spearman_distance=float(spearmanr(FB.loc[cs.FIDS].distance, SD.loc[CURM].conditioned).statistic))


TRIAGE_FIRST = ['target_health', 'asg_activity', 'console_a', 'console_b', 'read_app_logs', 'flow_logs']


def triage_order(m):
    rest = [a for a in conditioned_order(m) if a not in TRIAGE_FIRST]
    return TRIAGE_FIRST + rest


tri = {}
for m in SYM:
    r = dg.run_episode(W0, m, strategy='order', order=triage_order(m), budget=80)
    assert r['outcome'] == 'correct'
    tri[m] = r['n_actions']
SD['triage'] = pd.Series(tri)
S['triage_procedure'] = dict(curated_mean=float(SD.loc[CURM].triage.mean()), curated_median=float(SD.loc[CURM].triage.median()),
                             universe_mean=float(SD.loc[SYM].triage.mean()),
                             spearman_distance=float(spearmanr(FB.loc[cs.FIDS].distance, SD.loc[CURM].triage).statistic),
                             spearman_p=float(spearmanr(FB.loc[cs.FIDS].distance, SD.loc[CURM].triage).pvalue))
SD.loc[SYM, ['conditioned', 'triage']].to_csv(os.path.join(OUT, 'robust_conditioned_trace.csv'))

# ------------------------------------------------------------- R5 distance-cost association under random fixed orders
rng = random.Random(SEED)
dist = FB.loc[cs.FIDS].distance.values
rhos = []
base_acts = [a for a in dg.TRACE_ORDER]
for k in range(1000):
    order = base_acts[:]
    rng.shuffle(order)
    c = [dg.run_episode(W0, m, strategy='order', order=order, budget=80)['n_actions'] for m in CURM]
    rhos.append(spearmanr(dist, c).statistic)
rhos = np.array(rhos)
S['random_orders'] = dict(n=1000, mean_rho=float(np.nanmean(rhos)), q025=float(np.nanpercentile(rhos, 2.5)),
                          q975=float(np.nanpercentile(rhos, 97.5)), share_ge_outside_in=float(np.mean(rhos >= S.get('trace_rho', 0.0))))
S['outside_in_rho'] = float(spearmanr(dist, SD.loc[CURM].trace).statistic)
S['random_orders']['share_ge_outside_in'] = float(np.mean(rhos >= S['outside_in_rho']))
pd.DataFrame(dict(rho=rhos)).to_csv(os.path.join(OUT, 'robust_random_orders.csv'), index=False)

# ------------------------------------------------------------- R6 learner-model sensitivity
def episode(world, f, strategy='greedy', verify=True, eps=0.0, rng=None, budget=80, extra_true=None):
    """Variant of dg.run_episode with optional no-verify stopping and observation noise.
    Outcomes: correct | wrong_stable (no verify) | impasse | budget."""
    P, T = world.pred, (extra_true if extra_true is not None else world.true[f])
    Sx = [h for h in world.hyps if all(P[h][x] == T[x] for x in cs.MONITORS)]
    used, n, rep = set(), 0, 0
    while True:
        if not Sx:
            return dict(outcome='impasse', n=n, repairs=rep)
        if len(Sx) == 1 or n >= budget:
            if n >= budget and len(Sx) > 1:
                return dict(outcome='budget', n=n, repairs=rep)
            h = Sx[0]
            if h == f:
                return dict(outcome='correct', n=n, repairs=rep)
            if not verify:
                return dict(outcome='wrong_stable', n=n, repairs=rep)
            rep += 1
            n += 1
            Sx = [x for x in Sx if x != h]
            continue
        best, a = -1, None
        for c in world.actions:
            if c in used:
                continue
            g = defaultdict(int)
            for h in Sx:
                g[P[h][c]] += 1
            if len(g) < 2:
                continue
            k = len(Sx)
            gain = math.log2(k) - sum(v / k * math.log2(v) for v in g.values())
            if gain > best + 1e-12:
                best, a = gain, c
        if a is None:
            if f in Sx:
                return dict(outcome='correct', n=n + Sx.index(f), repairs=rep + Sx.index(f))
            return dict(outcome='impasse' if verify else 'wrong_stable', n=n + len(Sx), repairs=rep + len(Sx))
        used.add(a)
        n += 1
        o = T[a]
        if eps > 0 and rng.random() < eps:
            alts = sorted({P[h][a] for h in world.hyps} - {o})
            if alts:
                o = rng.choice(alts)
        Sx = [h for h in Sx if P[h][a] == o]


# impasse timing per misconception from run_analyses output
BANK = pd.read_csv(os.path.join(OUT, 'misconception_episodes.csv'))
tim = []
for mc in MC27:
    for strat in ['greedy', 'trace']:
        x = BANK[(BANK.mc == mc) & (BANK.strategy == strat) & BANK.code.isin(['I', 'WI'])]
        tim.append(dict(mc=mc, strategy=strat, impasses=len(x), ticket_excludes_true=int(x.ticket_blind.sum()),
                        impasse_at_zero_actions=int((x.impasse_at == 0).sum()), after_actions=int((x.impasse_at > 0).sum()),
                        with_wrong_repair=int((x.code == 'WI').sum()), median_actions=float(x.impasse_at.median()) if len(x) else np.nan))
TIM = pd.DataFrame(tim)
TIM.to_csv(os.path.join(OUT, 'robust_impasse_timing.csv'), index=False)

# misconception variants
def variant(mc, mask=True):
    d = dict(dg.MISCONCEPTIONS[mc])
    if not mask:
        d = dict(d, perceive=None)
    return d


def make_world(defn, hyp=HS, inj=None):
    key = '__tmp__'
    dg.MISCONCEPTIONS[key] = defn
    try:
        w = dg.World(key, hyp_cfgs=hyp, inject=inj or {m: m for m in hyp if m in SYM})
    finally:
        del dg.MISCONCEPTIONS[key]
    return w


def classify(world, base_world, verify=True, eps=0.0, seeds=1, ids=SYM):
    out = defaultdict(int)
    for m in ids:
        for s in range(seeds):
            rr = random.Random(SEED * 7 + s * 1009 + int(m[1:])) if eps > 0 else None
            r = episode(world, m, verify=verify, eps=eps, rng=rr)
            b = episode(base_world, m, verify=verify)
            if r['outcome'] == 'correct' and r['repairs'] <= b['repairs']:
                out['unaffected'] += 1
            else:
                out[r['outcome'] if r['outcome'] != 'correct' else 'wrong_repair_then_correct'] += 1
    return dict(out)


sens = []
for mc in ['none'] + MC27:
    Wm = WS[mc]
    sens.append(dict(mc=mc, condition='baseline (verify, masks)', **classify(Wm, W0)))
    sens.append(dict(mc=mc, condition='no repair verification', **classify(Wm, W0, verify=False)))
    if mc != 'none' and dg.MISCONCEPTIONS[mc]['perceive'] is not None:
        Wn = make_world(variant(mc, mask=False))
        sens.append(dict(mc=mc, condition='perception mask removed', **classify(Wn, W0)))
for eps in [0.05, 0.10]:
    for mc in ['none'] + MC27:
        sens.append(dict(mc=mc, condition='observation noise %.2f (10 seeds)' % eps,
                         **classify(WS[mc], W0, eps=eps, seeds=10)))
SENS = pd.DataFrame(sens).fillna(0)
SENS.to_csv(os.path.join(OUT, 'robust_learner_sensitivity.csv'), index=False)


# selectivity of impasses under observation noise: impasse rate on mutants the misconception
# affects in the noise-free model versus mutants it does not affect
AFF0 = {}
for mc in MC27:
    AFF0[mc] = {m for m in SYM if not (episode(WS[mc], m)['outcome'] == 'correct' and episode(WS[mc], m)['repairs'] <= episode(W0, m)['repairs'])}
selr = []
for eps in [0.05, 0.10]:
    base_imp = []
    for m in SYM:
        for s in range(10):
            rr = random.Random(SEED * 7 + s * 1009 + int(m[1:]))
            base_imp.append(episode(W0, m, eps=eps, rng=rr)['outcome'] == 'impasse')
    for mc in MC27:
        ia, iu = [], []
        for m in SYM:
            for s in range(10):
                rr = random.Random(SEED * 7 + s * 1009 + int(m[1:]))
                imp = episode(WS[mc], m, eps=eps, rng=rr)['outcome'] == 'impasse'
                (ia if m in AFF0[mc] else iu).append(imp)
        selr.append(dict(eps=eps, mc=mc, n_affected=len(AFF0[mc]), impasse_rate_affected=float(np.mean(ia)),
                         impasse_rate_unaffected=float(np.mean(iu)), impasse_rate_correct_model=float(np.mean(base_imp))))
SEL = pd.DataFrame(selr)
SEL.to_csv(os.path.join(OUT, 'robust_noise_selectivity.csv'), index=False)
S['noise_selectivity'] = selr

# open hypothesis space: k extra two-fault hypotheses (outside the single-fault universe)
rng3 = random.Random(SEED + 3)
pairs_all = [(x, y) for x, y in itertools.combinations(SYM, 2) if MRES[x] != MRES[y]]
open_rows = []
for k in [10, 20]:
    for rep in range(5):
        extra = rng3.sample(pairs_all, k)
        H2 = dict(HS)
        for x, y in extra:
            # compose the two single mutations
            cx, cy, cb = HYP[x], HYP[y], cs.base_config()

            def merge(a, b, base):
                for kk in b:
                    if isinstance(b[kk], dict):
                        merge(a[kk], b[kk], base[kk])
                    elif repr(b[kk]) != repr(base[kk]):
                        a[kk] = b[kk]
            c = __import__('copy').deepcopy(cx)
            merge(c, cy, cb)
            H2['X%s+%s' % (x, y)] = c
        for mc in ['none'] + MC27:
            Wk = dg.World(mc, hyp_cfgs=H2, inject={m: m for m in SYM})
            res = defaultdict(int)
            nact = []
            for m in SYM:
                r = episode(Wk, m)
                b = episode(WS[mc], m)
                res[r['outcome']] += 1
                res['changed_vs_closed'] += int(r['outcome'] != b['outcome'])
                if r['outcome'] == 'correct':
                    nact.append(r['n'])
            open_rows.append(dict(k=k, rep=rep, mc=mc, mean_actions_correct=float(np.mean(nact)) if nact else np.nan, **res))
OPEN = pd.DataFrame(open_rows).fillna(0)
OPEN.to_csv(os.path.join(OUT, 'robust_open_hypotheses.csv'), index=False)

# pairs of misconceptions (hypothesis-driven learner): is the affected set the union of the singles?
def compose(m1, m2):
    a, b = dg.MISCONCEPTIONS[m1], dg.MISCONCEPTIONS[m2]
    pa, pb = a['perceive'], b['perceive']
    per = None
    if pa or pb:
        per = lambda act, o, pa=pa, pb=pb: (pb(act, pa(act, o)) if pa and pb else (pa or pb)(act, o))
    return dict(label=m1 + '+' + m2, sem={**a['sem'], **b['sem']}, perceive=per)


single_aff = {}
for mc in MC27:
    single_aff[mc] = {m for m in SYM if not (episode(WS[mc], m)['outcome'] == 'correct' and
                                            episode(WS[mc], m)['repairs'] <= episode(W0, m)['repairs'])}
prow = []
for m1, m2 in itertools.combinations(MC27, 2):
    Wp = make_world(compose(m1, m2))
    aff = {m for m in SYM if not (episode(Wp, m)['outcome'] == 'correct' and episode(Wp, m)['repairs'] <= episode(W0, m)['repairs'])}
    imp = sum(episode(Wp, m)['outcome'] != 'correct' for m in aff)
    u = single_aff[m1] | single_aff[m2]
    prow.append(dict(pair=m1 + '+' + m2, affected=len(aff), union_of_singles=len(u), extra=len(aff - u), lost=len(u - aff),
                     affected_ending_in_impasse=imp))
PAIRS = pd.DataFrame(prow)
PAIRS.to_csv(os.path.join(OUT, 'robust_misconception_pairs.csv'), index=False)

# ------------------------------------------------------------- R7 two-fault incidents, single-fault learner, real re-observation
def two_fault(f, g, max_actions=80):
    cfg = cs.faulted([f, g])
    remaining = {CUR[f], CUR[g]}
    n, rep, fixed = 0, 0, []
    cur_cfg = cfg
    while True:
        cache = {}
        T = {a: cs.observe(cur_cfg, a, cs.TRUE_SEM, cache) for a in cs.MONITORS + cs.ACTIONS}
        if all(T[x] == nomT[x] for x in cs.MONITORS):
            return dict(outcome='resolved' if len(fixed) == 2 else 'resolved_one_silent', n=n, repairs=rep, fixed=len(fixed))
        Sx = [h for h in W0.hyps if all(W0.pred[h][x] == T[x] for x in cs.MONITORS)]
        used, refuted = set(), set()
        while True:
            if not Sx:
                return dict(outcome='impasse_after_%d_fixed' % len(fixed), n=n, repairs=rep, fixed=len(fixed))
            if len(Sx) == 1:
                h = Sx[0]
                n += 1
                if h in remaining:
                    remaining.discard(h)
                    fixed.append(h)
                    other = list(remaining)
                    cur_cfg = cs.faulted([INV[other[0]]]) if other else cs.base_config()
                    break  # re-observe the ticket after the effective repair
                rep += 1  # a wrong single-fault repair is a no-op: ticket unchanged, h refuted
                refuted.add(h)
                Sx = []
                continue
            best, a = -1, None
            for c in W0.actions:
                if c in used:
                    continue
                gg = defaultdict(int)
                for h in Sx:
                    gg[W0.pred[h][c]] += 1
                if len(gg) < 2:
                    continue
                k = len(Sx)
                gain = math.log2(k) - sum(v / k * math.log2(v) for v in gg.values())
                if gain > best + 1e-12:
                    best, a = gain, c
            if a is None:
                return dict(outcome='indistinguishable_%d_fixed' % len(fixed), n=n, repairs=rep, fixed=len(fixed))
            used.add(a)
            n += 1
            Sx = [h for h in Sx if W0.pred[h][a] == T[a]]
            if n > max_actions:
                return dict(outcome='budget', n=n, repairs=rep, fixed=len(fixed))


MK = pd.read_csv(os.path.join(OUT, 'pair_masking.csv'))
tf = []
for _, r in MK.iterrows():
    o = two_fault(r.f, r.g)
    tf.append(dict(f=r.f, g=r.g, masked_ticket=r.masked_ticket, **o))
TF = pd.DataFrame(tf)
TF.to_csv(os.path.join(OUT, 'robust_two_fault.csv'), index=False)
S['two_fault'] = dict(n=int(len(TF)), outcomes=TF.outcome.value_counts().to_dict(),
                      mean_actions_resolved=float(TF[TF.outcome.str.startswith('resolved')].n.mean()),
                      by_masking=TF.groupby('masked_ticket').outcome.value_counts().unstack(fill_value=0).to_dict())

# ------------------------------------------------------------- R8 worked example (fault A1, correct model and MC3)
MD = {m['mid']: m['desc'] for m in UNI}
wrows = []
for mc in ['none', 'MC3']:
    Wx = WS[mc]
    m = CUR['A1']
    Sx = Wx.ticket_consistent(m)
    wrows.append(dict(learner=mc, step=0, action='ticket', observed='', remaining=len(Sx), true_in_set=m in Sx))
    r = dg.run_episode(Wx, m, strategy='greedy', budget=80)
    for k, a in enumerate(r['trace'], 1):
        if a.startswith('REPAIR:'):
            h = a.split(':')[1]
            Sx = [x for x in Sx if x != h]
            wrows.append(dict(learner=mc, step=k, action='repair: ' + MD[h], observed='all monitors unchanged',
                              remaining=len(Sx), true_in_set=m in Sx))
        else:
            Sx = [h for h in Sx if Wx.pred[h][a] == Wx.true[m][a]]
            wrows.append(dict(learner=mc, step=k, action=a, observed=Wx.true[m][a], remaining=len(Sx), true_in_set=m in Sx))
    wrows.append(dict(learner=mc, step=len(r['trace']) + 1, action='outcome', observed=r['outcome'], remaining=len(Sx), true_in_set=m in Sx))
pd.DataFrame(wrows).to_csv(os.path.join(OUT, 'robust_worked_example.csv'), index=False)

with open(os.path.join(OUT, 'robust_summary.json'), 'w') as fh:
    json.dump(S, fh, indent=1, default=float)
print(json.dumps(S, indent=1, default=float)[:6000])
print(SENS.to_string())
print(SEL.to_string())
print(TIM.to_string())
print(PAIRS.to_string())
print(OPEN.groupby(['k', 'mc']).mean(numeric_only=True).to_string())
