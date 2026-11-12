"""
run_analyses.py - reproduces every model-based result reported in the manuscript.
Outputs go to results/ (CSV + JSON).  Deterministic (seeded).
"""
import json
import os
import random
import itertools
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import cloudsim as cs
import diagnosis as dg
import universe as uv

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)
SEED = 20261104
summary = {}

# ------------------------------------------------------------------ universe
UNI, CUR = uv.generate()
HYP = {m['mid']: m['cfg'] for m in UNI}
MDESC = {m['mid']: m['desc'] for m in UNI}
MRES = {m['mid']: m['resource'] for m in UNI}
INV = {v: k for k, v in CUR.items()}
INJ_ALL = {m: m for m in HYP}
pd.DataFrame([dict(mid=m['mid'], resource=m['resource'], desc=m['desc'], curated_fault=INV.get(m['mid'], ''))
              for m in UNI]).to_csv(os.path.join(OUT, 'mutation_universe.csv'), index=False)
summary['n_universe'] = len(UNI)

WA = {mc: dg.World(mc, hyp_cfgs=HYP, inject=INJ_ALL) for mc in dg.MISCONCEPTIONS}
W = WA['none']
WR = dg.World('none', actions=cs.ALL_ACTIONS, hyp_cfgs=HYP, inject=INJ_ALL)
nomT = {a: cs.observe(cs.base_config(), a) for a in cs.MONITORS + cs.ALL_ACTIONS}
symptomatic = [m for m in HYP if any(W.true[m][x] != nomT[x] for x in cs.MONITORS)]
summary['n_symptomatic'] = len(symptomatic)
summary['n_silent'] = len(HYP) - len(symptomatic)

# ------------------------------------------------------------------ fault bank table (curated faults)
OBS = {f: {a: cs.observe(cs.faulted(f), a) for a in cs.MONITORS + cs.ALL_ACTIONS} for f in cs.FIDS}
DEPS2 = {k: list(v) for k, v in cs.DEPS.items()}
DEPS2['cf'] = ['cf-config', 'alb']


def bfs(start):
    d = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in DEPS2.get(u, []):
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return d


D_dyn = bfs('cf')
D_static = {'cf': 0, 'cf-config': 1, 'bucket': 1}
D_ent = {n: bfs(n) for n in set(cs.MONITOR_ENTRY.values())}

rows = []
for f in cs.FIDS:
    F = cs.FAULT_BY_ID[f]
    node = cs.RESOURCE_NODE.get(F['resource'], F['resource'])
    failing = [m for m in cs.MONITORS if OBS[f][m] != nomT[m]]
    vis = [m for m in failing if m.startswith('cf_') or m.startswith('az_')]
    user_visible = any(m.startswith('cf_') for m in failing)
    if vis:
        dist = min((D_static if m == 'cf_static' else D_dyn).get(node, np.nan) for m in vis)
    else:
        dist = min(D_ent[cs.MONITOR_ENTRY[m]].get(node, np.nan) for m in failing)
    rows.append(dict(fault=f, description=F['desc'], layer=F['layer'], anchor_theme=F['category'], stage=F['stage'],
                     resource=F['resource'], failing_monitors=';'.join(failing), n_failing=len(failing),
                     user_visible=user_visible, latent=not user_visible, distance=dist,
                     ticket=' | '.join('%s=%s' % (m, OBS[f][m]) for m in failing)))
FB = pd.DataFrame(rows)

# ------------------------------------------------------------------ observability tiers
TIERS = [('T0 incident ticket', cs.MONITORS),
         ('T1 + edge probes', cs.MONITORS + cs.E_ACTIONS),
         ('T2 + service telemetry', cs.MONITORS + cs.E_ACTIONS + cs.M_ACTIONS),
         ('T3 + instance shell', cs.MONITORS + cs.E_ACTIONS + cs.M_ACTIONS + cs.S_ACTIONS),
         ('T4 + configuration inspection', cs.MONITORS + cs.ACTIONS),
         ('T2 + path analysis (no inspection)', cs.MONITORS + cs.E_ACTIONS + cs.M_ACTIONS + cs.R_ACTIONS)]


def classes(acts, ids, table):
    g = defaultdict(list)
    for f in ids:
        g[tuple(table[f][a] for a in acts)].append(f)
    return list(g.values())


tier_rows = []
for name, acts in TIERS:
    cl_c = classes(acts, cs.FIDS, OBS)
    cl_u = classes(acts, symptomatic, WR.true)
    tier_rows.append(dict(tier=name, n_actions=len(acts) - len(cs.MONITORS),
                          curated_classes=len(cl_c), curated_singletons=sum(len(c) == 1 for c in cl_c),
                          curated_largest=max(len(c) for c in cl_c),
                          curated_unresolved=';'.join('/'.join(c) for c in cl_c if len(c) > 1),
                          universe_classes=len(cl_u), universe_singletons=sum(len(c) == 1 for c in cl_u),
                          universe_largest=max(len(c) for c in cl_u),
                          universe_mean_class_size_per_fault=float(np.mean([len(c) for c in cl_u for _ in c]))))
TIERDF = pd.DataFrame(tier_rows)
TIERDF.to_csv(os.path.join(OUT, 'observability_tiers.csv'), index=False)

# ------------------------------------------------------------------ optimal policy + strategies (correct model)
pf_unit, V = dg.optimal_policy(W, 'unit')
pf_min, _ = dg.optimal_policy(W, 'min')
pf_ra, _ = dg.optimal_policy(WR, 'unit')
rng = random.Random(SEED)
strat_rows = []
for m in symptomatic:
    rec = dict(mid=m, fault=INV.get(m, ''), ticket_set=len(W.ticket_consistent(m)),
               optimal=pf_unit[m]['cost'], optimal_min=pf_min[m]['cost'], optimal_with_path_analysis=pf_ra[m]['cost'],
               optimal_path=' > '.join(pf_unit[m]['path']))
    for strat, order in [('greedy', None), ('trace', dg.TRACE_ORDER), ('sweep', dg.SWEEP_ORDER)]:
        r = dg.run_episode(W, m, strategy='greedy' if strat == 'greedy' else 'order', order=order, budget=80)
        assert r['outcome'] == 'correct'
        rec[strat] = r['n_actions']
        rec[strat + '_min'] = sum(cs.MINUTES[cs.TIER[a]] for a in r['trace'] if not a.startswith('REPAIR'))
    rr = [dg.run_episode(W, m, strategy='random', rng=rng, budget=80) for _ in range(200)]
    rec['random'] = float(np.mean([r['n_actions'] for r in rr]))
    strat_rows.append(rec)
SD = pd.DataFrame(strat_rows)
SD.to_csv(os.path.join(OUT, 'strategy_costs_universe.csv'), index=False)
cur = SD[SD.fault != ''].set_index('fault').loc[cs.FIDS]
for c in ['ticket_set', 'optimal', 'optimal_min', 'optimal_with_path_analysis', 'greedy', 'trace', 'sweep', 'random', 'optimal_path']:
    FB[c] = cur[c].values
FB.to_csv(os.path.join(OUT, 'fault_bank.csv'), index=False)

summary['strategy_means_curated'] = {s: float(FB[s].mean()) for s in ['optimal', 'greedy', 'trace', 'sweep', 'random', 'optimal_with_path_analysis']}
summary['strategy_medians_curated'] = {s: float(FB[s].median()) for s in ['optimal', 'greedy', 'trace', 'sweep', 'random']}
summary['strategy_means_universe'] = {s: float(SD[s].mean()) for s in ['optimal', 'greedy', 'trace', 'sweep', 'random', 'optimal_with_path_analysis']}
summary['distance_spearman'] = {}
for s in ['ticket_set', 'optimal', 'greedy', 'trace', 'sweep', 'random']:
    r = spearmanr(FB.distance, FB[s])
    summary['distance_spearman'][s] = dict(rho=float(r.statistic), p=float(r.pvalue))
summary['n_latent'] = int(FB.latent.sum())
summary['latent_faults'] = FB[FB.latent].fault.tolist()
summary['distance_distribution'] = FB.distance.value_counts().sort_index().to_dict()

# ------------------------------------------------------------------ ping vs service-probe agreement
cfgs = {'nominal': cs.base_config(), **{m: HYP[m] for m in HYP}}
pairs = [('ping_alb', 'alb_get_root', lambda o: o != 'timeout', 'client to load balancer'),
         ('ping_internet@a', 'curl_internet@a', lambda o: o == 'ok', 'instance to internet'),
         ('ping_internet@b', 'curl_internet@b', lambda o: o == 'ok', 'instance to internet'),
         ('ping_rds@a', 'nc_rds_3306@a', lambda o: o == 'open', 'instance to database'),
         ('ping_rds@b', 'nc_rds_3306@b', lambda o: o == 'open', 'instance to database')]
prow = []
for name, c in cfgs.items():
    cache = {}
    for p, t, ok, lab in pairs:
        po = cs.observe(c, p, cs.TRUE_SEM, cache)
        to = cs.observe(c, t, cs.TRUE_SEM, cache)
        if po in ('session unavailable', 'no instance') or to.startswith('unknown host'):
            continue
        prow.append(dict(state=name, path=lab, ping_ok=po == 'reply', service_ok=ok(to)))
PG = pd.DataFrame(prow)
PG['agree'] = PG.ping_ok == PG.service_ok
PGs = PG.groupby('path').agg(n=('agree', 'size'), agreement=('agree', 'mean'),
                             ping_fails_service_ok=('agree', lambda s: int(((~PG.loc[s.index, 'ping_ok']) & PG.loc[s.index, 'service_ok']).sum())),
                             ping_ok_service_fails=('agree', lambda s: int(((PG.loc[s.index, 'ping_ok']) & ~PG.loc[s.index, 'service_ok']).sum())),
                             both_fail=('agree', lambda s: int(((~PG.loc[s.index, 'ping_ok']) & ~PG.loc[s.index, 'service_ok']).sum()))).reset_index()
PGs.to_csv(os.path.join(OUT, 'ping_agreement.csv'), index=False)
summary['ping_overall_agreement'] = float(PG.agree.mean())
summary['ping_n_comparisons'] = int(len(PG))

# ------------------------------------------------------------------ fault-pair masking (curated)
def sig(c, acts):
    cache = {}
    return tuple(cs.observe(c, m, cs.TRUE_SEM, cache) for m in acts)


TEL = cs.MONITORS + cs.E_ACTIONS + cs.M_ACTIONS
TK = {f: sig(cs.faulted(f), cs.MONITORS) for f in cs.FIDS}
TL = {f: sig(cs.faulted(f), TEL) for f in cs.FIDS}
mrows = []
for f, g in itertools.combinations(cs.FIDS, 2):
    c = cs.faulted([f, g])
    if repr(c) in (repr(cs.faulted(f)), repr(cs.faulted(g))) or repr(c) != repr(cs.faulted([g, f])):
        continue  # the two faults write the same element
    tk, tl = sig(c, cs.MONITORS), sig(c, TEL)
    mrows.append(dict(f=f, g=g, ticket_equals_f=tk == TK[f], ticket_equals_g=tk == TK[g],
                      tel_equals_f=tl == TL[f], tel_equals_g=tl == TL[g],
                      ticket_new=(tk != TK[f]) and (tk != TK[g])))
MK = pd.DataFrame(mrows)
MK['masked_ticket'] = MK.ticket_equals_f | MK.ticket_equals_g
MK['masked_tel'] = MK.tel_equals_f | MK.tel_equals_g
MK.to_csv(os.path.join(OUT, 'pair_masking.csv'), index=False)
summary['pairs_valid'] = int(len(MK))
summary['pairs_masked_ticket'] = int(MK.masked_ticket.sum())
summary['pairs_masked_telemetry'] = int(MK.masked_tel.sum())
# which faults most often hide others (ticket level)
hider = defaultdict(int)
for _, r in MK.iterrows():
    if r.ticket_equals_f and not r.ticket_equals_g:
        hider[r.f] += 1
    if r.ticket_equals_g and not r.ticket_equals_f:
        hider[r.g] += 1
summary['top_maskers'] = dict(sorted(hider.items(), key=lambda kv: -kv[1])[:8])

# ------------------------------------------------------------------ misconception agents on the whole symptomatic bank
rows = []
for mc, Wm in WA.items():
    for m in symptomatic:
        tc = Wm.ticket_consistent(m)
        for strat, order in [('greedy', None), ('trace', dg.TRACE_ORDER)]:
            r = dg.run_episode(Wm, m, strategy='greedy' if strat == 'greedy' else 'order', order=order, budget=80)
            rows.append(dict(mc=mc, mid=m, fault=INV.get(m, ''), strategy=strat, outcome=r['outcome'],
                             n_actions=r['n_actions'], repairs=r['repairs'], ticket_blind=m not in tc,
                             impasse_at=r['impasse_at'], trace=' > '.join(r['trace'])))
BANK = pd.DataFrame(rows)
bb = BANK[BANK.mc == 'none'].set_index(['strategy', 'mid'])


def code(r):
    b = bb.loc[(r.strategy, r.mid)]
    if r.outcome == 'impasse':
        return 'I' if r.repairs <= b.repairs else 'WI'
    if r.outcome == 'budget':
        return 'B'
    if r.repairs > b.repairs:
        return 'W'
    if r.n_actions > b.n_actions:
        return '+'
    if r.n_actions < b.n_actions:
        return '-'
    return '.'


BANK['code'] = BANK.apply(code, axis=1)
BANK['revealed'] = BANK.code.isin(['I', 'WI', 'W', 'B'])
BANK.to_csv(os.path.join(OUT, 'misconception_episodes.csv'), index=False)
MCS = [m for m in dg.MISCONCEPTIONS if m != 'none']
mc_rows = []
for mc in MCS:
    for strat in ['greedy', 'trace']:
        x = BANK[(BANK.mc == mc) & (BANK.strategy == strat)]
        xc = x[x.fault != '']
        mc_rows.append(dict(mc=mc, label=dg.MISCONCEPTIONS[mc]['label'], strategy=strat,
                            bank_revealed=int(x.revealed.sum()), bank_impasse=int(x.code.isin(['I', 'WI']).sum()),
                            bank_wrong_repair=int(x.code.isin(['W', 'WI']).sum()), bank_ticket_blind=int(x.ticket_blind.sum()),
                            curated_revealed=int(xc.revealed.sum()), curated_impasse=int(xc.code.isin(['I', 'WI']).sum()),
                            curated_wrong_repair=int(xc.code.isin(['W', 'WI']).sum()),
                            curated_revealing=';'.join(xc[xc.revealed].fault.tolist()),
                            median_actions_to_impasse=float(x[x.code.isin(['I', 'WI'])].impasse_at.median()) if x.code.isin(['I', 'WI']).any() else np.nan))
MCDF = pd.DataFrame(mc_rows)
MCDF.to_csv(os.path.join(OUT, 'misconception_summary.csv'), index=False)
for strat in ['greedy', 'trace']:
    mat = BANK[(BANK.strategy == strat) & (BANK.mc != 'none') & (BANK.fault != '')].pivot(index='fault', columns='mc', values='code').loc[cs.FIDS, MCS]
    mat.to_csv(os.path.join(OUT, 'matrix_curated_%s.csv' % strat))

# ------------------------------------------------------------------ break-fix curriculum and post-test pool
# Revision 2: generation tasks are chosen by a reproducible rule inside stage-scoped hypothesis
# universes.  Every generation deployment is the complete reference service, but the ticket tells
# pairs which components are in scope (those built up to and including the current stage); the
# rest are declared verified.  The learner's hypothesis set is therefore restricted to mutants of
# in-scope settings.
STAGE_NAMES = {1: 'Stage 1: instance, security group, web server',
               2: 'Stage 2: custom VPC, routing, network ACLs, database',
               3: 'Stage 3a: NAT, load balancing, health checks, scaling',
               4: 'Stage 3b: identity, object storage, content delivery'}
RES_STAGE = {'lt': 1, 'sg-app': 1,
             'rt-pub': 2, 'rt-app-a': 2, 'rt-app-b': 2, 'rt-db': 2, 'subnet-assoc': 2, 'igw': 2,
             'nacl-pub': 2, 'nacl-app': 2, 'nacl-db': 2, 'rds': 2, 'sg-db': 2, 'appcfg': 2,
             'nat-a': 3, 'nat-b': 3, 'alb': 3, 'sg-alb': 3, 'tg-app': 3, 'asg': 3, 'app': 3,
             'role': 4, 'bucket': 4, 'cf': 4}


def mstage(mid):
    if MRES[mid] == 'lt' and 'profile' in MDESC[mid]:
        return 4  # the instance profile belongs to the identity stage
    return RES_STAGE[MRES[mid]]


MSTAGE = {m: mstage(m) for m in HYP}
pd.DataFrame([dict(mid=m, resource=MRES[m], stage=MSTAGE[m]) for m in HYP]).to_csv(
    os.path.join(OUT, 'mutant_stage.csv'), index=False)
MC27 = [m for m in dg.MISCONCEPTIONS if m not in ('none', 'MC1')]


def scoped_code(r, b):
    if r['outcome'] == 'impasse':
        return 'I' if r['repairs'] <= b['repairs'] else 'WI'
    if r['outcome'] == 'budget':
        return 'B'
    if r['repairs'] > b['repairs']:
        return 'W'
    return '+' if r['n_actions'] > b['n_actions'] else ('-' if r['n_actions'] < b['n_actions'] else '.')


cand_rows = []
for s in [1, 2, 3, 4]:
    scope = {m: HYP[m] for m in symptomatic if MSTAGE[m] <= s}
    cands = [f for f in cs.FIDS if CUR[f] in scope and MSTAGE[CUR[f]] == s]
    inj = {f: CUR[f] for f in cands}
    Ws = {mc: dg.World(mc, hyp_cfgs=scope, inject=inj) for mc in ['none'] + MC27}
    pf_s, _ = dg.optimal_policy(Ws['none'], 'unit')
    for f in cands:
        m = CUR[f]
        b = dg.run_episode(Ws['none'], m, strategy='greedy', budget=80)
        tr = dg.run_episode(Ws['none'], m, strategy='order', order=dg.TRACE_ORDER, budget=80)
        yl = []
        for mc in MC27:
            r = dg.run_episode(Ws[mc], m, strategy='greedy', budget=80)
            if scoped_code(r, b) in ('I', 'WI', 'W', 'B'):
                yl.append(mc)
        fb = FB.set_index('fault').loc[f]
        cand_rows.append(dict(stage=s, fault=f, mid=m, description=cs.FAULT_BY_ID[f]['desc'], scope_size=len(scope),
                              ticket_set=len(Ws['none'].ticket_consistent(m)), optimal_actions=pf_s[m]['cost'],
                              trace_actions=tr['n_actions'], user_visible=bool(fb.user_visible),
                              impasse_yield=','.join(yl), n_yield=len(yl),
                              eligible=1 <= pf_s[m]['cost'] <= 4))
CAND = pd.DataFrame(cand_rows)
# selection rule (reproducible): within each stage, in stage order, repeatedly add the eligible
# candidate that disables the most misconceptions (MC2-MC8) not yet covered by earlier picks
# (ties: user-visible first, larger total yield, fewer optimal actions, identifier); when no
# candidate adds coverage, fill the stage up to two tasks with the eligible candidate whose ticket
# is most ambiguous (ties: user-visible first, fewer optimal actions, identifier).
covered, sel = set(), []
for s in [1, 2, 3, 4]:
    E = CAND[(CAND.stage == s) & CAND.eligible].copy()
    picks = []
    while True:
        E = E[~E.fault.isin(sel)]
        if E.empty:
            break
        E['new'] = [len(set(str(y).split(',')) - covered - {'', 'nan'}) for y in E.impasse_yield.fillna('')]
        if E.new.max() > 0:
            best = E.sort_values(['new', 'user_visible', 'n_yield', 'optimal_actions', 'fault'],
                                 ascending=[False, False, False, True, True]).iloc[0]
        elif len(picks) < 2:
            best = E.sort_values(['user_visible', 'ticket_set', 'optimal_actions', 'fault'],
                                 ascending=[False, False, True, True]).iloc[0]
        else:
            break
        sel.append(best.fault)
        picks.append(best.fault)
        covered |= set(str(best.impasse_yield).split(',')) - {'', 'nan'}
CAND['selected'] = CAND.fault.isin(sel)
CAND.to_csv(os.path.join(OUT, 'curriculum_candidates.csv'), index=False)
CURR = CAND[CAND.selected].copy()
CURR['stage_name'] = CURR.stage.map(STAGE_NAMES)
CURR.to_csv(os.path.join(OUT, 'curriculum.csv'), index=False)
gen_mids = set(CURR.mid)
summary['curriculum'] = CURR.fault.tolist()
summary['curriculum_coverage'] = sorted(covered)
summary['curriculum_uncovered'] = sorted(set(MC27) - covered)
summary['candidates_per_stage'] = CAND.groupby('stage').size().to_dict()
summary['eligible_per_stage'] = CAND[CAND.eligible].groupby('stage').size().to_dict()
rev = BANK[(BANK.mc != 'none') & BANK.revealed].groupby('mid').mc.apply(lambda s: sorted(set(s)))

# post-test pool (full architecture in scope): two targeted items per misconception MC2-MC8 (items
# that the misconception disables), preferring items that need at least one expert action and curated
# items; plus six untargeted items that none of MC2-MC8 disables and that have user-visible symptoms.
rng2 = random.Random(SEED + 1)
pool = []
used = set(gen_mids)
for mc in MC27:
    cands = [m for m in symptomatic if m not in used and mc in rev.get(m, [])]
    rng2.shuffle(cands)
    cands.sort(key=lambda m: (pf_unit[m]['cost'] < 1, m not in INV))
    for m in cands[:2]:
        used.add(m)
        pool.append(dict(mid=m, fault=INV.get(m, ''), desc=MDESC[m], target=mc, type='targeted'))
unt = [m for m in symptomatic if m not in used and not [x for x in rev.get(m, []) if x != 'MC1']
       and pf_unit[m]['cost'] >= 1 and any(W.true[m][x] != nomT[x] for x in ['cf_root', 'cf_items', 'cf_upload', 'cf_static'])]
rng2.shuffle(unt)
unt.sort(key=lambda m: m not in INV)
for m in unt[:6]:
    used.add(m)
    pool.append(dict(mid=m, fault=INV.get(m, ''), desc=MDESC[m], target='', type='untargeted'))
POOL = pd.DataFrame(pool)
POOL['optimal_actions'] = [pf_unit[m]['cost'] for m in POOL.mid]
POOL['trace_actions'] = [int(SD.set_index('mid').loc[m, 'trace']) for m in POOL.mid]
POOL['n_misconceptions_revealing'] = [len(rev.get(m, [])) for m in POOL.mid]
POOL['ticket_identifiable'] = POOL.optimal_actions == 0
POOL.to_csv(os.path.join(OUT, 'posttest_pool.csv'), index=False)
summary['pool_n'] = int(len(POOL))
summary['pool_targeted'] = int((POOL.type == 'targeted').sum())
summary['pool_untargeted'] = int((POOL.type == 'untargeted').sum())
summary['pool_ticket_identifiable'] = POOL[POOL.ticket_identifiable].mid.tolist()

# item features for the whole symptomatic universe (descriptive; trial_sim.py draws its own
# item difficulties from N(0, 1) and does not read this file)
diff = SD[['mid', 'optimal', 'trace', 'ticket_set']].copy()
diff['n_revealing'] = [len(rev.get(m, [])) for m in diff.mid]
diff.to_csv(os.path.join(OUT, 'item_features.csv'), index=False)

with open(os.path.join(OUT, 'summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
print(TIERDF.to_string())
print(MCDF.to_string())
print(CURR.to_string())
print(POOL.to_string())
