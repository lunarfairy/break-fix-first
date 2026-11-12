"""
replay_llm.py - offline recomputation of the round-2 LLM-benchmark summaries (no API calls).

Inputs : results/llm_episodes.csv, results/llm_transcripts.json, results/matrix_curated_greedy.csv,
         results_r1/llm_episodes.csv (round-1 comparison, descriptive only).
Outputs: llm_summary.csv, llm_fault_level.csv, llm_stats.json (written to --out, default results/).

    python replay_llm.py                 # rewrite results/llm_summary.csv, llm_fault_level.csv, llm_stats.json
    python replay_llm.py --out tmp --check   # write to tmp/ and compare byte-for-byte with results/
"""
import argparse
import filecmp
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

SEED = 20261106
N_BOOT = 10000
N_PERM = 10000
SENSITIVE_CODES = {'I', 'WI', 'W', 'B'}
MC_COLS = ['MC2', 'MC3', 'MC4', 'MC5', 'MC6', 'MC7', 'MC8']
AGENTS = ['haiku', 'sonnet']
HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def kappa(a, b):
    """Cohen's kappa for two lists of categorical labels."""
    a, b = list(a), list(b)
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return float('nan') if pe == 1 else (po - pe) / (1 - pe)


def sensitive_faults(matrix_path):
    m = pd.read_csv(matrix_path, dtype=str)
    sens = {}
    for _, r in m.iterrows():
        sens[r['fault']] = any(str(r[c]).strip() in SENSITIVE_CODES for c in MC_COLS)
    return sens


def perm_test(fail_prop, labels, seed=SEED, n_perm=N_PERM):
    """Two-sided permutation test on mean(failure proportion | sensitive) - mean(... | other)."""
    x = np.asarray(fail_prop, float)
    lab = np.asarray(labels, bool)
    obs = x[lab].mean() - x[~lab].mean()
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(lab)
        d = x[p].mean() - x[~p].mean()
        if abs(d) >= abs(obs) - 1e-12:
            ge += 1
    return dict(diff_mean_failure_sensitive_minus_other=float(obs), n_perm=n_perm, n_extreme=int(ge),
                p_two_sided=(ge + 1) / (n_perm + 1), p_two_sided_raw=ge / n_perm,
                mean_failure_sensitive=float(x[lab].mean()), mean_failure_other=float(x[~lab].mean()),
                n_faults_sensitive=int(lab.sum()), n_faults_other=int((~lab).sum()))


def fisher_block(df):
    s = df[df.misconception_sensitive]
    o = df[~df.misconception_sensitive]
    a, b = int((~s.solved).sum()), int(s.solved.sum())
    c, d = int((~o.solved).sum()), int(o.solved.sum())
    orr, p = fisher_exact([[a, b], [c, d]], alternative='two-sided')
    return dict(sensitive_failed=a, sensitive_episodes=a + b, other_failed=c, other_episodes=c + d,
                failure_rate_sensitive=a / (a + b), failure_rate_other=c / (c + d),
                odds_ratio=(None if not np.isfinite(orr) else float(orr)), fisher_p_two_sided=float(p))


def analyse(ep, tr, matrix_path, r1_path):
    ep = ep.copy()
    for c in ('judge_verdicts', 'grader2_verdicts', 'actions', 'diagnoses'):
        ep[c] = ep[c].fillna('').astype(str)
    for c in ('solved', 'misconception_sensitive', 'used_ping'):
        ep[c] = ep[c].map(lambda v: v if isinstance(v, (bool, np.bool_)) else str(v) == 'True').astype(bool)
    sens = sensitive_faults(matrix_path)
    faults = list(dict.fromkeys(ep['fault']))
    assert len(faults) == 41
    assert (ep['misconception_sensitive'].astype(bool).values == ep['fault'].map(sens).values).all()
    ep['misconception_sensitive'] = ep['fault'].map(sens).astype(bool)

    # ---- fault level
    rows = []
    for a in AGENTS:
        for f in faults:
            s = ep[(ep.agent == a) & (ep.fault == f)].sort_values('run')
            rows.append(dict(fault=f, agent=a, gold_resource=s.gold_resource.iloc[0],
                             misconception_sensitive=bool(sens[f]), runs=len(s), solved_runs=int(s.solved.sum()),
                             outcome_run1=s.outcome.iloc[0], outcome_run2=s.outcome.iloc[1],
                             n_actions_until_end_run1=int(s.n_actions_until_end.iloc[0]),
                             n_actions_until_end_run2=int(s.n_actions_until_end.iloc[1])))
    fl = pd.DataFrame(rows)

    # ---- diagnosis-level agreement (from transcripts)
    dlist = []
    for e in tr:
        for i, d in enumerate(e['diags']):
            if d['resource_ok']:
                dlist.append(dict(fault=e['fault'], agent=e['agent'], run=e['run'], diag_index=i + 1, turn=d['turn'],
                                  resource=d['resource'], statement=d['text'], judge=d['judge'], grader2=d['grader2'],
                                  judge_reason=d.get('judge_reason', ''), grader2_reason=d.get('grader2_reason', '')))
    dd = pd.DataFrame(dlist)

    stats = dict(seed=SEED, n_bootstrap=N_BOOT, n_permutations=N_PERM,
                 sensitive_definition='fault row in matrix_curated_greedy.csv has a code in {I, WI, W, B} for any of MC2..MC8',
                 sensitive_faults=[f for f in faults if sens[f]], agents={})
    summ = []
    for a in AGENTS:
        s = ep[ep.agent == a]
        k, n = int(s.solved.sum()), len(s)
        lo, hi = wilson(k, n)
        mat = np.array([[bool(s[(s.fault == f) & (s.run == r)].solved.iloc[0]) for r in (1, 2)] for f in faults], float)
        rng = np.random.default_rng(SEED)
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, len(faults), len(faults))
            boots[b] = mat[idx].mean()
        blo, bhi = np.percentile(boots, [2.5, 97.5])
        fa = fl[fl.agent == a]
        uns = s[~s.solved].sort_values(['fault', 'run'])
        unsolved = [dict(fault=r.fault, run=int(r.run), outcome=r.outcome, n_diagnoses=int(r.n_diagnoses),
                         judge_verdicts=r.judge_verdicts, grader2_verdicts=r.grader2_verdicts,
                         n_actions_until_end=int(r.n_actions_until_end)) for r in uns.itertuples()]
        da = dd[dd.agent == a] if len(dd) else dd
        fl_a = fa.set_index('fault').loc[faults]
        st = dict(model_id=';'.join(sorted(set(s.model_id))), episodes=n, solved=k, solved_rate=k / n,
                  wilson95=[lo, hi], fault_cluster_bootstrap95=[float(blo), float(bhi)],
                  faults_solved_both_runs=int((fa.solved_runs == 2).sum()),
                  faults_solved_at_least_one_run=int((fa.solved_runs >= 1).sum()),
                  share_faults_solved_both_runs=float((fa.solved_runs == 2).mean()),
                  share_faults_solved_at_least_one_run=float((fa.solved_runs >= 1).mean()),
                  mean_actions_all=float(s.n_actions_until_end.mean()),
                  mean_actions_solved=float(s[s.solved].n_actions_until_end.mean()),
                  median_actions_all=float(s.n_actions_until_end.median()),
                  solved_first_diagnosis=int((s.solved & (s.n_diagnoses == 1)).sum()),
                  episodes_naming_gold_resource=int((s.n_gold_resource_diags > 0).sum()),
                  gold_resource_diagnoses=int(len(da)),
                  gold_resource_diagnoses_rejected_in_loop=int((da.judge == 'REJECT').sum()) if len(da) else 0,
                  solved_after_in_loop_reject=int((s.solved & s.judge_verdicts.str.contains('REJECT')).sum()),
                  outcomes={k2: int(v) for k2, v in s.outcome.value_counts().sort_index().items()},
                  unsolved_episodes=unsolved,
                  misconception_episode_level=fisher_block(s),
                  misconception_fault_level_permutation=perm_test(1 - fl_a.solved_runs.values / 2,
                                                                  [sens[f] for f in faults]))
        stats['agents'][a] = st
        summ.append(dict(agent=a, model_id=st['model_id'], episodes=n, solved=k, solved_rate=k / n,
                         wilson_lo=lo, wilson_hi=hi, boot_lo=float(blo), boot_hi=float(bhi),
                         faults_solved_both=st['faults_solved_both_runs'],
                         faults_solved_any=st['faults_solved_at_least_one_run'],
                         solved_first_diagnosis=st['solved_first_diagnosis'],
                         episodes_naming_gold_resource=st['episodes_naming_gold_resource'],
                         gold_resource_diags_rejected=st['gold_resource_diagnoses_rejected_in_loop'],
                         mean_actions_all=st['mean_actions_all'], mean_actions_solved=st['mean_actions_solved'],
                         median_actions_all=st['median_actions_all'], mean_turns=float(s.n_turns.mean()),
                         used_ping=float(s.used_ping.mean()), invalid_replies=int(s.invalid_replies.sum()),
                         api_errors=int(s.api_errors.sum()), truncated_replies=int(s.truncated_replies.sum()),
                         tokens_in=int(s.tokens_in.sum()), tokens_in_cache_read=int(s.tokens_in_cache_read.sum()),
                         tokens_in_cache_write=int(s.tokens_in_cache_write.sum()), tokens_out=int(s.tokens_out.sum()),
                         judge_tokens_in=int(s.judge_tokens_in.sum()), judge_tokens_out=int(s.judge_tokens_out.sum()),
                         grader2_tokens_in=int(s.grader2_tokens_in.sum()), grader2_tokens_out=int(s.grader2_tokens_out.sum())))

    # ---- pooled misconception association
    flp = 1 - np.array([ep[ep.fault == f].solved.mean() for f in faults])
    stats['misconception_pooled'] = dict(episode_level=fisher_block(ep),
                                         fault_level_permutation=perm_test(flp, [sens[f] for f in faults]))

    # ---- judge vs grader2 agreement
    if len(dd):
        agree = (dd.judge == dd.grader2)
        tab = {'%s|%s' % (j, g): int(((dd.judge == j) & (dd.grader2 == g)).sum())
               for j in ('ACCEPT', 'REJECT', 'UNPARSED') for g in ('ACCEPT', 'REJECT', 'UNPARSED')
               if ((dd.judge == j) & (dd.grader2 == g)).sum() > 0}
        dis = dd[~agree].sort_values(['fault', 'agent', 'run', 'diag_index'])
        stats['grader_agreement'] = dict(
            n_gold_resource_diagnoses=int(len(dd)), agreement=float(agree.mean()), n_agree=int(agree.sum()),
            cohen_kappa=kappa(dd.judge, dd.grader2), table_judge_by_grader2=tab,
            per_agent={a: dict(n=int((dd.agent == a).sum()), agreement=float(agree[dd.agent == a].mean()),
                               cohen_kappa=kappa(dd[dd.agent == a].judge, dd[dd.agent == a].grader2)) for a in AGENTS},
            episodes_with_grader2_accepted_gold_diag={a: int(sum(any(d['resource_ok'] and d['grader2'] == 'ACCEPT'
                                                                     for d in e['diags'])
                                                                 for e in tr if e['agent'] == a)) for a in AGENTS},
            episodes_with_grader2_accepted_gold_diag_note=('Descriptive: episodes containing at least one gold-resource '
                                                           'diagnosis that grader2 accepts; not a counterfactual rerun.'),
            disagreements=[dict(fault=r.fault, agent=r.agent, run=int(r.run), diag_index=int(r.diag_index),
                                statement=r.statement, judge=r.judge, grader2=r.grader2,
                                judge_reason=r.judge_reason, grader2_reason=r.grader2_reason)
                           for r in dis.itertuples()])

    # ---- round-1 comparison (descriptive)
    r1 = pd.read_csv(r1_path)
    r1['agent'] = r1['model'].str.replace('-class', '', regex=False)
    comp = {}
    for a in AGENTS:
        s1 = r1[r1.agent == a]
        s2 = ep[ep.agent == a]
        f1 = s1.groupby('fault').correct.sum()
        f2 = s2.groupby('fault').solved.sum()
        changed = {f: dict(r1_solved_runs=int(f1.get(f, 0)), r2_solved_runs=int(f2.get(f, 0)))
                   for f in faults if int(f1.get(f, 0)) != int(f2.get(f, 0))}
        comp[a] = dict(r1_correct_as_reported=int(s1.correct.sum()),
                       r1_correct_without_author_adjudication=int((s1.correct & s1.adjudicated.isna()).sum()),
                       r1_adjudicated_episodes=int(s1.adjudicated.notna().sum()),
                       r1_episodes=int(len(s1)), r2_solved=int(s2.solved.sum()), r2_episodes=int(len(s2)),
                       r1_mean_actions=float(s1.n_actions.mean()), r2_mean_actions_all=float(s2.n_actions_until_end.mean()),
                       r2_mean_actions_to_first_gold_resource_diag=float(s2.n_actions.mean()),
                       faults_with_changed_solved_runs=changed)
    comp['N7'] = {a: dict(r1_solved_runs=int(r1[(r1.agent == a) & (r1.fault == 'N7')].correct.sum()),
                          r2_solved_runs=int(ep[(ep.agent == a) & (ep.fault == 'N7')].solved.sum())) for a in AGENTS}
    comp['L6'] = {a: dict(r1_solved_runs=int(r1[(r1.agent == a) & (r1.fault == 'L6')].correct.sum()),
                          r2_solved_runs=int(ep[(ep.agent == a) & (ep.fault == 'L6')].solved.sum()),
                          note='L6 was redefined in round 2 (ALB node for AZ b in private subnet app-b)') for a in AGENTS}
    comp['note'] = ('Descriptive only. Round 1 ended an episode on the first diagnosis naming the gold resource and judged content '
                    'afterwards (with 3 author overrides); round 2 uses the setting-sensitive verify loop, a corrected cloud model '
                    'and a redefined L6, so the rounds are not a controlled comparison.')
    stats['round1_comparison'] = comp
    return pd.DataFrame(summ), fl, stats


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_outputs(summ, fl, stats, out):
    os.makedirs(out, exist_ok=True)
    summ.to_csv(os.path.join(out, 'llm_summary.csv'), index=False, lineterminator='\n')
    fl.to_csv(os.path.join(out, 'llm_fault_level.csv'), index=False, lineterminator='\n')
    with open(os.path.join(out, 'llm_stats.json'), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(_clean(stats), fh, indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--r1', default=os.path.join(HERE, 'results_r1', 'llm_episodes.csv'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    out = a.out or a.results
    ep = pd.read_csv(os.path.join(a.results, 'llm_episodes.csv'), keep_default_na=False)
    with open(os.path.join(a.results, 'llm_transcripts.json'), encoding='utf-8') as fh:
        tr = json.load(fh)
    summ, fl, stats = analyse(ep, tr, os.path.join(a.results, 'matrix_curated_greedy.csv'), a.r1)
    write_outputs(summ, fl, stats, out)
    if a.check:
        ok = True
        for fn in ('llm_summary.csv', 'llm_fault_level.csv', 'llm_stats.json'):
            same = filecmp.cmp(os.path.join(out, fn), os.path.join(a.results, fn), shallow=False)
            ok &= same
            print('%-22s %s' % (fn, 'IDENTICAL' if same else 'DIFFERENT'))
        print('replay check:', 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
