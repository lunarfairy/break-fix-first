"""
diagnosis.py - diagnostic agents over the cloudsim fault model.

A diagnosis episode: a single fault f* is injected; the learner receives the incident
ticket (outcomes of all MONITORS) for free and then chooses actions. Hypotheses are the
single-fault configurations of the world's hypothesis set (in the reported analyses, the
184 symptomatic members of the mutation universe, or a stage-scoped subset). Updating is by elimination: a hypothesis survives if the
outcome it predicts (under the learner's semantics, after the learner's perception) equals
the perceived true outcome. A learner whose forward model is wrong can therefore
(i) eliminate the true fault, (ii) converge on a wrong fault, or (iii) eliminate every
hypothesis (an impasse). After each declared diagnosis the learner "repairs" it and
re-checks the ticket (the verify loop): a wrong repair leaves the symptoms unchanged and
refutes the declared hypothesis, after which diagnosis continues.
"""
import math
import random
from functools import lru_cache
from collections import defaultdict

import cloudsim as cs

# ---------------------------------------------------------------- misconception models
def _mask_rep(a, o, pairs):
    for x, y in pairs:
        o = o.replace(x, y)
    return o


MISCONCEPTIONS = {
    'none': dict(label='Correct model', sem={}, perceive=None),
    'MC1': dict(label='Ping tests service reachability', sem={'icmp_as_tcp': True}, perceive=None),
    'MC2': dict(label='Security groups are stateless', sem={'sg_stateless': True}, perceive=None),
    'MC3': dict(label='Network ACLs are irrelevant', sem={'nacl_ignored': True},
                perceive=lambda a, o: 'nacl:as-expected' if a.startswith('c_nacl') else o),
    'MC4': dict(label='Identity policy alone decides S3 access', sem={'resource_policy_ignored': True},
                perceive=lambda a, o: 'bucket:as-expected' if a == 'c_bucket' else o),
    'MC5': dict(label='A public subnet gives internet access', sem={'igw_route_suffices': True},
                perceive=lambda a, o: _mask_rep(a, o, [('pub-', 'app-')]) if a == 'c_asg' else o),
    'MC6': dict(label='Instances act with the console user\'s permissions', sem={'instance_has_user_perms': True},
                perceive=lambda a, o: ('role:as-expected' if a == 'c_role' else
                                       (_mask_rep(a, o, [("('profile', 'None')", "('profile', 'role-app')")]) if a == 'c_lt' else o))),
    'MC7': dict(label='Health checks probe the user-facing page', sem={'hc_root_path': True},
                perceive=lambda a, o: _mask_rep(a, o, [("'/healthz'", "'/health'")]) if a == 'c_tg' else o),
    'MC8': dict(label='A NAT gateway reaches the internet wherever it is placed', sem={'nat_private_ok': True},
                perceive=lambda a, o: _mask_rep(a, o, [("'app-a'", "'pub-a'"), ("'app-b'", "'pub-b'")]) if a.startswith('c_nat') else o),
}


def sem_of(mc):
    s = dict(cs.TRUE_SEM)
    s.update(MISCONCEPTIONS[mc]['sem'])
    return s


class World:
    """Pre-computes true and predicted outcome tables for one learner model.
    hyp_cfgs: dict hypothesis-id -> configuration (the learner's hypothesis space).
    inject:   dict task-id -> hypothesis-id of the injected fault (true outcomes are computed for these)."""

    def __init__(self, mc='none', actions=None, hyp_cfgs=None, inject=None):
        self.mc = mc
        self.actions = list(actions or cs.ACTIONS)
        if hyp_cfgs is None:
            hyp_cfgs = {f: cs.faulted(f) for f in cs.FIDS}
            inject = {f: f for f in cs.FIDS}
        self.hyps = list(hyp_cfgs)
        self.inject = inject
        self.sem = sem_of(mc)
        per = MISCONCEPTIONS[mc]['perceive'] or (lambda a, o: o)
        acts = cs.MONITORS + self.actions
        self.pred = {}
        for h, cfg in hyp_cfgs.items():
            c2 = {}
            self.pred[h] = {a: per(a, cs.observe(cfg, a, self.sem, c2)) for a in acts}
        self.true = {}
        for t, h in inject.items():
            c1 = {}
            self.true[h] = {a: per(a, cs.observe(hyp_cfgs[h], a, cs.TRUE_SEM, c1)) for a in acts}
        self.cost_unit = {a: 1.0 for a in self.actions}
        self.cost_min = {a: cs.MINUTES[cs.TIER[a]] for a in self.actions}

    def ticket_consistent(self, fstar):
        t = self.true[fstar]
        return [h for h in self.hyps if all(self.pred[h][m] == t[m] for m in cs.MONITORS)]


# ---------------------------------------------------------------- optimal policy (correct model)
def optimal_policy(world, cost='unit'):
    """Exact minimum expected cost (uniform prior) to isolate the fault, by memoised
    recursion over hypothesis sets. Returns (expected cost per ticket class, per-fault cost)."""
    c = world.cost_unit if cost == 'unit' else world.cost_min
    acts = list(world.actions)
    P = world.pred

    @lru_cache(maxsize=None)
    def V(S):
        if len(S) <= 1:
            return 0.0, None
        best, besta = math.inf, None
        for a in acts:
            g = defaultdict(list)
            for h in S:
                g[P[h][a]].append(h)
            if len(g) == 1:
                continue
            v = c[a] + sum(len(B) / len(S) * V(tuple(sorted(B)))[0] for B in g.values())
            if v < best - 1e-12:
                best, besta = v, a
        if besta is None:
            return 0.0, None  # indistinguishable
        return best, besta

    per_fault = {}
    for f in world.true:
        S = tuple(sorted(world.ticket_consistent(f)))
        cost_f, path = 0.0, []
        while len(S) > 1:
            v, a = V(S)
            if a is None:
                break
            cost_f += c[a]
            path.append(a)
            S = tuple(sorted(h for h in S if P[h][a] == P[f][a]))
        per_fault[f] = dict(cost=cost_f, n_actions=len(path), path=path, resolved=(len(S) == 1))
    return per_fault, V


# ---------------------------------------------------------------- agents
def entropy_gain(S, a, P):
    g = defaultdict(int)
    for h in S:
        g[P[h][a]] += 1
    n = len(S)
    return math.log2(n) - sum(k / n * math.log2(k) for k in g.values()), len(g)


def run_episode(world, fstar, strategy='greedy', cost='unit', budget=40, order=None, rng=None,
                max_repairs=10):
    """Simulate one diagnosis episode. Returns dict with outcome and cost.
    outcome: 'correct' | 'impasse' | 'budget' ; plus counts of failed repairs and actions."""
    c = world.cost_unit if cost == 'unit' else world.cost_min
    P, T = world.pred, world.true[fstar]
    S = world.ticket_consistent(fstar)
    used, spent, n_act, repairs = set(), 0.0, 0, 0
    first_wrong = None
    trace = []
    seq = list(order) if order is not None else None
    if strategy == 'random':
        seq = list(world.actions)
        (rng or random).shuffle(seq)
    impasse_at = None
    while True:
        if not S:
            impasse_at = n_act
            return dict(outcome='impasse', cost=spent, n_actions=n_act, repairs=repairs,
                        first_wrong=first_wrong, impasse_at=impasse_at, trace=trace)
        if len(S) == 1 or n_act >= budget:
            if n_act >= budget and len(S) > 1:
                return dict(outcome='budget', cost=spent, n_actions=n_act, repairs=repairs,
                            first_wrong=first_wrong, impasse_at=None, trace=trace)
            h = S[0]
            if h == fstar:
                return dict(outcome='correct', cost=spent, n_actions=n_act, repairs=repairs,
                            first_wrong=first_wrong, impasse_at=None, trace=trace)
            # wrong repair -> symptoms persist -> h refuted
            repairs += 1
            spent += 1.0
            n_act += 1
            trace.append('REPAIR:' + h)
            if first_wrong is None:
                first_wrong = h
            S = [x for x in S if x != h]
            if repairs > max_repairs:
                return dict(outcome='budget', cost=spent, n_actions=n_act, repairs=repairs,
                            first_wrong=first_wrong, impasse_at=None, trace=trace)
            continue
        # choose next action
        a = None
        if strategy == 'greedy':
            best = -1
            for cand in world.actions:
                if cand in used:
                    continue
                gain, k = entropy_gain(S, cand, P)
                if k < 2:
                    continue
                score = gain / c[cand]
                if score > best + 1e-12:
                    best, a = score, cand
        else:  # fixed order (trace / sweep / random): take next unused action in sequence
            while seq:
                cand = seq.pop(0)
                if cand not in used:
                    a = cand
                    break
        if a is None:
            # no splitting action left under the learner's model: guess among S, repair-verify
            # try remaining hypotheses in list order through the repair-verify loop
            for h in S:
                if h == fstar:
                    return dict(outcome='correct', cost=spent, n_actions=n_act, repairs=repairs,
                                first_wrong=first_wrong, impasse_at=None, trace=trace)
                if n_act >= budget:   # same caps as the main loop
                    return dict(outcome='budget', cost=spent, n_actions=n_act, repairs=repairs,
                                first_wrong=first_wrong, impasse_at=None, trace=trace)
                repairs += 1
                spent += 1.0
                n_act += 1
                trace.append('REPAIR:' + h)
                if first_wrong is None:
                    first_wrong = h
                if repairs > max_repairs:
                    return dict(outcome='budget', cost=spent, n_actions=n_act, repairs=repairs,
                                first_wrong=first_wrong, impasse_at=None, trace=trace)
            return dict(outcome='impasse', cost=spent, n_actions=n_act, repairs=repairs,
                        first_wrong=first_wrong, impasse_at=n_act, trace=trace)
        used.add(a)
        spent += c[a]
        n_act += 1
        trace.append(a)
        S = [h for h in S if P[h][a] == T[a]]


TRACE_ORDER = ['alb_get_root', 'ping_alb', 'c_cf', 'c_alb', 'c_sg_alb', 'c_nacl_pub', 'c_rt_pub', 'c_igw',
               'target_health', 'c_tg', 'asg_activity', 'c_asg', 'c_lt', 'console_a', 'console_b',
               'c_sg_app', 'c_nacl_app', 'c_rt_app_a', 'c_rt_app_b', 'c_nat_a', 'c_nat_b', 'c_role',
               'read_app_logs', 'c_bucket', 'c_appcfg', 'c_rds', 'c_sg_db', 'c_nacl_db', 'flow_logs'] + \
              [a for a in cs.S_ACTIONS]
SWEEP_ORDER = ['c_rt_pub', 'c_rt_app_a', 'c_rt_app_b', 'c_igw', 'c_nat_a', 'c_nat_b', 'c_nacl_pub', 'c_nacl_app',
               'c_nacl_db', 'c_sg_alb', 'c_sg_app', 'c_sg_db', 'c_lt', 'c_asg', 'c_alb', 'c_tg', 'c_role',
               'c_bucket', 'c_rds', 'c_appcfg', 'c_cf'] + cs.E_ACTIONS + cs.M_ACTIONS + cs.S_ACTIONS
