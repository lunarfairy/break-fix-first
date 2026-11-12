"""
tests/test_model.py - executable checks of the fault model's semantics and invariants.
Run from the breakfix_model folder:  python -m pytest -q tests   (or: python tests/test_model.py)
These are internal verification tests. They check that the code implements the documented semantics;
they are not an external validation against live AWS deployments.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cloudsim as cs      # noqa: E402
import diagnosis as dg     # noqa: E402
import universe as uv      # noqa: E402

BASE = cs.base_config()
NOM = {a: cs.observe(BASE, a) for a in cs.MONITORS + cs.ALL_ACTIONS}


def obs(f, a):
    return cs.observe(cs.faulted(f), a)


def test_nominal_monitors_healthy():
    for m in ['cf_root', 'cf_items', 'cf_upload', 'cf_static']:
        assert NOM[m] == '200', (m, NOM[m])
    assert NOM['alarm_unhealthy'] == 'OK'
    assert NOM['ssm_available'] == 'managed:a,b'
    assert NOM['logs_present'] == 'yes'
    assert NOM['az_test_a'] == NOM['az_test_b'] == 'root=200 items=200'


def test_universe_admissible_and_complete():
    uni, cur = uv.generate()
    assert len(uni) == 228
    keys = {uv._canon(m['cfg']) for m in uni}
    assert len(keys) == len(uni)                       # no duplicate configurations
    assert set(cur) == set(cs.FIDS)                     # every curated fault is a universe member
    for m in uni:                                       # ALB: two subnets in two different AZs
        s = m['cfg']['alb']['subnets']
        assert len(s) == 2 and BASE['subnets'][s[0]][0] != BASE['subnets'][s[1]][0], m['desc']


def test_security_groups_stateful_acls_stateless():
    # removing all sg-app egress must not block responses to the load balancer (stateful) ...
    assert 'Target.Timeout' not in obs('S5', 'target_health')
    # ... whereas removing the ephemeral inbound rule of nacl-app blocks return traffic (stateless):
    assert 'timeout' in obs('A1', 'console_a')


def test_explicit_deny_overrides_identity_allow():
    assert obs('I5', 'cf_upload').startswith('500')
    assert NOM['cf_upload'] == '200'


def test_alb_fail_open_and_reason_codes():
    th = obs('L1', 'target_health')
    assert 'ResponseCodeMismatch' in th
    assert obs('L1', 'cf_root').startswith('200')        # fail open: users still served
    assert obs('L1', 'alarm_unhealthy') == 'ALARM'


def test_alb_never_answers_icmp():
    assert NOM['ping_alb'] == 'timeout'


def test_unreachable_alb_node_gives_intermittent_failures():
    assert 'intermittent-504' in obs('L6', 'cf_root')


def test_latent_faults_leave_user_requests_intact():
    for f in ['N1', 'N2', 'N3', 'C5', 'I6', 'I7', 'D3']:
        for m in ['cf_root', 'cf_items', 'cf_upload', 'cf_static']:
            assert obs(f, m) == NOM[m], (f, m)


def test_perception_masks():
    per = dg.MISCONCEPTIONS['MC3']['perceive']
    assert per('c_nacl_app', 'anything') == 'nacl:as-expected'
    assert per('c_sg_app', 'x') == 'x'


def test_wrong_single_fault_repair_is_a_no_op():
    # Under the single-fault assumption, restoring the design value of a setting that is not the faulted
    # one leaves the configuration, and hence every monitor, unchanged.
    uni, cur = uv.generate()
    for f in ['A1', 'I5', 'N7', 'L1']:
        cfg = cs.faulted(f)
        for m in uni[:60]:
            if m['mid'] == cur[f]:
                continue
            repaired = copy.deepcopy(cfg)
            # restore every top-level block that m changes back to the design value, unless f changed it
            for k in m['cfg']:
                if repr(m['cfg'][k]) != repr(BASE[k]) and repr(cfg[k]) == repr(BASE[k]):
                    repaired[k] = copy.deepcopy(BASE[k])
            assert repr(repaired) == repr(cfg)


def test_optimal_not_worse_than_greedy():
    uni, _ = uv.generate()
    hyp = {m['mid']: m['cfg'] for m in uni}
    w = dg.World('none', hyp_cfgs=hyp, inject={m: m for m in hyp})
    sym = [m for m in hyp if any(w.true[m][x] != NOM[x] for x in cs.MONITORS)]
    w = dg.World('none', hyp_cfgs={m: hyp[m] for m in sym}, inject={m: m for m in sym})
    pf, _ = dg.optimal_policy(w, 'unit')
    # the optimal policy minimises the EXPECTED cost within each ticket class (uniform prior),
    # so the class mean, not every single fault, must be no worse than the greedy learner's
    classes = {}
    for m in sym:
        classes.setdefault(tuple(sorted(w.ticket_consistent(m))), []).append(m)
    for members in classes.values():
        g = [dg.run_episode(w, m, strategy='greedy', budget=80) for m in members]
        assert all(x['outcome'] == 'correct' for x in g)
        assert sum(pf[m]['cost'] for m in members) <= sum(x['n_actions'] for x in g) + 1e-9


if __name__ == '__main__':
    fns = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for fn in fns:
        fn()
        print('ok', fn.__name__)
    print('%d tests passed' % len(fns))
