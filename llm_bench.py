"""
llm_bench.py (v2, round-2 revision) - interactive LLM-agent diagnosis episodes against the cloudsim fault model.

Each episode: the agent receives the intended design (reference configuration), the incident ticket (outcomes of all
synthetic monitors) and the action catalogue. On each turn it returns one JSON object: {"action": <name>} or
{"diagnosis": {"resource": <name>, "misconfiguration": <text>}}. Limits: 15 turns, 3 diagnoses.

v2 change (setting-sensitive verify loop): a diagnosis that names the gold resource is passed to an in-loop judge model
that applies the fixed RUBRIC below. ACCEPT ends the episode as solved ("the incident is resolved"); REJECT, like a
wrong resource, is answered with "all monitors unchanged; the incident persists" and the episode continues. There is no
author adjudication: correctness is the in-loop verdict. grade2() re-grades every gold-resource diagnosis with an
independent second model under the same rubric (post hoc, does not affect outcomes).
Requires the platform `host` object (host.llm) - run inside the analysis kernel.
"""
import json
import re
import time

import cloudsim as cs

RESOURCES = ['rt-pub', 'rt-app-a', 'rt-app-b', 'igw', 'nat-a', 'nat-b', 'nacl-pub', 'nacl-app', 'nacl-db',
             'sg-alb', 'sg-app', 'sg-db', 'alb', 'target-group', 'asg', 'launch-template', 'iam-role',
             'bucket-policy', 'rds', 'app-config', 'cloudfront']
GOLD_RES = {'N1': 'rt-app-a', 'N2': 'rt-app-b', 'N3': 'nat-a', 'N4': 'rt-pub', 'N5': 'igw', 'N6': 'alb', 'N7': 'asg',
            'S1': 'sg-app', 'S2': 'sg-app', 'S3': 'sg-alb', 'S4': 'sg-db', 'S5': 'sg-app', 'S6': 'sg-alb',
            'A1': 'nacl-app', 'A2': 'nacl-db', 'A3': 'nacl-pub', 'A4': 'nacl-app',
            'L1': 'target-group', 'L2': 'target-group', 'L3': 'asg', 'L4': 'alb', 'L5': 'launch-template', 'L6': 'alb',
            'C1': 'launch-template', 'C2': 'asg', 'C3': 'launch-template', 'C4': 'launch-template', 'C5': 'asg',
            'I1': 'iam-role', 'I2': 'iam-role', 'I3': 'launch-template', 'I4': 'iam-role', 'I5': 'bucket-policy',
            'I6': 'iam-role', 'I7': 'iam-role', 'D1': 'app-config', 'D2': 'app-config', 'D3': 'rds',
            'E1': 'cloudfront', 'E2': 'cloudfront', 'E3': 'bucket-policy'}

DESIGN = """INTENDED DESIGN (what the deployment should look like)
- VPC with subnets pub-a, pub-b (public, AZ a/b), app-a, app-b (private application tier), db-a, db-b (private data tier).
- Internet gateway attached. Route tables: rt-pub (pub-a, pub-b): default route -> igw. rt-app-a (app-a): default -> nat-a.
  rt-app-b (app-b): default -> nat-b. rt-db (db-a, db-b): local only.
- NAT gateways nat-a in pub-a and nat-b in pub-b, each with an elastic IP.
- Network ACLs: nacl-pub (public subnets) allows all traffic in and out. nacl-app (app subnets): inbound tcp/80 from the VPC,
  inbound tcp/1024-65535 from anywhere; outbound all. nacl-db (db subnets): inbound tcp/3306 from app subnets; outbound tcp/1024-65535 to app subnets.
- Security groups: sg-alb inbound tcp/80 and tcp/443 from anywhere, outbound all. sg-app inbound tcp/80 from sg-alb, outbound all.
  sg-db inbound tcp/3306 from sg-app, outbound all. No security group allows ICMP.
- Internet-facing Application Load Balancer in pub-a and pub-b with sg-alb; listener :80 forwards to target group tg-app
  (port 80, health check GET /health expecting 200).
- Auto Scaling group across app-a and app-b, desired 2, attached to tg-app, ELB health checks, launch template:
  AMI ami-web-v3, instance profile role-app, sg-app, no public IP, web server on :80; user data fetches config/app.json from S3 at boot
  (the web server does not start if this fails).
- IAM role role-app (trusted by ec2.amazonaws.com): s3:GetObject and s3:PutObject on arn:aws:s3:::assets-bkt/*,
  logs:PutLogEvents, Systems Manager agent permissions.
- Bucket assets-bkt: bucket policy grants CloudFront origin access control read access to static/; no deny statements.
- RDS MySQL, Multi-AZ, subnets db-a/db-b, sg-db. App config (in S3) holds DB endpoint appdb.current and the current password.
- CloudFront distribution: default origin = the ALB over HTTP (http-only); /static/* origin = assets-bkt via origin access control.
"""

ACTION_HELP = """ACTIONS (one per turn)
Edge probes: alb_get_root (HTTP GET / directly on the ALB from the internet), ping_alb (ICMP echo to the ALB).
Service telemetry: target_health (tg-app target states and reason codes), asg_activity (Auto Scaling activity history),
  console_a / console_b (boot log of the instance in AZ a / b), read_app_logs (recent application errors in CloudWatch Logs),
  flow_logs (VPC flow-log summary at an application instance ENI: inbound health-check port, outbound tcp/443).
Instance shell via Session Manager (suffix @a or @b selects the instance): curl_localhost_health@a, curl_internet@a,
  ping_internet@a, nc_rds_3306@a, ping_rds@a, s3_put_test@a, sts_identity@a (same with @b).
Configuration inspection (shows the deployed settings): c_rt_pub, c_rt_app_a, c_rt_app_b, c_igw, c_nat_a, c_nat_b,
  c_nacl_pub, c_nacl_app, c_nacl_db, c_sg_alb, c_sg_app, c_sg_db, c_alb, c_tg, c_asg, c_lt, c_role, c_bucket, c_rds, c_appcfg, c_cf.
"""

PROTOCOL = """You are diagnosing a deliberately broken cloud deployment in a training exercise. Exactly one setting differs from
the intended design. Find it with as few actions as possible. Reply with ONE JSON object and nothing else, either
{"action": "<action name>"} or {"diagnosis": {"resource": "<one of: %s>", "misconfiguration": "<what is wrong, one sentence>"}}.
You have at most 15 turns and at most 3 diagnoses. After a diagnosis you are told whether the repair fixed the incident.""" % ', '.join(RESOURCES)


def ticket_text(cfg):
    cache = {}
    lines = ['INCIDENT TICKET (synthetic monitors)']
    labels = {'cf_root': 'GET https://site/ via CloudFront', 'cf_items': 'GET /items (database-backed page) via CloudFront',
              'cf_upload': 'POST /upload (stores a file in S3) via CloudFront', 'cf_static': 'GET /static/logo.png via CloudFront',
              'az_test_a': 'Resilience test: AZ a taken offline', 'az_test_b': 'Resilience test: AZ b taken offline',
              'alarm_unhealthy': 'CloudWatch alarm UnHealthyHostCount>0', 'ssm_available': 'Instances managed by Systems Manager',
              'logs_present': 'Application log streams receiving events'}
    for m in cs.MONITORS:
        lines.append('- %s: %s' % (labels[m], cs.observe(cfg, m, cs.TRUE_SEM, cache)))
    return '\n'.join(lines)


def parse(text):
    t = text.strip()
    m = re.search(r'\{.*\}', t, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0), strict=False)
    except Exception:
        return None


RUBRIC = ("ACCEPT if the statement names the changed setting and either its wrong value/omission or the intended correct "
          "value, so that applying it would restore the design; REJECT if it names a different setting of the same "
          "resource, a different resource, or is too vague to apply.")

JUDGE_TEMPLATE = """You are grading one diagnosis submitted in a cloud-troubleshooting training exercise. Exactly one setting
of the deployment was changed from the intended design below.

%s
CHANGED SETTING (gold fault): resource=%s; "%s"

SUBMITTED DIAGNOSIS: resource=%s; misconfiguration: "%s"

RUBRIC: %s

Reply with ONE JSON object and nothing else: {"verdict": "ACCEPT" or "REJECT", "reason": "<one sentence>"}"""

JUDGE_MODEL = 'claude-sonnet-5'
GRADER2_MODEL = 'claude-opus-5'
JUDGE_MAX_TOKENS = 200


def judge_prompt(fid, resource, text):
    return JUDGE_TEMPLATE % (DESIGN, GOLD_RES[fid], cs.FAULT_BY_ID[fid]['desc'], resource, text, RUBRIC)


def _usage(o):
    u = o.get('usage') or {}
    return dict(input=u.get('input_tokens', 0) or 0, output=u.get('output_tokens', 0) or 0,
                cache_read=u.get('cache_read_input_tokens', 0) or 0,
                cache_write=u.get('cache_creation_input_tokens', 0) or 0)


def grade(host, items, model, max_concurrency=16, attempts=3):
    """Apply the rubric to a list of (fid, resource, text). Returns one dict per item:
    verdict in {'ACCEPT','REJECT','UNPARSED'}, reason, raw text, model id, usage, attempts used."""
    res = [None] * len(items)
    todo = list(range(len(items)))
    tries = [0] * len(items)
    usage = [dict(input=0, output=0, cache_read=0, cache_write=0) for _ in items]
    raws = [[] for _ in items]
    while todo:
        reqs = [dict(prompt=judge_prompt(*items[i]), model=model, max_tokens=JUDGE_MAX_TOKENS,
                     thinking={'type': 'disabled'}) for i in todo]
        outs = host.llm(reqs, max_concurrency=max_concurrency)
        nxt = []
        for i, o in zip(todo, outs):
            tries[i] += 1
            if 'error' in o and o.get('error'):
                raws[i].append('ERROR: %s' % o['error'])
                js = None
                mid = None
            else:
                for k, v in _usage(o).items():
                    usage[i][k] += v
                raws[i].append(o.get('text') or '')
                js = parse(o.get('text') or '')
                mid = o.get('model')
            v = str(js.get('verdict', '')).strip().upper() if isinstance(js, dict) else ''
            if v in ('ACCEPT', 'REJECT'):
                res[i] = dict(verdict=v, reason=str(js.get('reason', '')), raw=raws[i], model_id=mid,
                              usage=usage[i], attempts=tries[i])
            elif tries[i] >= attempts:
                res[i] = dict(verdict='UNPARSED', reason='', raw=raws[i], model_id=mid, usage=usage[i], attempts=tries[i])
            else:
                nxt.append(i)
        todo = nxt
    return res


def run(host, fids, model, max_turns=15, max_diag=3, label='model', run_id=1, max_tokens=400, thinking_off=True,
        judge_model=JUDGE_MODEL, max_concurrency=16):
    """Run one replicate of all episodes in `fids` for one agent model, with the setting-sensitive verify loop.
    Temperature is never set (API default sampling)."""
    eps = {}
    for f in fids:
        cfg = cs.faulted(f)
        first = DESIGN + '\n' + ACTION_HELP + '\n' + ticket_text(cfg) + '\n\n' + PROTOCOL
        eps[f] = dict(agent=label, run=run_id, fault=f, gold_resource=GOLD_RES[f], gold_desc=cs.FAULT_BY_ID[f]['desc'],
                      cfg=cfg, cache={}, msgs=[{'role': 'user', 'content': first}], actions=[], diags=[],
                      done=False, outcome=None, turns=0, invalid=0, errors=0, truncated=0,
                      usage=dict(input=0, output=0, cache_read=0, cache_write=0),
                      judge_usage=dict(input=0, output=0, cache_read=0, cache_write=0),
                      model_ids=[], omitted=[])
    t_start = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    for turn in range(max_turns):
        active = [f for f in fids if not eps[f]['done']]
        if not active:
            break
        reqs = []
        for f in active:
            r = dict(messages=eps[f]['msgs'], max_tokens=max_tokens)
            if model:
                r['model'] = model
            if thinking_off:
                r['thinking'] = {'type': 'disabled'}
            reqs.append(r)
        outs = host.llm(reqs, max_concurrency=max_concurrency)
        pending = []   # (f, diag) awaiting the in-loop judge
        for f, o in zip(active, outs):
            e = eps[f]
            e['turns'] += 1
            if 'error' in o and o.get('error'):
                e['msgs'].append({'role': 'assistant', 'content': '{}'})
                e['msgs'].append({'role': 'user', 'content': 'Error: request failed; reply with one JSON object.'})
                e['invalid'] += 1
                e['errors'] += 1
                e.setdefault('error_texts', []).append(str(o['error']))
                if e['errors'] >= 3:
                    e['done'], e['outcome'] = True, 'api_error'
                continue
            for k, v in _usage(o).items():
                e['usage'][k] += v
            e['model_ids'].append(o.get('model'))
            if o.get('omitted'):
                e['omitted'].append(o.get('omitted'))
            if o.get('stop_reason') == 'max_tokens':
                e['truncated'] += 1
            txt = o.get('text') or ''
            e['msgs'].append({'role': 'assistant', 'content': txt or '{}'})
            js = parse(txt)
            if js and 'action' in js and js['action'] in cs.ACTIONS:
                a = js['action']
                e['actions'].append(a)
                obs = cs.observe(e['cfg'], a, cs.TRUE_SEM, e['cache'])
                e['msgs'].append({'role': 'user', 'content': 'Result of %s: %s\nTurns left: %d' % (a, obs, max_turns - e['turns'])})
            elif js and 'diagnosis' in js and isinstance(js['diagnosis'], dict):
                d = js['diagnosis']
                dg = dict(resource=str(d.get('resource', '')).strip(), text=str(d.get('misconfiguration', '')),
                          turn=e['turns'], n_actions=len(e['actions']))
                dg['resource_ok'] = dg['resource'] == GOLD_RES[f]
                e['diags'].append(dg)
                if dg['resource_ok']:
                    pending.append((f, dg))
                else:
                    dg['judge'] = 'wrong_resource'
                    _after_reject(e, dg, max_turns, max_diag)
            else:
                e['invalid'] += 1
                e['msgs'].append({'role': 'user', 'content': 'Invalid reply. Reply with ONE JSON object: an action from the list or a diagnosis.'})
        if pending:
            verdicts = grade(host, [(f, dg['resource'], dg['text']) for f, dg in pending], judge_model,
                             max_concurrency=max_concurrency)
            for (f, dg), v in zip(pending, verdicts):
                e = eps[f]
                dg['judge'] = v['verdict']
                dg['judge_reason'] = v['reason']
                dg['judge_raw'] = v['raw']
                dg['judge_model_id'] = v['model_id']
                dg['judge_attempts'] = v['attempts']
                for k, x in v['usage'].items():
                    e['judge_usage'][k] += x
                if v['verdict'] == 'ACCEPT':
                    e['msgs'].append({'role': 'user', 'content': 'Repair applied to %s. The incident is resolved.' % dg['resource']})
                    e['done'], e['outcome'] = True, 'solved'
                else:
                    _after_reject(e, dg, max_turns, max_diag)
    for f in fids:
        if not eps[f]['done']:
            eps[f]['done'], eps[f]['outcome'] = True, 'out_of_turns'
    t_end = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    for f in fids:
        eps[f]['t_start_utc'], eps[f]['t_end_utc'] = t_start, t_end
    return eps


def _after_reject(e, dg, max_turns, max_diag):
    """Verify-loop reply for a repair that did not fix the incident (wrong resource or judge REJECT)."""
    if len(e['diags']) >= max_diag:
        e['done'], e['outcome'] = True, 'failed'
    else:
        e['msgs'].append({'role': 'user', 'content': 'Repair applied to %s. All monitors unchanged; the incident persists. Turns left: %d'
                          % (dg['resource'], max_turns - e['turns'])})


def grade2(host, eps_list, model=GRADER2_MODEL, max_concurrency=16):
    """Independent post-hoc second grader: same rubric, every diagnosis that named the gold resource."""
    items, refs = [], []
    for e in eps_list:
        for dg in e['diags']:
            if dg['resource_ok']:
                items.append((e['fault'], dg['resource'], dg['text']))
                refs.append(dg)
    out = grade(host, items, model, max_concurrency=max_concurrency)
    for dg, v in zip(refs, out):
        dg['grader2'] = v['verdict']
        dg['grader2_reason'] = v['reason']
        dg['grader2_raw'] = v['raw']
        dg['grader2_model_id'] = v['model_id']
        dg['grader2_usage'] = v['usage']
    return out
