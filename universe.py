"""
universe.py - systematic single-field mutation universe of the reference configuration.

Every mutation changes exactly one configuration element to another value from a small,
type-appropriate value domain (route targets, subnet placements, port numbers, rule
sources, flags, permissions). The universe is the learner's hypothesis space: a
misconfiguration could be any of these, whereas break-fix tasks inject only the curated
faults in cloudsim.FAULTS (all of which are members of the universe).
"""
import copy
import itertools

import cloudsim as cs

SUBNETS = ['pub-a', 'pub-b', 'app-a', 'app-b', 'db-a', 'db-b']
PORTS = [22, 80, 443, 3306, 8080]
SRC_CLS = ['any', 'vpc', 'pub', 'app', 'db']
SG_SRC = ['any', 'vpc', 'sg-alb', 'sg-app', 'sg-db']


def _canon(cfg):
    return repr(sorted((k, repr(v)) for k, v in cfg.items()))


def generate():
    base = cs.base_config()
    muts = []  # (mid, resource, description, cfg)

    def add(resource, desc, fn, admissible=True):
        c = copy.deepcopy(base)
        fn(c)
        muts.append((resource, desc, c, admissible))

    # route tables
    for rt, cur in base['rts'].items():
        for v in ['igw', 'nat-a', 'nat-b', None]:
            if v != cur:
                add(rt, '%s default route -> %s' % (rt, v), lambda c, rt=rt, v=v: c['rts'].__setitem__(rt, v))
    # subnet route-table associations
    for s, (az, rt, nacl) in base['subnets'].items():
        for rt2 in base['rts']:
            if rt2 != rt:
                add('subnet-assoc', '%s associated with %s' % (s, rt2),
                    lambda c, s=s, rt2=rt2: c['subnets'].__setitem__(s, (c['subnets'][s][0], rt2, c['subnets'][s][2])))
    add('igw', 'internet gateway detached', lambda c: c.__setitem__('igw_attached', False))
    # NAT gateways
    for n, d in base['nats'].items():
        for s in SUBNETS:
            if s != d['subnet']:
                add(n, '%s placed in %s' % (n, s), lambda c, n=n, s=s: c['nats'][n].__setitem__('subnet', s))
        add(n, '%s created with private connectivity (no Elastic IP)' % n, lambda c, n=n: c['nats'][n].__setitem__('eip', False))
    # network ACLs: delete a rule, or change its source/destination class
    for nacl, d in base['nacls'].items():
        for direc in ('inb', 'outb'):
            for k, rule in enumerate(d[direc]):
                add(nacl, '%s %s rule %d deleted' % (nacl, direc, rule[0]),
                    lambda c, nacl=nacl, direc=direc, k=k: c['nacls'][nacl][direc].pop(k))
                for cl in SRC_CLS:
                    if cl != rule[5]:
                        add(nacl, '%s %s rule %d peer -> %s' % (nacl, direc, rule[0], cl),
                            lambda c, nacl=nacl, direc=direc, k=k, cl=cl:
                            c['nacls'][nacl][direc].__setitem__(k, c['nacls'][nacl][direc][k][:5] + (cl,)))
                if rule[2] == 'tcp':
                    for p in PORTS:
                        if not (rule[3] <= p <= rule[4]) or rule[3] != rule[4]:
                            if rule[3] == rule[4] and p == rule[3]:
                                continue
                            add(nacl, '%s %s rule %d port -> %d' % (nacl, direc, rule[0], p),
                                lambda c, nacl=nacl, direc=direc, k=k, p=p:
                                c['nacls'][nacl][direc].__setitem__(k, c['nacls'][nacl][direc][k][:3] + (p, p) + c['nacls'][nacl][direc][k][5:]))
        # whole custom ACL that allows only inbound 80/443 (common hand-made ACL)
        if nacl == 'nacl-pub':
            add(nacl, 'nacl-pub inbound only 80/443',
                lambda c: c['nacls'].__setitem__('nacl-pub', dict(inb=[(100, 'allow', 'tcp', 80, 80, 'any'), (110, 'allow', 'tcp', 443, 443, 'any')], outb=list(cs.ALLOW_ALL))))
    # security groups: delete rule, change port, change source
    for sg, d in base['sgs'].items():
        for direc in ('inb', 'outb'):
            for k, rule in enumerate(d[direc]):
                add(sg, '%s %s rule %d deleted' % (sg, direc, k), lambda c, sg=sg, direc=direc, k=k: c['sgs'][sg][direc].pop(k))
                for src in SG_SRC:
                    if src != rule[3]:
                        add(sg, '%s %s rule %d peer -> %s' % (sg, direc, k, src),
                            lambda c, sg=sg, direc=direc, k=k, src=src: c['sgs'][sg][direc].__setitem__(k, c['sgs'][sg][direc][k][:3] + (src,)))
                for p in PORTS:
                    if rule[0] == 'all' or p != rule[1]:
                        add(sg, '%s %s rule %d -> tcp/%d' % (sg, direc, k, p),
                            lambda c, sg=sg, direc=direc, k=k, p=p: c['sgs'][sg][direc].__setitem__(k, ('tcp', p, p, c['sgs'][sg][direc][k][3])))
    # load balancer
    pubapp = ['pub-a', 'pub-b', 'app-a', 'app-b']
    # An ALB needs subnets in at least two Availability Zones and at most one subnet per zone.
    # Single-subnet and same-zone combinations are generated (so that mutant identifiers stay
    # comparable with the first version of the model) but flagged inadmissible and dropped.
    for combo in list(itertools.combinations(pubapp, 1)) + list(itertools.combinations(pubapp, 2)):
        ok = len(combo) == 2 and base['subnets'][combo[0]][0] != base['subnets'][combo[1]][0]
        if list(combo) != base['alb']['subnets']:
            add('alb', 'ALB subnets -> %s' % '/'.join(combo), lambda c, combo=combo: c['alb'].__setitem__('subnets', list(combo)), admissible=ok)
    add('alb', 'listener :80 -> tg-old', lambda c: c['alb'].__setitem__('listeners', {80: 'tg-old'}))
    add('alb', 'ALB security group -> sg-app', lambda c: c['alb'].__setitem__('sg', 'sg-app'))
    # target group
    for p in [443, 3306, 8080]:
        add('tg-app', 'target group port -> %d' % p, lambda c, p=p: c['tgs']['tg-app'].__setitem__('port', p))
    for hp in ['/', '/healthz', '/index.html']:
        add('tg-app', 'health-check path -> %s' % hp, lambda c, hp=hp: c['tgs']['tg-app'].__setitem__('hc_path', hp))
    for m in [301, 302]:
        add('tg-app', 'health-check matcher -> %d' % m, lambda c, m=m: c['tgs']['tg-app'].__setitem__('matcher', m))
    # auto scaling group
    for combo in list(itertools.combinations(pubapp + ['db-a', 'db-b'], 1)) + list(itertools.combinations(pubapp + ['db-a', 'db-b'], 2)):
        if list(combo) != base['asg']['subnets']:
            add('asg', 'ASG subnets -> %s' % '/'.join(combo), lambda c, combo=combo: c['asg'].__setitem__('subnets', list(combo)))
    for dsr in [0, 1]:
        add('asg', 'ASG desired -> %d' % dsr, lambda c, dsr=dsr: c['asg'].__setitem__('desired', dsr))
    for tg in [None, 'tg-old']:
        add('asg', 'ASG target group -> %s' % tg, lambda c, tg=tg: c['asg'].__setitem__('tg', tg))
    add('asg', 'ASG health-check type -> EC2', lambda c: c['asg'].__setitem__('hc_type', 'EC2'))
    # launch template
    add('lt', 'AMI deregistered', lambda c: c['lt'].__setitem__('ami_exists', False))
    add('lt', 'no instance profile', lambda c: c['lt'].__setitem__('profile', None))
    for sg in ['sg-alb', 'sg-db']:
        add('lt', 'instance SG -> %s' % sg, lambda c, sg=sg: c['lt'].__setitem__('sg', sg))
    add('lt', 'public IPv4 assigned', lambda c: c['lt'].__setitem__('public_ip', True))
    for p in [443, 8080]:
        add('lt', 'web server port -> %d' % p, lambda c, p=p: c['lt'].__setitem__('app_port', p))
    add('lt', 'user-data script error', lambda c: c['lt'].__setitem__('userdata_ok', False))
    add('app', '/health redirects (301)', lambda c: c['lt'].__setitem__('health_redirect', True))
    # IAM role
    add('role', 'trust policy -> lambda.amazonaws.com', lambda c: c['role'].__setitem__('trust', 'lambda.amazonaws.com'))
    for p in list(base['role']['allow']):
        add('role', 'role lacks %s' % p, lambda c, p=p: c['role']['allow'].pop(p))
    for p in ['s3:GetObject', 's3:PutObject']:
        add('role', '%s resource -> bucket ARN' % p, lambda c, p=p: c['role']['allow'].__setitem__(p, 'arn:aws:s3:::assets-bkt'))
    add('role', 'S3 resources -> bucket ARN (both)', lambda c: c['role']['allow'].update({'s3:GetObject': 'arn:aws:s3:::assets-bkt', 's3:PutObject': 'arn:aws:s3:::assets-bkt'}))
    # bucket policy
    add('bucket', 'bucket policy denies PutObject to role', lambda c: c['bucket'].__setitem__('deny_put_role', True))
    add('bucket', 'bucket policy lacks OAC grant', lambda c: c['bucket'].__setitem__('oac_grant', False))
    # database
    for sg in ['sg-app', 'sg-alb']:
        add('rds', 'RDS security group -> %s' % sg, lambda c, sg=sg: c['rds'].__setitem__('sg', sg))
    add('rds', 'RDS single-AZ', lambda c: c['rds'].__setitem__('multi_az', False))
    add('rds', 'RDS port -> 3307', lambda c: c['rds'].__setitem__('port', 3307))
    add('appcfg', 'stale DB endpoint in app config', lambda c: c['appcfg'].__setitem__('db_endpoint', 'appdb.old'))
    add('appcfg', 'stale DB password in app config', lambda c: c['appcfg'].__setitem__('db_password', 'old'))
    # CloudFront
    add('cf', 'CloudFront origin -> deleted ALB', lambda c: c['cf'].__setitem__('origin', 'alb-old'))
    add('cf', 'CloudFront origin protocol -> https-only', lambda c: c['cf'].__setitem__('origin_protocol', 'https-only'))

    # dedupe by resulting configuration; drop identity mutations
    base_key = _canon(base)
    seen, uni, n = {}, [], 0
    for (res, desc, c, admissible) in muts:
        k = _canon(c)
        if k == base_key or k in seen:
            continue
        mid = 'U%03d' % n
        n += 1
        if not admissible:
            seen[k] = None
            continue
        seen[k] = len(uni)
        uni.append(dict(mid=mid, resource=res, desc=desc, cfg=c))
    # map curated faults onto universe members (add any that are not single-field mutations)
    curated = {}
    for f in cs.FAULTS:
        k = _canon(cs.faulted(f['id']))
        if seen.get(k) is not None:
            curated[f['id']] = uni[seen[k]]['mid']
        else:
            assert k not in seen, 'curated fault %s is an inadmissible configuration' % f['id']
            m = dict(mid='U%03d' % n, resource=f['resource'], desc=f['desc'], cfg=cs.faulted(f['id']))
            seen[k] = len(uni)
            uni.append(m)
            curated[f['id']] = m['mid']
    return uni, curated
