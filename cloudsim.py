"""
cloudsim.py - executable fault model of a three-tier AWS-style reference architecture
for break-fix (fault-injection) diagnosis tasks.

The model is deliberately symbolic: addresses are represented by location classes
(internet / subnet), and every observable is a deterministic function of the
configuration, an optional Availability-Zone outage, and a semantics vector `sem`.
`sem` = TRUE_SEM reproduces documented platform behaviour; misconception models
flip individual semantic flags (see misconceptions.py).

Documented behaviours encoded (AWS documentation):
  * security groups are stateful (return traffic always allowed);
  * network ACLs are stateless (return traffic needs an explicit ephemeral-port rule)
    and are evaluated in rule-number order with an implicit final deny; traffic that
    stays inside one subnet is not filtered by its NACL;
  * an instance reaches the internet through an internet gateway only if it has a
    public IPv4 address; private instances need a NAT gateway placed in a public subnet;
  * an ALB does not answer ICMP echo requests;
  * an ALB routes only to healthy targets in enabled AZs; if no target is healthy it
    "fails open" and routes to all in-use targets;
  * target-health reason codes: Target.Timeout, Target.FailedHealthChecks,
    Target.ResponseCodeMismatch, Target.NotInUse;
  * IAM: an explicit Deny in a resource (bucket) policy overrides any Allow;
  * instance credentials come from an instance profile whose role trusts ec2.amazonaws.com;
  * CloudFront reports distinct 502 messages for origin DNS failure and connection failure,
    and 504 for origin time-outs; S3 origins with origin access control need a bucket-policy
    grant to the CloudFront service principal.
"""

EPH = (1024, 65535)

TRUE_SEM = dict(icmp_as_tcp=False, sg_stateless=False, nacl_ignored=False,
                resource_policy_ignored=False, igw_route_suffices=False,
                instance_has_user_perms=False, hc_root_path=False, nat_private_ok=False)

ALLOW_ALL = [(100, 'allow', 'all', 0, 65535, 'any')]


def base_config():
    return dict(
        igw_attached=True,
        subnets={  # name: (az, route table, nacl)
            'pub-a': ('a', 'rt-pub', 'nacl-pub'), 'pub-b': ('b', 'rt-pub', 'nacl-pub'),
            'app-a': ('a', 'rt-app-a', 'nacl-app'), 'app-b': ('b', 'rt-app-b', 'nacl-app'),
            'db-a': ('a', 'rt-db', 'nacl-db'), 'db-b': ('b', 'rt-db', 'nacl-db')},
        rts={'rt-pub': 'igw', 'rt-app-a': 'nat-a', 'rt-app-b': 'nat-b', 'rt-db': None},
        nats={'nat-a': dict(subnet='pub-a', eip=True), 'nat-b': dict(subnet='pub-b', eip=True)},
        nacls={
            'nacl-pub': dict(inb=list(ALLOW_ALL), outb=list(ALLOW_ALL)),
            'nacl-app': dict(inb=[(100, 'allow', 'tcp', 80, 80, 'vpc'),
                                  (110, 'allow', 'tcp', 1024, 65535, 'any')],
                             outb=list(ALLOW_ALL)),
            'nacl-db': dict(inb=[(100, 'allow', 'tcp', 3306, 3306, 'app')],
                            outb=[(100, 'allow', 'tcp', 1024, 65535, 'app')])},
        sgs={
            'sg-alb': dict(inb=[('tcp', 80, 80, 'any'), ('tcp', 443, 443, 'any')],
                           outb=[('all', 0, 65535, 'any')]),
            'sg-app': dict(inb=[('tcp', 80, 80, 'sg-alb')], outb=[('all', 0, 65535, 'any')]),
            'sg-db': dict(inb=[('tcp', 3306, 3306, 'sg-app')], outb=[('all', 0, 65535, 'any')])},
        alb=dict(subnets=['pub-a', 'pub-b'], sg='sg-alb', listeners={80: 'tg-app'}),
        tgs={'tg-app': dict(port=80, hc_path='/health', matcher=200),
             'tg-old': dict(port=80, hc_path='/', matcher=200)},
        asg=dict(subnets=['app-a', 'app-b'], desired=2, tg='tg-app', hc_type='ELB'),
        lt=dict(ami='ami-web-v3', ami_exists=True, profile='role-app', sg='sg-app',
                public_ip=False, app_port=80, userdata_ok=True, health_redirect=False),
        role=dict(trust='ec2.amazonaws.com',
                  allow={'s3:GetObject': 'arn:aws:s3:::assets-bkt/*',
                         's3:PutObject': 'arn:aws:s3:::assets-bkt/*',
                         'logs:PutLogEvents': '*', 'ssm:UpdateInstanceInformation': '*'}),
        bucket=dict(deny_put_role=False, oac_grant=True),
        rds=dict(subnets=['db-a', 'db-b'], sg='sg-db', multi_az=True, port=3306),
        appcfg=dict(db_endpoint='appdb.current', db_password='current'),
        cf=dict(origin='alb-current', origin_protocol='http-only'),
    )


# ----------------------------------------------------------------------------- helpers
def sub_az(cfg, s):
    return cfg['subnets'][s][0]


def cls_match(cidr, loc):
    """CIDR class ('any', 'vpc', 'pub', 'app', 'db') vs location ('internet' or a subnet)."""
    if cidr == 'any':
        return True
    if loc == 'internet':
        return False
    if cidr == 'vpc':
        return True
    return loc.startswith(cidr + '-')


def nacl_allows(cfg, sem, subnet, direction, proto, port, peer_loc):
    if sem['nacl_ignored']:
        return True
    rules = cfg['nacls'][cfg['subnets'][subnet][2]]['inb' if direction == 'in' else 'outb']
    for (num, act, p, lo, hi, cidr) in sorted(rules):
        if p != 'all' and p != proto:
            continue
        if proto == 'tcp' and not (lo <= port <= hi):
            continue
        if not cls_match(cidr, peer_loc):
            continue
        return act == 'allow'
    return False  # implicit deny


def sg_allows(cfg, sg, direction, proto, port, peer):
    rules = cfg['sgs'][sg]['inb' if direction == 'in' else 'outb']
    for (p, lo, hi, src) in rules:
        if p != 'all' and p != proto:
            continue
        if proto == 'tcp' and p != 'all' and not (lo <= port <= hi):
            continue
        if src.startswith('sg-'):
            if src in peer.get('sgs', ()):
                return True
            continue
        if cls_match(src, peer['loc']):
            return True
    return False


def ep(loc, sgs=(), public_ip=False, kind='host', name=''):
    return dict(loc=loc, sgs=tuple(sgs), public_ip=public_ip, kind=kind, name=name)


INTERNET = ep('internet', kind='internet', name='internet')


def subnet_up(s, az_down, cfg):
    return az_down is None or sub_az(cfg, s) != az_down


# ----------------------------------------------------------------------------- network reachability
def conn(cfg, sem, src, dst, port, proto='tcp', az_down=None):
    """Network-level reachability of src->dst:port (does not check that a service
    listens). Returns (ok, blocking_element)."""
    sl, dl = src['loc'], dst['loc']
    eport = EPH[0] if proto == 'tcp' else 0
    if sl == 'internet':  # ---- internet -> VPC
        if not subnet_up(dl, az_down, cfg):
            return False, 'az-down'
        rt = cfg['subnets'][dl][1]
        if cfg['rts'][rt] != 'igw' or not cfg['igw_attached']:
            return False, 'route:' + rt
        if not (dst['public_ip'] or dst['kind'] == 'alb'):
            return False, 'no-public-ip'
        if not nacl_allows(cfg, sem, dl, 'in', proto, port, 'internet'):
            return False, 'nacl-in:' + cfg['subnets'][dl][2]
        if not nacl_allows(cfg, sem, dl, 'out', proto, eport, 'internet'):
            return False, 'nacl-out:' + cfg['subnets'][dl][2]
        if not any(sg_allows(cfg, g, 'in', proto, port, src) for g in dst['sgs']):
            return False, 'sg:' + ','.join(dst['sgs'])
        if sem['sg_stateless'] and not any(sg_allows(cfg, g, 'out', proto, eport, src) for g in dst['sgs']):
            return False, 'sg-egress:' + ','.join(dst['sgs'])
        return True, None
    if not subnet_up(sl, az_down, cfg):
        return False, 'az-down'
    target_peer = INTERNET if dl == 'internet' else dst
    if not any(sg_allows(cfg, g, 'out', proto, port, target_peer) for g in src['sgs']):
        return False, 'sg-egress:' + ','.join(src['sgs'])
    if dl == 'internet':  # ---- VPC -> internet
        rt = cfg['subnets'][sl][1]
        tgt = cfg['rts'][rt]
        if tgt is None:
            return False, 'route:' + rt
        if tgt == 'igw':
            if not cfg['igw_attached']:
                return False, 'igw'
            if not (src['public_ip'] or sem['igw_route_suffices']):
                return False, 'no-public-ip'
            if not nacl_allows(cfg, sem, sl, 'out', proto, port, 'internet'):
                return False, 'nacl-out:' + cfg['subnets'][sl][2]
            if not nacl_allows(cfg, sem, sl, 'in', proto, eport, 'internet'):
                return False, 'nacl-in:' + cfg['subnets'][sl][2]
        else:  # NAT gateway
            nat = cfg['nats'].get(tgt)
            if nat is None:
                return False, 'route:' + rt
            ns = nat['subnet']
            if not subnet_up(ns, az_down, cfg):
                return False, 'nat-az-down'
            nrt = cfg['subnets'][ns][1]
            nat_ok = cfg['rts'][nrt] == 'igw' and cfg['igw_attached'] and nat['eip']
            if not nat_ok and not (sem['nat_private_ok'] and cfg['igw_attached'] and nat['eip']
                                   and cfg['rts']['rt-pub'] == 'igw'):
                return False, 'nat:' + tgt
            if not nacl_allows(cfg, sem, sl, 'out', proto, port, 'internet'):
                return False, 'nacl-out:' + cfg['subnets'][sl][2]
            if not nacl_allows(cfg, sem, sl, 'in', proto, eport, 'internet'):
                return False, 'nacl-in:' + cfg['subnets'][sl][2]
            if ns != sl and nat_ok:
                for (d, pp, peer) in [('in', port, sl), ('out', eport, sl), ('out', port, 'internet'), ('in', eport, 'internet')]:
                    if not nacl_allows(cfg, sem, ns, d, proto, pp, peer):
                        return False, ('nacl-%s:' % d) + cfg['subnets'][ns][2]
        return True, None
    # ---- VPC -> VPC (local route)
    if not subnet_up(dl, az_down, cfg):
        return False, 'az-down'
    if sl != dl:
        if not nacl_allows(cfg, sem, sl, 'out', proto, port, dl):
            return False, 'nacl-out:' + cfg['subnets'][sl][2]
        if not nacl_allows(cfg, sem, dl, 'in', proto, port, sl):
            return False, 'nacl-in:' + cfg['subnets'][dl][2]
        if not nacl_allows(cfg, sem, dl, 'out', proto, eport, sl):
            return False, 'nacl-out:' + cfg['subnets'][dl][2]
        if not nacl_allows(cfg, sem, sl, 'in', proto, eport, dl):
            return False, 'nacl-in:' + cfg['subnets'][sl][2]
    if not any(sg_allows(cfg, g, 'in', proto, port, src) for g in dst['sgs']):
        return False, 'sg:' + ','.join(dst['sgs'])
    if sem['sg_stateless']:
        if not any(sg_allows(cfg, g, 'out', proto, eport, src) for g in dst['sgs']):
            return False, 'sg-egress:' + ','.join(dst['sgs'])
    return True, None


# ----------------------------------------------------------------------------- derived state
def s3_allowed(cfg, sem, action, creds):
    if not creds:
        return False, 'Unable to locate credentials'
    if not sem['instance_has_user_perms']:
        res = cfg['role']['allow'].get(action)
        if res is None or not (res == '*' or res.endswith('/*')):
            return False, 'AccessDenied'
    if action == 's3:PutObject' and cfg['bucket']['deny_put_role'] and not sem['resource_policy_ignored']:
        return False, 'AccessDenied'
    return True, None


def has_perm(cfg, sem, action):
    return sem['instance_has_user_perms'] or action in cfg['role']['allow']


def rds_endpoint(cfg, az_down):
    prim = cfg['rds']['subnets'][0]
    if subnet_up(prim, az_down, cfg):
        return ep(prim, [cfg['rds']['sg']], kind='rds', name='rds'), True
    if cfg['rds']['multi_az']:
        stb = cfg['rds']['subnets'][1]
        if subnet_up(stb, az_down, cfg):
            return ep(stb, [cfg['rds']['sg']], kind='rds', name='rds'), True
    return ep(prim, [cfg['rds']['sg']], kind='rds', name='rds'), False


def app_code(cfg, path):
    if path == '/health':
        return 301 if cfg['lt']['health_redirect'] else 200
    if path in ('/', '/items', '/upload'):
        return 200
    return 404


def derive(cfg, sem=TRUE_SEM, az_down=None):
    """Instances, boot results, target health and ALB state."""
    st = {}
    lt, asg = cfg['lt'], cfg['asg']
    up_subs = [s for s in asg['subnets'] if subnet_up(s, az_down, cfg)]
    insts, launch_error = [], None
    if asg['desired'] > 0 and up_subs:
        if not lt['ami_exists']:
            launch_error = "The image id '[%s]' does not exist" % lt['ami']
        else:
            for k in range(asg['desired']):
                s = up_subs[k % len(up_subs)]
                insts.append(dict(name='i-%d%s' % (k + 1, sub_az(cfg, s)), subnet=s, az=sub_az(cfg, s)))
    creds = (lt['profile'] is not None and cfg['role']['trust'] == 'ec2.amazonaws.com') or sem['instance_has_user_perms']
    rds_ep, rds_alive = rds_endpoint(cfg, az_down)
    for i in insts:
        e = ep(i['subnet'], [lt['sg']], public_ip=lt['public_ip'], kind='instance', name=i['name'])
        i['ep'] = e
        net443, blk = conn(cfg, sem, e, INTERNET, 443, az_down=az_down)
        i['net443'], i['net443_blk'] = net443, blk
        if not lt['userdata_ok']:
            i['boot'], i['app_up'] = 'user-data: script error at line 14; httpd not started', False
        elif not creds:
            i['boot'], i['app_up'] = 'fetch-config: Unable to locate credentials', False
        elif not net443:
            i['boot'], i['app_up'] = 'fetch-config: connect timeout s3.amazonaws.com:443', False
        else:
            ok, err = s3_allowed(cfg, sem, 's3:GetObject', creds)
            if not ok:
                i['boot'], i['app_up'] = 'fetch-config: AccessDenied (GetObject config/app.json)', False
            else:
                i['boot'], i['app_up'] = 'fetch-config: ok; httpd listening on :%d' % lt['app_port'], True
        i['ssm'] = creds and net443 and has_perm(cfg, sem, 'ssm:UpdateInstanceInformation')
        i['logs'] = creds and net443 and has_perm(cfg, sem, 'logs:PutLogEvents')
        dbn, dbblk = conn(cfg, sem, e, rds_ep, cfg['rds']['port'], az_down=az_down)
        i['db_net'] = dbn and rds_alive
    st['insts'], st['launch_error'], st['creds'] = insts, launch_error, creds
    st['rds_ep'], st['rds_alive'] = rds_ep, rds_alive
    alb_azs = {sub_az(cfg, s) for s in cfg['alb']['subnets']}
    nodes = [ep(s, [cfg['alb']['sg']], kind='alb', name='alb-' + s) for s in cfg['alb']['subnets'] if subnet_up(s, az_down, cfg)]
    st['alb_nodes'] = nodes
    th = {}
    reg_tg = asg['tg']
    for i in insts:
        if reg_tg is None:
            continue
        tg = cfg['tgs'][reg_tg]
        if reg_tg not in cfg['alb']['listeners'].values() or i['az'] not in alb_azs:
            th[i['name']] = 'unused (Target.NotInUse)'
            continue
        ok = any(conn(cfg, sem, n, i['ep'], tg['port'], az_down=az_down)[0] for n in nodes)
        if not ok:
            th[i['name']] = 'unhealthy (Target.Timeout)'
        elif not (i['app_up'] and lt['app_port'] == tg['port']):
            th[i['name']] = 'unhealthy (Target.FailedHealthChecks)'
        else:
            path = '/' if sem['hc_root_path'] else tg['hc_path']
            code = app_code(cfg, path)
            th[i['name']] = 'healthy' if code == tg['matcher'] else 'unhealthy (Target.ResponseCodeMismatch [%d])' % code
    st['th'] = th
    st['churn'] = asg['hc_type'] == 'ELB' and any(v.startswith('unhealthy') for v in th.values())
    return st


def app_response(cfg, sem, st, inst, path):
    if not inst['app_up']:
        return None
    if path == '/items':
        if not inst['db_net']:
            return '500 db-timeout'
        if cfg['appcfg']['db_endpoint'] != 'appdb.current':
            return '500 db-unknown-host'
        if cfg['appcfg']['db_password'] != 'current':
            return '500 db-auth-failed'
        return '200'
    if path == '/upload':
        if not inst['net443']:
            return '500 s3-timeout'
        ok, err = s3_allowed(cfg, sem, 's3:PutObject', st['creds'])
        return '200' if ok else '500 s3-' + err.split()[0]
    return '200'


def alb_request(cfg, sem, st, path, az_down=None, detail=False):
    """detail=False: clients see only the HTTP status (application error text goes to logs)."""
    tgn = cfg['alb']['listeners'].get(80)
    tg = cfg['tgs'][tgn]
    targets = [i for i in st['insts'] if cfg['asg']['tg'] == tgn]
    alb_azs = {sub_az(cfg, s) for s in cfg['alb']['subnets'] if subnet_up(s, az_down, cfg)}
    inuse = [i for i in targets if i['az'] in alb_azs]
    if not inuse:
        return '503'
    healthy = [i for i in inuse if st['th'].get(i['name']) == 'healthy']
    pool = healthy if healthy else inuse  # fail open
    outs = set()
    for i in pool:
        ok = any(conn(cfg, sem, n, i['ep'], tg['port'], az_down=az_down)[0] for n in st['alb_nodes'])
        if not ok:
            outs.add('504')
        elif not (i['app_up'] and cfg['lt']['app_port'] == tg['port']):
            outs.add('502')
        else:
            r = app_response(cfg, sem, st, i, path)
            outs.add(r if detail else r.split()[0])
    out = '|'.join(sorted(outs))
    if not healthy and st['churn']:
        out += ' intermittent-503'
    return out


# Revision 2: an internet-facing ALB publishes one address per enabled node, and clients are
# spread across nodes by DNS.  A node that the internet cannot reach (e.g. a node placed in a
# private subnet) therefore makes a share of requests time out instead of being bypassed.
ALB_NODE_ROUND_ROBIN = True


def client_to_alb(cfg, sem, st, port, az_down=None, share=False):
    ok = [conn(cfg, sem, INTERNET, n, port, az_down=az_down)[0] for n in st['alb_nodes']]
    if share:
        return (sum(ok), len(ok))
    return any(ok)


def cf_request(cfg, sem, st, path, az_down=None):
    if path.startswith('/static'):
        if cfg['bucket']['oac_grant'] or sem['resource_policy_ignored']:
            return '200'
        return '403 AccessDenied (S3 origin)'
    if cfg['cf']['origin'] != 'alb-current':
        return '502 cannot resolve origin'
    if cfg['cf']['origin_protocol'] == 'https-only':
        if not client_to_alb(cfg, sem, st, 443, az_down):
            return '504 origin timeout'
        return '502 cannot connect to origin'
    if not client_to_alb(cfg, sem, st, 80, az_down):
        return '504 origin timeout'
    out = alb_request(cfg, sem, st, path, az_down)
    k, n = client_to_alb(cfg, sem, st, 80, az_down, share=True)
    if ALB_NODE_ROUND_ROBIN and k < n:
        out += ' intermittent-504'
    return out


# ----------------------------------------------------------------------------- observations
MONITORS = ['cf_root', 'cf_items', 'cf_upload', 'cf_static', 'az_test_a', 'az_test_b',
            'alarm_unhealthy', 'ssm_available', 'logs_present']
E_ACTIONS = ['alb_get_root', 'ping_alb']
M_ACTIONS = ['target_health', 'asg_activity', 'console_a', 'console_b', 'read_app_logs', 'flow_logs']
R_ACTIONS = ['ra_internet_to_alb', 'ra_alb_to_app', 'ra_app_to_internet', 'ra_app_to_rds']
SHELL = ['curl_localhost_health', 'curl_internet', 'ping_internet', 'nc_rds_3306', 'ping_rds', 's3_put_test', 'sts_identity']
S_ACTIONS = [s + '@' + az for az in 'ab' for s in SHELL]
C_ACTIONS = ['c_rt_pub', 'c_rt_app_a', 'c_rt_app_b', 'c_igw', 'c_nat_a', 'c_nat_b', 'c_nacl_pub', 'c_nacl_app',
             'c_nacl_db', 'c_sg_alb', 'c_sg_app', 'c_sg_db', 'c_alb', 'c_tg', 'c_asg', 'c_lt', 'c_role',
             'c_bucket', 'c_rds', 'c_appcfg', 'c_cf']
ACTIONS = E_ACTIONS + M_ACTIONS + S_ACTIONS + C_ACTIONS
ALL_ACTIONS = ACTIONS + R_ACTIONS
TIER = {a: 'E' for a in E_ACTIONS}
TIER.update({a: 'M' for a in M_ACTIONS})
TIER.update({a: 'S' for a in S_ACTIONS})
TIER.update({a: 'C' for a in C_ACTIONS})
TIER.update({a: 'R' for a in R_ACTIONS})
MINUTES = {'E': 1.0, 'M': 2.0, 'S': 3.0, 'C': 2.0, 'R': 5.0}  # time-weighted cost model (sensitivity only)


def describe(cfg, a):
    d = {
        'c_rt_pub': lambda: ('rt-pub', cfg['rts']['rt-pub'], sorted(s for s, v in cfg['subnets'].items() if v[1] == 'rt-pub')),
        'c_rt_app_a': lambda: ('rt-app-a', cfg['rts']['rt-app-a'], sorted(s for s, v in cfg['subnets'].items() if v[1] == 'rt-app-a')),
        'c_rt_app_b': lambda: ('rt-app-b', cfg['rts']['rt-app-b'], sorted(s for s, v in cfg['subnets'].items() if v[1] == 'rt-app-b')),
        'c_igw': lambda: ('igw', 'attached' if cfg['igw_attached'] else 'detached'),
        'c_nat_a': lambda: ('nat-a', cfg['nats']['nat-a']['subnet'], cfg['nats']['nat-a']['eip']),
        'c_nat_b': lambda: ('nat-b', cfg['nats']['nat-b']['subnet'], cfg['nats']['nat-b']['eip']),
        'c_nacl_pub': lambda: ('nacl-pub', sorted(cfg['nacls']['nacl-pub']['inb']), sorted(cfg['nacls']['nacl-pub']['outb'])),
        'c_nacl_app': lambda: ('nacl-app', sorted(cfg['nacls']['nacl-app']['inb']), sorted(cfg['nacls']['nacl-app']['outb'])),
        'c_nacl_db': lambda: ('nacl-db', sorted(cfg['nacls']['nacl-db']['inb']), sorted(cfg['nacls']['nacl-db']['outb'])),
        'c_sg_alb': lambda: ('sg-alb', cfg['sgs']['sg-alb']['inb'], cfg['sgs']['sg-alb']['outb']),
        'c_sg_app': lambda: ('sg-app', cfg['sgs']['sg-app']['inb'], cfg['sgs']['sg-app']['outb']),
        'c_sg_db': lambda: ('sg-db', cfg['sgs']['sg-db']['inb'], cfg['sgs']['sg-db']['outb']),
        'c_alb': lambda: ('alb', cfg['alb']['subnets'], sorted(cfg['alb']['listeners'].items())),
        'c_tg': lambda: ('tg-app', sorted(cfg['tgs']['tg-app'].items())),
        'c_asg': lambda: ('asg', sorted((k, str(v)) for k, v in cfg['asg'].items())),
        'c_lt': lambda: ('lt', sorted((k, str(v)) for k, v in cfg['lt'].items())),
        'c_role': lambda: ('role', cfg['role']['trust'], sorted(cfg['role']['allow'].items())),
        'c_bucket': lambda: ('bucket', sorted(cfg['bucket'].items())),
        'c_rds': lambda: ('rds', sorted((k, str(v)) for k, v in cfg['rds'].items())),
        'c_appcfg': lambda: ('appcfg', sorted(cfg['appcfg'].items())),
        'c_cf': lambda: ('cf', sorted(cfg['cf'].items())),
    }[a]
    return repr(d())


def inst_in(st, az):
    for i in st['insts']:
        if i['az'] == az:
            return i
    return None


def observe(cfg, a, sem=TRUE_SEM, _cache=None):
    """Outcome string of action/monitor `a` on configuration `cfg` under semantics `sem`."""
    if _cache is not None and 'st' in _cache:
        st = _cache['st']
    else:
        st = derive(cfg, sem)
        if _cache is not None:
            _cache['st'] = st
    if a == 'cf_root':
        return cf_request(cfg, sem, st, '/')
    if a == 'cf_items':
        return cf_request(cfg, sem, st, '/items')
    if a == 'cf_upload':
        return cf_request(cfg, sem, st, '/upload')
    if a == 'cf_static':
        return cf_request(cfg, sem, st, '/static/logo.png')
    if a in ('az_test_a', 'az_test_b'):
        az = a[-1]
        st2 = derive(cfg, sem, az_down=az)
        return 'root=%s items=%s' % (cf_request(cfg, sem, st2, '/', az), cf_request(cfg, sem, st2, '/items', az))
    if a == 'alarm_unhealthy':
        return 'ALARM' if any(v.startswith('unhealthy') for v in st['th'].values()) else 'OK'
    if a == 'ssm_available':
        m = sorted(i['az'] for i in st['insts'] if i['ssm'])
        return 'managed:' + ','.join(m) if m else 'none'
    if a == 'logs_present':
        return 'yes' if any(i['logs'] for i in st['insts']) else 'no'
    if a == 'alb_get_root':
        if not client_to_alb(cfg, sem, st, 80):
            return 'timeout'
        out = alb_request(cfg, sem, st, '/')
        k, n = client_to_alb(cfg, sem, st, 80, share=True)
        if ALB_NODE_ROUND_ROBIN and k < n:
            out += ' intermittent-timeout'
        return out
    if a == 'ping_alb':
        if sem['icmp_as_tcp']:
            return 'reply' if client_to_alb(cfg, sem, st, 80) else 'timeout'
        return 'timeout'  # ALBs never answer ICMP echo
    if a == 'target_health':
        if cfg['asg']['tg'] is None or not st['insts']:
            return 'no registered targets'
        return ';'.join('%s:%s' % (k[-1], v) for k, v in sorted(st['th'].items()))
    if a == 'asg_activity':
        if st['launch_error']:
            return 'launch failed: ' + st['launch_error']
        if cfg['asg']['desired'] == 0:
            return 'no activity (desired=0)'
        if st['churn']:
            bad = sorted(i['az'] for i in st['insts'] if st['th'].get(i['name'], '').startswith('unhealthy'))
            return 'replacing unhealthy (ELB health check) in AZ ' + ','.join(bad)
        return 'steady (%d InService in AZ %s)' % (len(st['insts']), ','.join(sorted(i['az'] for i in st['insts'])))
    if a in ('console_a', 'console_b'):
        i = inst_in(st, a[-1])
        return i['boot'] if i else 'no instance in AZ ' + a[-1]
    if a == 'read_app_logs':
        shipping = [i for i in st['insts'] if i['logs']]
        if not shipping:
            return 'no log streams'
        errs = set()
        for i in shipping:
            if not i['app_up']:
                errs.add('app not running')
                continue
            for p in ('/items', '/upload'):
                r = app_response(cfg, sem, st, i, p)
                if r != '200':
                    errs.add(p + ':' + r)
        return ';'.join(sorted(errs)) if errs else 'no errors'
    if a == 'flow_logs':
        i = inst_in(st, 'a') or inst_in(st, 'b')
        if i is None:
            return 'no instance ENIs'
        tg = cfg['tgs'][cfg['asg']['tg'] or 'tg-app']
        p = tg['port']
        if st['alb_nodes']:
            ok, b = conn(cfg, sem, st['alb_nodes'][0], i['ep'], p)
            if ok or b == 'nacl-in:nacl-pub':
                f1 = 'in tcp/%d ACCEPT' % p
            elif b in ('sg:sg-app', 'nacl-in:nacl-app'):
                f1 = 'in tcp/%d REJECT' % p
            elif b == 'nacl-out:nacl-app':
                f1 = 'in tcp/%d ACCEPT, return REJECT' % p
            else:
                f1 = 'no inbound records'
        else:
            f1 = 'no inbound records'
        ok, b = conn(cfg, sem, i['ep'], INTERNET, 443)
        if ok:
            f2 = 'out tcp/443 ACCEPT, return ACCEPT'
        elif b in ('sg-egress:sg-app', 'nacl-out:nacl-app'):
            f2 = 'out tcp/443 REJECT'
        elif b == 'nacl-in:nacl-app':
            f2 = 'out tcp/443 ACCEPT, return REJECT'
        else:
            f2 = 'out tcp/443 ACCEPT, no return'
        return f1 + '; ' + f2
    if a.startswith('ra_'):
        i = inst_in(st, 'a') or inst_in(st, 'b')
        if a == 'ra_internet_to_alb':
            if not st['alb_nodes']:
                return 'no source'
            res = [conn(cfg, sem, INTERNET, n, 80) for n in st['alb_nodes']]
            return 'reachable' if any(r[0] for r in res) else 'blocked at ' + res[0][1]
        if i is None:
            return 'no instance ENI'
        if a == 'ra_alb_to_app':
            p = cfg['tgs'][cfg['alb']['listeners'][80]]['port']
            ok, b = conn(cfg, sem, st['alb_nodes'][0], i['ep'], p) if st['alb_nodes'] else (False, 'no source')
        elif a == 'ra_app_to_internet':
            ok, b = conn(cfg, sem, i['ep'], INTERNET, 443)
        else:
            ok, b = conn(cfg, sem, i['ep'], st['rds_ep'], cfg['rds']['port'])
        return 'reachable' if ok else 'blocked at ' + b
    if '@' in a:
        cmd, az = a.split('@')
        i = inst_in(st, az)
        if i is None:
            return 'no instance'
        if not i['ssm']:
            return 'session unavailable'
        e = i['ep']
        if cmd == 'curl_localhost_health':
            if not i['app_up']:
                return 'connection refused'
            return str(app_code(cfg, '/health')) + ' on :%d' % cfg['lt']['app_port']
        if cmd == 'curl_internet':
            return 'ok' if i['net443'] else 'timeout'
        if cmd == 'ping_internet':
            if sem['icmp_as_tcp']:
                return 'reply' if i['net443'] else 'timeout'
            return 'reply' if conn(cfg, sem, e, INTERNET, 0, proto='icmp')[0] else 'timeout'
        if cmd == 'nc_rds_3306':
            if cfg['appcfg']['db_endpoint'] != 'appdb.current':
                return 'unknown host (app config endpoint)'
            return 'open' if i['db_net'] else 'timeout'
        if cmd == 'ping_rds':
            if sem['icmp_as_tcp']:
                return 'reply' if i['db_net'] else 'timeout'
            ok = conn(cfg, sem, e, st['rds_ep'], 0, proto='icmp')[0] and st['rds_alive']
            return 'reply' if ok else 'timeout'
        if cmd == 's3_put_test':
            if not i['net443']:
                return 'timeout'
            ok, err = s3_allowed(cfg, sem, 's3:PutObject', st['creds'])
            return 'ok' if ok else err
        if cmd == 'sts_identity':
            if not st['creds']:
                return 'Unable to locate credentials'
            return 'assumed-role/role-app' if i['net443'] else 'timeout'
    if a.startswith('c_'):
        return describe(cfg, a)
    raise KeyError(a)


def signature(cfg, actions, sem=TRUE_SEM):
    cache = {}
    return tuple(observe(cfg, a, sem, cache) for a in actions)


# ----------------------------------------------------------------------------- fault operators
def _f(fid, desc, layer, category, resource, stage, fn):
    return dict(id=fid, desc=desc, layer=layer, category=category, resource=resource, stage=stage, fn=fn)


def _set(path, value):
    def fn(c):
        d = c
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    return fn


# `category` = nearest student-reported challenge theme in the anchor course (Zarnegar et al. 2026, Table 1)
# `stage`    = project stage in which the faulted component is first introduced (1-3)
FAULTS = [
    _f('N1', 'Private route table rt-app-a has no default route to the NAT gateway', 'routing', 'NAT access', 'rt-app-a', 3, _set(['rts', 'rt-app-a'], None)),
    _f('N2', 'rt-app-b routes through nat-a in the other AZ (cross-AZ dependency)', 'routing', 'NAT access', 'rt-app-b', 3, _set(['rts', 'rt-app-b'], 'nat-a')),
    _f('N3', 'NAT gateway nat-a placed in private subnet app-a', 'routing', 'NAT access', 'nat-a', 3, _set(['nats', 'nat-a', 'subnet'], 'app-a')),
    _f('N4', 'Public route table rt-pub lost its default route to the internet gateway', 'routing', 'Security vs functionality', 'rt-pub', 2, _set(['rts', 'rt-pub'], None)),
    _f('N5', 'Internet gateway detached from the VPC', 'routing', 'Security vs functionality', 'igw', 2, _set(['igw_attached'], False)),
    _f('N6', 'Internet-facing ALB placed in private subnets app-a/app-b', 'routing', 'Security vs functionality', 'alb', 3, _set(['alb', 'subnets'], ['app-a', 'app-b'])),
    _f('N7', 'ASG launches into public subnets without public IPv4 addresses', 'routing', 'NAT access', 'asg', 3, _set(['asg', 'subnets'], ['pub-a', 'pub-b'])),
    _f('S1', 'sg-app allows port 8080 instead of 80 from the ALB', 'security group', 'Security vs functionality', 'sg-app', 1, _set(['sgs', 'sg-app', 'inb'], [('tcp', 8080, 8080, 'sg-alb')])),
    _f('S2', 'sg-app ingress references sg-db instead of sg-alb', 'security group', 'Security vs functionality', 'sg-app', 3, _set(['sgs', 'sg-app', 'inb'], [('tcp', 80, 80, 'sg-db')])),
    _f('S3', 'sg-alb allows only 443, not 80, from the internet', 'security group', 'Security vs functionality', 'sg-alb', 3, _set(['sgs', 'sg-alb', 'inb'], [('tcp', 443, 443, 'any')])),
    _f('S4', 'sg-db ingress references sg-alb instead of sg-app', 'security group', 'Security vs functionality', 'sg-db', 2, _set(['sgs', 'sg-db', 'inb'], [('tcp', 3306, 3306, 'sg-alb')])),
    _f('S5', 'All egress rules removed from sg-app', 'security group', 'Security vs functionality', 'sg-app', 3, _set(['sgs', 'sg-app', 'outb'], [])),
    _f('S6', 'sg-alb egress restricted to tcp/443', 'security group', 'Security vs functionality', 'sg-alb', 3, _set(['sgs', 'sg-alb', 'outb'], [('tcp', 443, 443, 'any')])),
    _f('A1', 'nacl-app has no inbound rule for ephemeral return ports', 'network ACL', 'Security vs functionality', 'nacl-app', 3, _set(['nacls', 'nacl-app', 'inb'], [(100, 'allow', 'tcp', 80, 80, 'vpc')])),
    _f('A2', 'nacl-db inbound rule for 3306 removed', 'network ACL', 'Security vs functionality', 'nacl-db', 2, _set(['nacls', 'nacl-db', 'inb'], [])),
    _f('A3', 'nacl-pub replaced by a custom ACL allowing only inbound 80/443', 'network ACL', 'Security vs functionality', 'nacl-pub', 3,
       _set(['nacls', 'nacl-pub'], dict(inb=[(100, 'allow', 'tcp', 80, 80, 'any'), (110, 'allow', 'tcp', 443, 443, 'any')], outb=list(ALLOW_ALL)))),
    _f('A4', 'nacl-app inbound port 80 restricted to the db subnets', 'network ACL', 'Security vs functionality', 'nacl-app', 3,
       _set(['nacls', 'nacl-app', 'inb'], [(100, 'allow', 'tcp', 80, 80, 'db'), (110, 'allow', 'tcp', 1024, 65535, 'any')])),
    _f('L1', 'Target-group health-check path is /healthz (returns 404)', 'load balancing', 'Auto Scaling / health', 'tg-app', 3, _set(['tgs', 'tg-app', 'hc_path'], '/healthz')),
    _f('L2', 'Target group forwards to port 8080; app listens on 80', 'load balancing', 'Auto Scaling / health', 'tg-app', 3, _set(['tgs', 'tg-app', 'port'], 8080)),
    _f('L3', 'ASG not attached to the target group', 'load balancing', 'Auto Scaling / health', 'asg', 3, _set(['asg', 'tg'], None)),
    _f('L4', 'Listener :80 forwards to stale target group tg-old', 'load balancing', 'Auto Scaling / health', 'alb', 3, _set(['alb', 'listeners'], {80: 'tg-old'})),
    _f('L5', 'App redirects /health to HTTPS (301)', 'load balancing', 'SDK / application code', 'app', 3, _set(['lt', 'health_redirect'], True)),
    _f('L6', 'ALB node for AZ b placed in private subnet app-b (subnets pub-a/app-b)', 'load balancing', 'Security vs functionality', 'alb', 3, _set(['alb', 'subnets'], ['pub-a', 'app-b'])),
    _f('C1', 'Launch template references a deregistered AMI', 'compute', 'NAT/AMI change', 'lt', 2, _set(['lt', 'ami_exists'], False)),
    _f('C2', 'ASG desired capacity set to 0', 'compute', 'Auto Scaling / health', 'asg', 3, _set(['asg', 'desired'], 0)),
    _f('C3', 'User-data script fails before starting the web server', 'compute', 'SDK / application code', 'lt', 1, _set(['lt', 'userdata_ok'], False)),
    _f('C4', 'Web server configured to listen on 8080', 'compute', 'SDK / application code', 'lt', 1, _set(['lt', 'app_port'], 8080)),
    _f('C5', 'ASG restricted to subnet app-a', 'compute', 'Auto Scaling / health', 'asg', 3, _set(['asg', 'subnets'], ['app-a'])),
    _f('I1', 'Instance role lacks s3:PutObject', 'identity', 'IAM roles', 'role', 3, lambda c: c['role']['allow'].pop('s3:PutObject')),
    _f('I2', 'Role policy resource is the bucket ARN, not bucket/*', 'identity', 'IAM roles', 'role', 3,
       lambda c: c['role']['allow'].update({'s3:GetObject': 'arn:aws:s3:::assets-bkt', 's3:PutObject': 'arn:aws:s3:::assets-bkt'})),
    _f('I3', 'Launch template has no instance profile', 'identity', 'IAM roles', 'lt', 3, _set(['lt', 'profile'], None)),
    _f('I4', 'Role trust policy names lambda.amazonaws.com', 'identity', 'IAM roles', 'role', 3, _set(['role', 'trust'], 'lambda.amazonaws.com')),
    _f('I5', 'Bucket policy explicitly denies PutObject to the instance role', 'identity', 'S3 access', 'bucket', 3, _set(['bucket', 'deny_put_role'], True)),
    _f('I6', 'Role lacks Systems Manager permissions', 'identity', 'IAM roles', 'role', 3, lambda c: c['role']['allow'].pop('ssm:UpdateInstanceInformation')),
    _f('I7', 'Role lacks CloudWatch Logs permissions', 'identity', 'Auto Scaling / health', 'role', 3, lambda c: c['role']['allow'].pop('logs:PutLogEvents')),
    _f('D1', 'App config points to a stale database endpoint', 'data', 'SDK / application code', 'appcfg', 2, _set(['appcfg', 'db_endpoint'], 'appdb.old')),
    _f('D2', 'App config holds a rotated-out database password', 'data', 'SQL / database', 'appcfg', 2, _set(['appcfg', 'db_password'], 'old')),
    _f('D3', 'RDS instance is single-AZ', 'data', 'SQL / database', 'rds', 2, _set(['rds', 'multi_az'], False)),
    _f('E1', 'CloudFront origin points to a deleted ALB DNS name', 'edge', 'Security vs functionality', 'cf', 3, _set(['cf', 'origin'], 'alb-old')),
    _f('E2', 'CloudFront origin protocol is HTTPS-only; ALB has no 443 listener', 'edge', 'Security vs functionality', 'cf', 3, _set(['cf', 'origin_protocol'], 'https-only')),
    _f('E3', 'Bucket policy lacks the CloudFront origin-access-control grant', 'edge', 'S3 access', 'bucket', 3, _set(['bucket', 'oac_grant'], False)),
]
FAULT_BY_ID = {f['id']: f for f in FAULTS}
FIDS = [f['id'] for f in FAULTS]


def faulted(fids):
    c = base_config()
    for fid in ([fids] if isinstance(fids, str) else fids):
        FAULT_BY_ID[fid]['fn'](c)
    return c


# Request-path dependency graph (X -> Y: X depends on Y); used for symptom-cause distance.
DEPS = {
    'cf': ['cf-config', 'alb', 'bucket'],
    'alb': ['alb-config', 'sg-alb', 'nacl-pub', 'rt-pub', 'tg-app'],
    'rt-pub': ['igw'],
    'tg-app': ['tg-config', 'asg', 'sg-app', 'nacl-app', 'instance'],
    'asg': ['asg-config', 'lt'],
    'instance': ['lt', 'app', 'role', 'rt-app-a', 'rt-app-b', 'sg-app', 'nacl-app'],
    'app': ['lt', 'appcfg', 'bucket', 'role', 'rds'],
    'lt': ['role'],
    'rt-app-a': ['nat-a'], 'rt-app-b': ['nat-b'],
    'nat-a': ['rt-pub', 'nacl-pub'], 'nat-b': ['rt-pub', 'nacl-pub'],
    'rds': ['rds-config', 'sg-db', 'nacl-db'],
}
MONITOR_ENTRY = {'cf_root': 'cf', 'cf_items': 'cf', 'cf_upload': 'cf', 'cf_static': 'cf',
                 'az_test_a': 'cf', 'az_test_b': 'cf', 'alarm_unhealthy': 'tg-app',
                 'ssm_available': 'instance', 'logs_present': 'instance'}
RESOURCE_NODE = {'cf': 'cf-config', 'tg-app': 'tg-config', 'alb': 'alb-config', 'asg': 'asg-config',
                 'rds': 'rds-config'}
