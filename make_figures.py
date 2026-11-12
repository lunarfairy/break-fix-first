"""make_figures.py (revision 2) - regenerates Figures 2-6 from results/.
Run after run_analyses.py, robustness.py, the LLM benchmark (or replay_llm.py) and trial_sim.py.
Writes figs/*.pdf and figs/*.png and the plotted values as figs/*_data.csv."""
import json, os
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import diagnosis as dg

META_GREY = '#8a8a8a'
BLUE, LBLUE, ORANGE, DARK = '#1f5fa8', '#7fa7d6', '#c05a00', '#0b2d59'
mpl.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
                     'font.family': 'DejaVu Sans'})
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results') + os.sep
FIG = os.path.join(HERE, 'figs') + os.sep
os.makedirs(FIG, exist_ok=True)


def panel_letter(ax, l):
    ax.text(-0.16, 1.06, l, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')


def save(fig, name):
    fig.savefig(FIG + name + '.pdf', bbox_inches='tight')
    fig.savefig(FIG + name + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


FB = pd.read_csv(R + 'fault_bank.csv'); TIER = pd.read_csv(R + 'observability_tiers.csv')
SUM = json.load(open(R + 'summary.json')); RS = json.load(open(R + 'robust_summary.json'))
TRI = pd.read_csv(R + 'robust_conditioned_trace.csv', index_col=0)
SD = pd.read_csv(R + 'strategy_costs_universe.csv').set_index('mid')


def fig2():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1, 1.25, 1.1], wspace=0.5))
    ax = axs[0]
    y = TIER.universe_mean_class_size_per_fault.values[:5]; xs = np.arange(5)
    ax.vlines(xs, 0, y, color=META_GREY, lw=1.2); ax.plot(xs, y, 'o', color=BLUE, ms=5, zorder=3)
    yra = TIER.universe_mean_class_size_per_fault.values[5]
    ax.plot([2.3], [yra], 'D', mfc='white', mec=ORANGE, ms=4.5, zorder=4)
    ax.text(2.55, yra - 0.6, 'T2 + path\nanalysis', fontsize=6, color=ORANGE, va='center')
    for x_, v in zip(xs, y):
        ax.text(x_, v + 0.6, '%.1f' % v, fontsize=6, ha='center')
    ax.set_xticks(xs); ax.set_xticklabels(['T0', 'T1', 'T2', 'T3', 'T4'])
    ax.set_xlabel('Cumulative evidence tier'); ax.set_ylabel('Mutants consistent with\nthe evidence (mean)')
    ax.set_ylim(0, 17); ax.set_xlim(-0.5, 4.5); ax.set_title('Only inspection isolates', loc='left')
    ax = axs[1]
    cur = SD.loc[[m for m in SD.index if isinstance(SD.loc[m, 'fault'], str)]]
    cur = cur.join(TRI[['triage']])
    order = [('optimal', 'Optimal'), ('greedy', 'Greedy'), ('triage', 'Triage\nfirst'), ('sweep', 'Console\nsweep'), ('trace', 'Outside-in\ntrace'), ('random', 'Random')]
    cols = [BLUE, LBLUE, META_GREY, META_GREY, ORANGE, META_GREY]
    rng_ = np.random.default_rng(7); rows = []
    for k, ((c, l), col) in enumerate(zip(order, cols)):
        v = cur[c].values; jx = k + rng_.uniform(-0.18, 0.18, len(v))
        ax.scatter(jx, v, s=7, color=col, alpha=0.75, lw=0, zorder=2)
        ax.hlines(np.median(v), k - 0.3, k + 0.3, color='black', lw=1.4, zorder=3)
        rows += [dict(strategy=c, fault=f, actions=a) for f, a in zip(cur.fault, v)]
    pd.DataFrame(rows).to_csv(FIG + 'fig2b_data.csv', index=False)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([l.replace('\n', ' ') for _, l in order], fontsize=6, rotation=40, ha='right')
    ax.set_ylabel('Actions to isolate the fault'); ax.set_ylim(-1.2, 35.5)
    ax.set_title('Strategy sets the cost', loc='left')
    ax = axs[2]; d = FB.distance.values
    for c, col, mk in [('trace', ORANGE, 'o'), ('optimal', BLUE, 's')]:
        jx = d + rng_.uniform(-0.15, 0.15, len(d)); ax.scatter(jx, FB[c], s=9, color=col, alpha=0.8, lw=0, marker=mk)
    ax.text(0.7, 34, 'Outside-in trace  $\\rho$ = %.2f' % SUM['distance_spearman']['trace']['rho'], color=ORANGE, fontsize=6)
    ax.text(0.7, 31.8, 'Optimal  $\\rho$ = %.2f' % SUM['distance_spearman']['optimal']['rho'], color=BLUE, fontsize=6)
    ro = RS['random_orders']
    ax.text(0.7, 30.6, 'Random orders  $\\rho$ = %.2f\n(95%%: %.2f to %.2f)' % (ro['mean_rho'], ro['q025'], ro['q975']),
            color=META_GREY, fontsize=6, va='top')
    ax.set_xlabel('Symptom-to-cause distance (hops)'); ax.set_ylabel('Actions to isolate the fault')
    ax.set_xticks(range(1, 7)); ax.set_ylim(-1.2, 35.5); ax.set_xlim(0.5, 6.5)
    ax.set_title('Tracing pays for depth', loc='left')
    for a_, l_ in zip(axs, 'abc'):
        panel_letter(a_, l_)
    save(fig, 'fig2_diagnosability')


def fig3():
    mg = pd.read_csv(R + 'matrix_curated_greedy.csv', index_col=0); mt = pd.read_csv(R + 'matrix_curated_trace.csv', index_col=0)
    MCS = list(mg.columns)
    cat = {'.': 0, '-': 1, '+': 1, 'W': 2, 'I': 3, 'WI': 4, 'B': 4}
    letter = {'.': '', '-': '', '+': '', 'W': 'w', 'I': 'I', 'WI': 'WI', 'B': 'B'}
    cmap = ListedColormap(['#f2f2f2', '#cfd8e3', '#e8b04b', BLUE, ORANGE])
    labels_full = ['no effect', 'cost changed', 'wrong repair, then solved', 'impasse (I)', 'wrong repair(s), then impasse (WI)']
    mlab = {m: dg.MISCONCEPTIONS[m]['label'] for m in MCS}
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 4.6), gridspec_kw=dict(hspace=0.62))
    for ax, (mat, ttl) in zip(axs, [(mg, 'Hypothesis-driven learner (information-greedy action choice)'),
                                    (mt, 'Procedural learner (fixed outside-in path tracing)')]):
        A = mat.T.apply(lambda c: c.map(cat)).values.astype(float)
        ax.imshow(A, cmap=cmap, vmin=-0.5, vmax=4.5, aspect='auto', interpolation='nearest')
        for i, mc in enumerate(MCS):
            for j, f in enumerate(mat.index):
                s = letter[mat.loc[f, mc]]
                if s:
                    ax.text(j, i, s, ha='center', va='center', fontsize=4.5, color='white' if s in ('I', 'WI') else 'black')
        ax.set_yticks(range(len(MCS))); ax.set_yticklabels(['%s  %s' % (m, mlab[m]) for m in MCS], fontsize=6)
        ax.set_xticks(range(mat.shape[0])); ax.set_xticklabels(mat.index, fontsize=6, rotation=90)
        ax.set_xticks(np.arange(-0.5, mat.shape[0], 1), minor=True); ax.set_yticks(np.arange(-0.5, len(MCS), 1), minor=True)
        ax.grid(which='minor', color='white', lw=0.6); ax.tick_params(which='minor', length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(ttl, loc='left')
    present = sorted(set(np.unique(np.concatenate([mg.T.apply(lambda c: c.map(cat)).values.ravel(), mt.T.apply(lambda c: c.map(cat)).values.ravel()]).astype(int))))
    axs[1].legend(handles=[mpatches.Patch(color=cmap(i), label=labels_full[i]) for i in present], ncol=4, fontsize=6,
                  frameon=False, loc='upper left', bbox_to_anchor=(-0.25, -0.3), handlelength=1.2)
    panel_letter(axs[0], 'a'); panel_letter(axs[1], 'b')
    save(fig, 'fig3_misconceptions')


def fig4():
    SE = pd.read_csv(R + 'robust_learner_sensitivity.csv'); NS = pd.DataFrame(RS['noise_selectivity'])
    TF = pd.read_csv(R + 'robust_two_fault.csv')
    MCS = ['MC%d' % i for i in range(2, 9)]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.35, 1, 0.9], wspace=0.62))
    ax = axs[0]; y = np.arange(len(MCS))[::-1]
    v = SE[SE.condition == 'baseline (verify, masks)'].set_index('mc'); nv = SE[SE.condition == 'no repair verification'].set_index('mc')
    h = 0.36
    for i, mc in enumerate(MCS):
        ax.barh(y[i] + h / 2, v.loc[mc, 'impasse'], height=h, color=BLUE)
        ax.barh(y[i] - h / 2, nv.loc[mc, 'impasse'], height=h, color=BLUE, alpha=0.45)
        ax.barh(y[i] - h / 2, nv.loc[mc, 'wrong_stable'], left=nv.loc[mc, 'impasse'], height=h, color=ORANGE)
    ax.set_yticks(y); ax.set_yticklabels(MCS); ax.set_xlabel('Affected episodes (of 184 mutants)')
    ax.legend(handles=[mpatches.Patch(color=BLUE, label='verified: impasse'), mpatches.Patch(color=BLUE, alpha=0.45, label='not verified: impasse'),
                       mpatches.Patch(color=ORANGE, label='not verified:\nstable wrong diagnosis')], fontsize=6, frameon=False, loc='center right', bbox_to_anchor=(1.02, 0.32))
    ax.set_title('Verification exposes errors', loc='left')
    ax = axs[1]
    eps = [0.0, 0.05, 0.10]
    aff = [1.0] + [NS[NS.eps == e].impasse_rate_affected.mean() for e in eps[1:]]
    una = [0.0] + [NS[NS.eps == e].impasse_rate_unaffected.mean() for e in eps[1:]]
    ax.plot(eps, np.array(aff) * 100, '-o', color=BLUE, ms=4); ax.plot(eps, np.array(una) * 100, '-o', color=META_GREY, ms=4)
    ax.text(0.102, aff[-1] * 100 - 7, 'faults the\nmisconception\ndisables', fontsize=6, color=BLUE, ha='right', va='top')
    ax.text(0.102, una[-1] * 100 + 5, 'other faults', fontsize=6, color=META_GREY, ha='right', va='bottom')
    ax.set_xlabel('Probability of misreading an observation'); ax.set_ylabel('Episodes ending in impasse (%)')
    ax.set_xticks(eps); ax.set_ylim(-4, 104); ax.set_xlim(-0.008, 0.108)
    ax.set_title('Impasses stay selective', loc='left')
    pd.DataFrame(dict(eps=eps, affected=aff, unaffected=una)).to_csv(FIG + 'fig4b_data.csv', index=False)
    ax = axs[2]
    m1 = TF[TF.masked_ticket]; m0 = TF[~TF.masked_ticket]
    for k, (d, lab) in enumerate([(m1, 'ticket equals\none fault'), (m0, 'new\nticket')]):
        res = (d.outcome == 'resolved').sum(); imp = d.outcome.str.startswith('impasse').sum()
        ax.bar(k, res, color=BLUE, width=0.6); ax.bar(k, imp, bottom=res, color=ORANGE, width=0.6)
        ax.text(k, res + imp + 8, '%d' % (res + imp), ha='center', fontsize=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['ticket of\none fault', 'new\nticket'], fontsize=6)
    ax.set_ylabel('Pairs of curated faults'); ax.set_ylim(0, 640); ax.set_xlim(-0.6, 1.6)
    ax.legend(handles=[mpatches.Patch(color=BLUE, label='both repaired in turn'), mpatches.Patch(color=ORANGE, label='impasse')],
              fontsize=6, frameon=False, loc='upper right')
    ax.set_title('Two-fault incidents', loc='left')
    for a_, l_ in zip(axs, 'abc'):
        panel_letter(a_, l_)
    save(fig, 'fig4_sensitivity')


def fig5():
    LE = pd.read_csv(R + 'llm_episodes.csv'); LS = json.load(open(R + 'llm_stats.json'))
    fb = FB.set_index('fault'); order_f = fb.sort_values(['optimal', 'trace']).index.tolist()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[3.2, 1], wspace=0.32))
    ax = axs[0]; xs = np.arange(len(order_f))
    ax.scatter(xs, fb.loc[order_f, 'optimal'], marker='_', color='black', s=30, lw=1.4, zorder=4, label='Optimal policy')
    for ag, col, off, nm in [('haiku', LBLUE, -0.18, 'Haiku-class'), ('sonnet', BLUE, 0.18, 'Sonnet-class')]:
        d = LE[LE.agent == ag]
        for k, (rep, dd) in enumerate(d.groupby('run')):
            dd = dd.set_index('fault').loc[order_f]; ok = dd.solved.values.astype(bool)
            ax.scatter(xs[ok] + off, dd.n_actions_until_end.values[ok], s=10, color=col, lw=0, zorder=3, label=('%s, solved' % nm) if k == 0 else None)
            ax.scatter(xs[~ok] + off, dd.n_actions_until_end.values[~ok], s=14, facecolor='white', edgecolor=col, lw=0.9, zorder=3,
                       label=('%s, not solved' % nm) if k == 0 else None)
    ax.set_xticks(xs); ax.set_xticklabels(order_f, rotation=90, fontsize=6)
    ax.set_ylabel('Actions taken'); ax.set_ylim(-0.8, 20.5); ax.set_xlim(-0.8, len(xs) - 0.2); ax.set_yticks(range(0, 17, 4))
    ax.legend(fontsize=6, frameon=False, loc='upper left', ncol=3, bbox_to_anchor=(0, 1.02), columnspacing=0.8, handletextpad=0.3)
    ax.set_title('Agents solve most faults but take extra actions', loc='left', pad=12)
    ax = axs[1]; cols = [LBLUE, BLUE]
    for k, ag in enumerate(['haiku', 'sonnet']):
        a = LS['agents'][ag]; acc = a['solved_rate']
        lo, hi = a['fault_cluster_bootstrap95']; wl, wh = a['wilson95']
        ax.vlines(k - 0.08, wl * 100, wh * 100, color=META_GREY, lw=1.2)
        ax.vlines(k + 0.08, lo * 100, hi * 100, color=cols[k], lw=1.6)
        ax.plot(k, acc * 100, 'o', color=cols[k], ms=5)
        ax.text(k + 0.18, acc * 100, '%.0f%%' % (acc * 100), fontsize=6, va='center')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Haiku-\nclass', 'Sonnet-\nclass']); ax.set_xlim(-0.5, 1.7)
    ax.set_ylabel('Episodes solved (%)'); ax.set_ylim(70, 101)
    ax.text(-0.45, 72, 'grey: Wilson 95% CI\ncolour: fault-cluster\nbootstrap 95% CI', fontsize=6, color=META_GREY, va='bottom')
    ax.set_title('Accuracy (82 episodes each)', loc='left', pad=12)
    panel_letter(axs[0], 'a'); panel_letter(axs[1], 'b')
    save(fig, 'fig5_llm')


def fig6():
    TP = pd.read_csv(R + 'trial_power.csv'); TR = pd.read_csv(R + 'trial_reliability.csv'); H = pd.read_csv(R + 'trial_headline.csv')
    CL = pd.read_csv(R + 'trial_clusters.csv'); RB = pd.read_csv(R + 'trial_robust.csv')
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), gridspec_kw=dict(wspace=0.38, hspace=0.6))
    cols = {0.05: '#9cc0e6', 0.10: BLUE, 0.20: DARK}
    ax = axs[0, 0]
    for rho in [0.05, 0.20]:
        d = TP[(TP.rho == rho) & (TP.r2 == 0.3) & (TP.K == 8)].sort_values('delta')
        ax.plot(d.delta, d.power, '-o', color=cols[rho], ms=3, lw=1.0)
        ax.text(0.54, d.power.values[-1] - (0.12 if rho == 0.20 else -0.035), 'ICC %.2f' % rho, fontsize=6, color=cols[rho], va='center')
    d = H.sort_values('delta'); ax.plot(d.delta, d.power, '-o', color=cols[0.10], ms=4, lw=1.6)
    ax.text(0.54, d.power.values[-1] - 0.03, 'ICC 0.10', fontsize=6, color=cols[0.10], va='center')
    ax.axhline(0.8, color=META_GREY, lw=0.7, ls='--'); ax.axhline(0.05, color=META_GREY, lw=0.7, ls=':')
    lam = float(H[H.delta == 0.4].d_observed.iloc[0]) / 0.4
    ax.axvspan(0.20 / lam, 0.51 / lam, color='#e8b04b', alpha=0.18, lw=0)
    ax.text(0.20 / lam + 0.005, 0.12, 'meta-analytic 95% CI,\nlatent scale', fontsize=6, color='#8a6d1a', va='bottom')
    ax.set_xlabel('True effect $\\delta$ (latent SD)'); ax.set_ylabel('Power (two-sided $\\alpha$ = .05)')
    ax.set_xlim(-0.02, 0.74); ax.set_ylim(0, 1.03); ax.set_title('28 groups detect about 0.4 SD', loc='left')
    ax = axs[0, 1]
    for dl, col in [(0.2, '#9cc0e6'), (0.3, BLUE), (0.4, DARK)]:
        d = CL[CL.delta == dl].sort_values('J'); ax.plot(d.J, d.power, '-o', color=col, ms=3.5, lw=1.2)
        ax.text(57.5, d.power.values[-1], '$\\delta$ = %.1f' % dl, fontsize=6, color=col, va='center')
    ax.axhline(0.8, color=META_GREY, lw=0.7, ls='--'); ax.axvline(28, color=META_GREY, lw=0.7, ls=':')
    ax.set_xticks([20, 28, 40, 56]); ax.set_xlabel('Tutorial groups randomised'); ax.set_ylabel('Power')
    ax.set_xlim(16, 64); ax.set_ylim(0, 1.03); ax.set_title('More groups buy power', loc='left')
    ax = axs[1, 0]
    rb = RB[RB.delta == 0.4].copy()
    base = float(H[H.delta == 0.4].power.iloc[0])
    rb = rb.sort_values('power'); yy = np.arange(len(rb))
    ax.hlines(yy, base, rb.power, color=META_GREY, lw=0.8)
    ax.plot(rb.power, yy, 'o', color=BLUE, ms=4)
    ax.axvline(base, color='black', lw=0.8); ax.text(base + 0.005, -0.6, 'pre-specified scenario', fontsize=6, va='center')
    ax.set_yticks(yy); ax.set_yticklabels(rb.label.values, fontsize=6); ax.set_ylim(-1.2, len(rb) - 0.4); ax.set_xlabel('Power at $\\delta$ = 0.4'); ax.set_xlim(0.6, 1.0)
    ax.set_title('Sensitivity of power', loc='left')
    ax = axs[1, 1]
    ac = {0.8: META_GREY, 1.2: LBLUE, 1.6: BLUE}
    for a in [0.8, 1.2, 1.6]:
        d = TR[TR.a == a].sort_values('K'); ax.plot(d.K, d.rel, '-o', color=ac[a], ms=3.5, lw=1.2)
        ax.text(16.6, d.rel.values[-1] + (0.015 if a == 1.6 else (-0.015 if a == 0.8 else 0)), '$a$ = %.1f' % a, fontsize=6, color=ac[a], va='center')
    ax.axhline(0.7, color=META_GREY, lw=0.7, ls='--')
    ax.set_xlabel('Items in the practical (K)'); ax.set_ylabel('Squared correlation of score\nwith latent skill'); ax.set_ylim(0, 1); ax.set_xlim(1, 19.8); ax.set_xticks([2, 4, 8, 12, 16])
    ax.set_title('Short practicals are noisy', loc='left')
    for a_, l_ in zip(axs.ravel(), 'abcd'):
        panel_letter(a_, l_)
    save(fig, 'fig6_trial')


if __name__ == '__main__':
    fig2(); fig3(); fig4(); fig5(); fig6()
    print('figures written to figs/')
