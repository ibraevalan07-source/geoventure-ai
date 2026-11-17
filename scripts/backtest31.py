"""GeoVenture AI v3.1 - honest historical backtest with abstention and Next Best Action.
Usage: python3 backtest31.py <data_dir> <features_csv (built with 365-day lag)> <out_dir>

Pre-registered design (fixed before looking at money results):
  * decision date = 1 year before spud: features use neighbour results known >=365 days before spud (features.py LAG=365);
    model for wells spudded in year Y is trained only on wells completed before 1 Jan (Y-1).
  * target: new commercial discovery = P(new discovery) x P(commercial | discovery).
      P(new discovery): logistic model, Platt-recalibrated on a held-out 3-year window before the cutoff.
      P(commercial | discovery): Kaplan-Meier, 20-year horizon, using only commercial dates known before the cutoff (censoring-correct).
  * abstention: the model is trusted only if its rolling out-of-sample skill on the 5 previous years
    (new-discovery outcome, known at completion) has an 80% bootstrap lower bound > 0. Otherwise it abstains:
    decisions fall back to the base estimate and the case is flagged 'needs information'.
  * primary baseline = actual history (all these wells were drilled). Other strategies are each compared separately.
  * main scenario: $3/bbl net value, well cost $70M, planning size S = 25 mill Sm3 o.e., reserves to the first discovery of a field.
"""
import sys, os, json, io, base64, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
D, FEAT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
SNAP = pd.Timestamp('2015-07-01'); BBL = 6.29; Y0, Y1 = 1990, 2005
H_KM = 20; CAL_WIN = 3; REL_YEARS = 5; BASE_WIN = 10
MAIN = dict(v=3, C=70, S=25, attr='first')
SURVEY = dict(se=0.75, sp=0.75, cost=10)          # assumed information source for Next Best Action (scenario, not data)
rng = np.random.default_rng(31)

# ------------------------------------------------------------------ data & labels
t = pd.read_csv(FEAT, parse_dates=['wlbEntryDate', 'wlbCompletionDate'])
t = t[(t.year >= 1966) & (t.year <= 2013)].copy()
d = pd.read_csv(f'{D}/discovery.csv'); fr = pd.read_csv(f'{D}/field_reserves.csv'); dr = pd.read_csv(f'{D}/discovery_reserves.csv')
d['inc'] = pd.to_datetime(d.dscDateFromInclInField.replace('9999-12-31', np.nan), errors='coerce')
par = d.set_index('dscName'); d['fld'] = d.fldNpdidField
m = d.dscCurrentActivityStatus == 'INCLUDED IN OTHER DISCOVERY'
d.loc[m, 'fld'] = d.loc[m, 'fld'].fillna(d.loc[m, 'dscResInclInDiscoveryName'].map(par.fldNpdidField))
d['comm_date'] = pd.NaT
d.loc[d.inc.notna() & d.fld.notna(), 'comm_date'] = d.inc
d.loc[d.dscCurrentActivityStatus.isin({'PRODUCING', 'SHUT DOWN', 'PDO APPROVED'}) & d.comm_date.isna(), 'comm_date'] = SNAP
d.loc[d.dscCurrentActivityStatus.isin({'PLANNING PHASE', 'DECIDED BY THE LICENSEES'}), 'comm_date'] = SNAP
fs = fr.set_index('fldNpdidField').fldRecoverableOE
dr['oe'] = dr.dscRecoverableOil + dr.dscRecoverableGas + dr.dscRecoverableNGL * 1.9 + dr.dscRecoverableCondensate
own = d.dscNpdidDiscovery.map(dr.groupby('dscNpdidDiscovery').oe.sum())
infld = d[d.fld.notna()]
first = infld.sort_values(['dscDiscoveryYear', 'dscName']).groupby('fld').head(1).dscNpdidDiscovery
d['size_first'] = np.where(d.dscNpdidDiscovery.isin(first), d.fld.map(fs), np.where(d.fld.notna(), 0.0, own))
d['size_split'] = np.where(d.fld.notna(), d.fld.map(fs) / d.fld.map(infld.groupby('fld').size()), own)
for c in ['size_first', 'size_split']:
    d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0.0)
    assert abs(d.loc[d.fld.notna(), c].sum() - fs[fs.index.isin(d.fld.dropna())].sum()) < 1e-6
g = d.dropna(subset=['wlbNpdidWellbore']).groupby('wlbNpdidWellbore').agg(comm_date=('comm_date', 'min'), size_first=('size_first', 'sum'), size_split=('size_split', 'sum'))
t = t.merge(g, left_on='wlbNpdidWellbore', right_index=True, how='left')
t['disc'] = (t.wlbNpdidWellbore.isin(g.index) & (t.hc == 1)).astype(int)        # new discovery (known at completion)
t.loc[t.disc == 0, 'comm_date'] = pd.NaT
t[['size_first', 'size_split']] = t[['size_first', 'size_split']].where(t.disc == 1, 0.0).fillna(0.0)
t['y_final'] = t.comm_date.notna().astype(int)
assert t[t.disc == 1].wlbNpdidWellbore.duplicated().sum() == 0
t['log_dist_hc'] = np.log1p(t.dist_nearest_hc); t['local_hit_25'] = (t.n_hc_25km + 1) / (t.n_wells_25km + 2)
FE = ['lat', 'lon', 'water_depth', 'log_dist_hc', 'nearest_well_hc', 'local_hit_25', 'kernel_rate_20km', 'n_wells_25km', 'n_hc_25km']
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))

def fit(tr):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(tr[FE], tr.disc)

def p_disc_calibrated(cut, X):
    """model fitted on wells completed before cut-CAL_WIN, Platt-calibrated on wells completed in [cut-CAL_WIN, cut)."""
    c0 = cut - pd.DateOffset(years=CAL_WIN)
    m0 = fit(t[t.wlbCompletionDate < c0]); w = t[(t.wlbCompletionDate >= c0) & (t.wlbCompletionDate < cut)]
    pl = LogisticRegression(C=1e6).fit(logit(m0.predict_proba(w[FE])[:, 1]).reshape(-1, 1), w.disc)
    return pl.predict_proba(logit(m0.predict_proba(X[FE])[:, 1]).reshape(-1, 1))[:, 1], m0.predict_proba(X[FE])[:, 1]

def base_disc(cut):
    w = t[(t.wlbCompletionDate < cut) & (t.wlbCompletionDate >= cut - pd.DateOffset(years=BASE_WIN))]
    return w.disc.mean()

def km_commercial(cut, horizon=H_KM):
    """Kaplan-Meier P(commercial within horizon | discovery), events known before cut, censored at cut."""
    w = t[(t.disc == 1) & (t.wlbCompletionDate < cut)]
    ev = (w.comm_date < cut).values
    dur = np.where(ev, (w.comm_date - w.wlbCompletionDate).dt.days, (cut - w.wlbCompletionDate).dt.days) / 365.25
    dur = np.clip(np.nan_to_num(dur.astype(float)), 0, None); S = 1.0
    for tt in np.sort(np.unique(dur[ev])):
        if tt > horizon: break
        n_risk = (dur >= tt).sum(); n_ev = ((dur == tt) & ev).sum()
        if n_risk > 0: S *= 1 - n_ev / n_risk
    naive = float((w[w.wlbCompletionDate < cut - pd.DateOffset(years=10)].comm_date < cut).mean())
    return 1 - S, naive

def rolling_reliability(Y):
    """out-of-sample skill of the calibrated new-discovery model on the REL_YEARS spud years before Y-1 (outcomes known by cut)."""
    yy, pm, pb = [], [], []
    for k in range(Y - 1 - REL_YEARS, Y - 1):
        cut_k = pd.Timestamp(f'{k - 1}-01-01'); te = t[(t.year == k) & (t.wlbCompletionDate < pd.Timestamp(f'{Y - 1}-01-01'))]
        if len(te) == 0: continue
        pc, _ = p_disc_calibrated(cut_k, te)
        yy += list(te.disc); pm += list(pc); pb += [base_disc(cut_k)] * len(te)
    yy, pm, pb = map(np.array, (yy, pm, pb))
    bss = 1 - brier_score_loss(yy, pm) / brier_score_loss(yy, pb)
    bs = []
    for _ in range(500):
        i = rng.integers(0, len(yy), len(yy)); bs.append(1 - brier_score_loss(yy[i], pm[i]) / brier_score_loss(yy[i], pb[i]))
    return bss, float(np.percentile(bs, 20)), len(yy)

# ------------------------------------------------------------------ walk-forward
rows, yearinfo = [], []
for Y in range(Y0, Y1 + 1):
    te = t[t.year == Y]
    if len(te) == 0: continue
    cut = pd.Timestamp(f'{Y - 1}-01-01')
    q_km, q_naive = km_commercial(cut)
    pdc, pdu = p_disc_calibrated(cut, te); pdb = base_disc(cut)
    bss, lo, nrel = rolling_reliability(Y)
    trust = lo > 0
    o = te[['wlbNpdidWellbore', 'wlbWellboreName', 'year', 'disc', 'y_final', 'size_first', 'size_split', 'dist_nearest_hc']].copy()
    o['p_base'] = pdb * q_km; o['p_model_raw'] = pdu * q_naive; o['p_model_cal'] = pdc * q_km
    o['p_geovent'] = o.p_model_cal if trust else o.p_base
    o['abstain'] = int(not trust); o['q_km'] = q_km; o['q_naive'] = q_naive
    rows.append(o)
    yearinfo.append(dict(year=Y, n=len(te), q_km=round(q_km, 3), q_naive=round(q_naive, 3), base_disc=round(pdb, 3),
                         rolling_BSS=round(bss, 3), rolling_BSS_lo80=round(lo, 3), n_rel=nrel, trusted=bool(trust)))
P = pd.concat(rows).reset_index(drop=True); YI = pd.DataFrame(yearinfo)
y = P.y_final.values

def ece(p, yy, bins=5):
    df = pd.DataFrame({'p': p, 'y': yy, 'b': pd.qcut(p, bins, duplicates='drop')})
    gg = df.groupby('b', observed=True).agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
    return float((gg.n * (gg.p - gg.y).abs()).sum() / gg.n.sum()), gg

M = {'design': {'test_years': [Y0, Y1], 'decision_lag_days': 365, 'train_cutoff': 'Jan 1 of Y-1', 'km_horizon_years': H_KM,
                'calibration_window_years': CAL_WIN, 'reliability_years': REL_YEARS, 'base_window_years': BASE_WIN,
                'main_scenario': MAIN, 'survey_assumption': SURVEY, 'primary_baseline': 'actual history (drill all)'},
     'n_test': int(len(P)), 'n_commercial': int(y.sum()), 'commercial_rate': round(float(y.mean()), 4),
     'n_new_discoveries': int(P.disc.sum())}
cal = {}
for k, lab in [('p_base', 'База (геолог/статистика)'), ('p_model_raw', 'Модель без калибровки (как в v3.0)'),
               ('p_model_cal', 'Модель, калибровка + Каплан–Майер'), ('p_geovent', 'GeoVenture 3.1 (с отказом)')]:
    p = P[k].values; e, _ = ece(p, y)
    try: auc = roc_auc_score(y, p)
    except ValueError: auc = float('nan')
    cal[k] = dict(label=lab, AUC=round(auc, 3) if np.isfinite(auc) else None, Brier=round(brier_score_loss(y, p), 4), ECE=round(e, 4), mean_p=round(float(p.mean()), 3))
M['calibration'] = cal
M['discovery_model'] = dict(AUC_new_discovery=round(roc_auc_score(P.disc, P.p_model_cal / P.q_km), 3),
                            BSS_new_discovery_vs_base=round(1 - brier_score_loss(P.disc, P.p_model_cal / P.q_km) / brier_score_loss(P.disc, P.p_base / P.q_km), 3))
M['abstention'] = dict(years_trusted=int(YI.trusted.sum()), years_total=int(len(YI)), share_wells_abstained=round(float(P.abstain.mean()), 3))
YI.to_csv(f'{OUT}/years.csv', index=False)

# ------------------------------------------------------------------ money: pre-registered threshold strategies
def realized(attr, v, C, Q=P):
    return Q.y_final.values * Q[f'size_{attr}'].values * BBL * v - C
pre = t[t.wlbCompletionDate < pd.Timestamp(f'{Y0 - 1}-01-01')]
def geo_D(v, C, S):
    py = (pre.comm_date < pd.Timestamp(f'{Y0 - 1}-01-01')).astype(int).values
    return max([5, 10, 15, 20, 30, 40, 50, 75, 100, 1e9], key=lambda Dk: (py * S * BBL * v - C)[pre.dist_nearest_hc.values <= Dk].sum())
def decisions(v, C, S, Q=P):
    V = S * BBL * v; Dg = geo_D(v, C, S)
    return {'Факт (бурить всё)': np.ones(len(Q), bool), 'Правило по базе': Q.p_base.values * V > C,
            'Правило геолога': Q.dist_nearest_hc.values <= Dg, 'Модель без отказа': Q.p_model_cal.values * V > C,
            'GeoVenture 3.1': Q.p_geovent.values * V > C}, Dg
real = realized(MAIN['attr'], MAIN['v'], MAIN['C']); dec, Dg = decisions(MAIN['v'], MAIN['C'], MAIN['S'])
years = P.year.unique(); iy = {yy: np.where(P.year.values == yy)[0] for yy in years}
main = {}
for k, msk in dec.items():
    diffs = []
    for _ in range(3000):
        j = np.concatenate([iy[yy] for yy in rng.choice(years, len(years))])
        diffs.append(real[j][msk[j]].sum() - real[j].sum())
    diffs = np.array(diffs)
    main[k] = dict(value_musd=round(float(real[msk].sum())), drilled=round(float(msk.mean()), 3),
                   diff_vs_primary=round(float(real[msk].sum() - real.sum())),
                   diff_CI90=[round(float(np.percentile(diffs, 5))), round(float(np.percentile(diffs, 95)))],
                   P_worse_than_primary=round(float((diffs < 0).mean()), 3))
main['Идеальная информация'] = dict(value_musd=round(float(np.clip(real, 0, None).sum())))
M['main_money'] = dict(geologist_D_km=Dg, strategies=main,
                       note='diff_CI90 / P_worse: estimation uncertainty of the backtest (year-block bootstrap), not investment risk')

# scenario grid: each strategy vs primary baseline separately
G = []
for attr in ['first', 'split']:
    for v in [1, 3, 5]:
        for C in [40, 70, 120]:
            for S in [10, 25, 50]:
                r = realized(attr, v, C); dd, _ = decisions(v, C, S)
                G.append(dict(attr=attr, v=v, C=C, S=S, **{k: float(r[m_].sum()) for k, m_ in dd.items()}))
G = pd.DataFrame(G); G.to_csv(f'{OUT}/scenario_grid.csv', index=False)
strats = list(dec)
M['grid'] = {'n': int(len(G))}
for a in strats[1:]:
    M['grid'][a] = {'beats_primary': int((G[a] > G['Факт (бурить всё)']).sum()), 'beats_base_rule': int((G[a] > G['Правило по базе']).sum()),
                    'beats_geologist_rule': int((G[a] > G['Правило геолога']).sum())}

# money curve (threshold sweep) - descriptive, the pre-registered point is marked
V = MAIN['S'] * BBL * MAIN['v']; thr_pre = MAIN['C'] / V
ths = np.round(np.arange(0.0, 0.41, 0.01), 2)
curve = {k: [float(real[P[k].values > th].sum()) for th in ths] for k in ['p_base', 'p_model_cal']}
M['money_curve'] = dict(thresholds=ths.tolist(), pre_registered_threshold=round(thr_pre, 3),
                        best_threshold_model_in_hindsight=float(ths[int(np.argmax(curve['p_model_cal']))]),
                        best_threshold_base_in_hindsight=float(ths[int(np.argmax(curve['p_base']))]))

# equal-budget experiment: each year drill the top 50% by each strategy's ranking
def budget(rank_key, frac=0.5):
    tot = 0.0
    for yy, idx in iy.items():
        k = int(round(frac * len(idx)))
        if rank_key is None: tot += real[idx].mean() * k; continue   # random / base (no ranking) = expectation
        order = idx[np.argsort(rank_key[idx])[::-1]][:k]; tot += real[order].sum()
    return tot
def budget_boot(rank_key, B=2000, frac=0.5):
    # per-year contribution minus random expectation, then year-block bootstrap
    contrib = {}
    for yy, idx in iy.items():
        k = int(round(frac * len(idx))); rnd = real[idx].mean() * k
        if rank_key is None: contrib[yy] = 0.0; continue
        order = idx[np.argsort(rank_key[idx])[::-1]][:k]; contrib[yy] = real[order].sum() - rnd
    arr = np.array([sum(contrib[yy] for yy in rng.choice(years, len(years))) for _ in range(B)])
    return [round(float(np.percentile(arr, 5))), round(float(np.percentile(arr, 95)))], round(float((arr < 0).mean()), 3)
M['equal_budget_vs_random_CI90'] = {'Правило геолога (ближе к открытию)': budget_boot(-P.dist_nearest_hc.values),
                                    'Модель без отказа': budget_boot(P.p_model_cal.values)}
M['equal_budget_50pct'] = {'Случайный выбор / база (ожидание)': round(budget(None)),
                           'Правило геолога (ближе к открытию)': round(budget(-P.dist_nearest_hc.values)),
                           'Модель без отказа': round(budget(P.p_model_cal.values)),
                           'GeoVenture 3.1': round(budget(np.where(P.abstain.values == 1, rng.random(len(P)) * 0 + P.p_base.values, P.p_geovent.values)) if P.abstain.mean() < 1 else budget(None))}

# ------------------------------------------------------------------ investment risk (ex ante) vs realized
def exante(pcol, msk, nsim=5000, recent=True):
    out = np.zeros(nsim); idx = np.where(msk)[0]
    for yy in years:
        ii = idx[P.year.values[idx] == yy]
        if len(ii) == 0: continue
        cut = pd.Timestamp(f'{yy - 1}-01-01')
        known = t[(t.comm_date < cut)]                          # commercial discoveries known by cut (incl. 0 for secondary discoveries, same rule as the outcome)
        if recent: known = known[known.wlbCompletionDate >= cut - pd.DateOffset(years=BASE_WIN)]   # creaming: recent discoveries only
        pool = known.size_first.values
        succ = rng.random((nsim, len(ii))) < P[pcol].values[ii]
        sz = rng.choice(pool, (nsim, len(ii)))
        out += (succ * sz * BBL * MAIN['v'] - MAIN['C']).sum(1)
    return out
risk = {}
for k, pcol in [('Факт (бурить всё)', 'p_base'), ('Правило по базе', 'p_base'), ('GeoVenture 3.1', 'p_geovent')]:
    rv = real[dec[k]].sum(); sim_all = exante(pcol, dec[k], recent=False)
    risk[k + ' — размеры всех открытий (v3.1a, провал проверки)'] = dict(ex_ante_P_loss=round(float((sim_all < 0).mean()), 3), ex_ante_p05=round(float(np.percentile(sim_all, 5))),
                   ex_ante_median=round(float(np.median(sim_all))), ex_ante_p95=round(float(np.percentile(sim_all, 95))), realized=round(float(rv)), realized_percentile=round(float((sim_all < rv).mean()), 3))
    sim = exante(pcol, dec[k])
    risk[k] = dict(ex_ante_P_loss=round(float((sim < 0).mean()), 3), ex_ante_p05=round(float(np.percentile(sim, 5))),
                   ex_ante_median=round(float(np.median(sim))), ex_ante_p95=round(float(np.percentile(sim, 95))),
                   realized=round(float(rv)), realized_percentile=round(float((sim < rv).mean()), 3))
M['investment_risk_ex_ante'] = dict(note='Monte Carlo from the strategy own probabilities and sizes of commercial discoveries known at the decision date (main: discovered in the last 10 years - revision made AFTER the all-history version failed the honesty check); realized_percentile checks if the risk model was honest', **risk)

# ------------------------------------------------------------------ Next Best Action / VOI
se, sp, cs = SURVEY['se'], SURVEY['sp'], SURVEY['cost']
p = P.p_geovent.values; C = MAIN['C']
ev_drill = p * V - C
pp = p * se + (1 - p) * (1 - sp); p_pos = p * se / pp; p_neg = p * (1 - se) / (1 - pp)
ev_info = pp * np.maximum(0, p_pos * V - C) + (1 - pp) * np.maximum(0, p_neg * V - C) - cs
act = np.select([ev_info > np.maximum(0, ev_drill), ev_drill > 0], ['buy_info', 'drill'], 'skip')
evpi = p * (V - C) - np.maximum(0, ev_drill)       # value of perfect information per well (ex ante)
def breakeven(pi):
    for a in np.arange(0.5, 1.0001, 0.01):
        q_ = pi * a + (1 - pi) * (1 - a); pp_ = pi * a / q_; pn_ = pi * (1 - a) / (1 - q_)
        if q_ * max(0, pp_ * V - C) + (1 - q_) * max(0, pn_ * V - C) - cs > max(0, pi * V - C): return round(a, 2)
    return None
be = [breakeven(pi) for pi in p]
# simulation: survey signal drawn from the TRUE outcome with the assumed accuracy (shows mechanics, gain is driven by the assumption)
sims = []
for _ in range(500):
    sig = np.where(y == 1, rng.random(len(y)) < se, rng.random(len(y)) > sp)
    drill = np.where(act == 'drill', True, np.where(act == 'buy_info', sig & (np.where(sig, p_pos, p_neg) * V > C), False))
    sims.append((real * drill).sum() - cs * (act == 'buy_info').sum())
M['next_best_action'] = dict(survey_assumption=SURVEY, actions={a: int((act == a).sum()) for a in ['drill', 'buy_info', 'skip']},
                             mean_EVPI_per_well_musd=round(float(evpi.mean()), 1),
                             median_breakeven_accuracy=float(np.median([b for b in be if b is not None])) if any(b is not None for b in be) else None,
                             share_wells_info_never_pays=round(float(np.mean([b is None for b in be])), 3),
                             simulated_value_musd=dict(median=round(float(np.median(sims))), p05=round(float(np.percentile(sims, 5))), p95=round(float(np.percentile(sims, 95)))),
                             simulated_vs_primary_musd=round(float(np.median(sims) - real.sum())),
                             note='SIMULATION: survey signal generated from true outcomes with assumed accuracy; not evidence of real survey value')
nba_grid = []
for se_, cs_ in [(0.65, 10), (0.75, 10), (0.85, 10), (0.65, 30), (0.75, 30), (0.85, 30)]:
    pp_ = p * se_ + (1 - p) * (1 - se_); ppos = p * se_ / pp_; pneg = p * (1 - se_) / (1 - pp_)
    evi = pp_ * np.maximum(0, ppos * V - C) + (1 - pp_) * np.maximum(0, pneg * V - C) - cs_
    a_ = np.select([evi > np.maximum(0, ev_drill), ev_drill > 0], ['buy_info', 'drill'], 'skip')
    sv = []
    for _ in range(300):
        sig = np.where(y == 1, rng.random(len(y)) < se_, rng.random(len(y)) > se_)
        dr_ = np.where(a_ == 'drill', True, np.where(a_ == 'buy_info', sig & (np.where(sig, ppos, pneg) * V > C), False))
        sv.append((real * dr_).sum() - cs_ * (a_ == 'buy_info').sum())
    nba_grid.append(dict(accuracy=se_, cost_musd=cs_, buy_info=int((a_ == 'buy_info').sum()), drill=int((a_ == 'drill').sum()), skip=int((a_ == 'skip').sum()),
                         simulated_median_musd=round(float(np.median(sv))), vs_primary=round(float(np.median(sv) - real.sum()))))
M['next_best_action']['sensitivity'] = nba_grid
P['action'] = act; P['ev_drill'] = ev_drill; P['ev_info'] = ev_info; P['evpi'] = evpi; P['breakeven_acc'] = be
P.to_csv(f'{OUT}/predictions.csv', index=False)
json.dump(M, open(f'{OUT}/metrics.json', 'w'), ensure_ascii=False, indent=1)

# ------------------------------------------------------------------ figures
def fig64(f):
    b = io.BytesIO(); f.savefig(b, format='png', dpi=130, bbox_inches='tight'); plt.close(f); return base64.b64encode(b.getvalue()).decode()
C1, C2, C3, C4 = '#2a6f97', '#c8553d', '#7a8b99', '#e0a458'
f, ax = plt.subplots(figsize=(5.4, 4.6))
for k, col in [('p_model_raw', C2), ('p_model_cal', C1), ('p_base', C3)]:
    _, gg = ece(P[k].values, y); ax.plot(gg.p, gg.y, 'o-', color=col, label=f"{cal[k]['label']} (средняя p {cal[k]['mean_p']:.0%})")
ax.axhline(y.mean(), color='#bbb', lw=.8, ls='--'); ax.text(0.005, y.mean() + .005, f'факт {y.mean():.1%}', fontsize=7, color='#777')
ax.plot([0, .45], [0, .45], ':', color='#999'); ax.set_xlim(0, .45); ax.set_ylim(0, .45)
ax.set_xlabel('Прогноз P(коммерческое открытие)'); ax.set_ylabel('Факт'); ax.legend(frameon=False, fontsize=7, loc='upper left')
ax.set_title('Калибровка на независимых годах 1990–2005', fontsize=10); ax.spines[['top', 'right']].set_visible(False)
img_cal = fig64(f)

f, ax = plt.subplots(figsize=(7.2, 3.6))
ax.bar(YI.year, YI.rolling_BSS, color=np.where(YI.trusted, C1, C3)); ax.scatter(YI.year, YI.rolling_BSS_lo80, color=C2, s=12, zorder=3, label='нижняя граница 80%')
ax.axhline(0, color='k', lw=.8); ax.set_ylabel('Skill модели за 5 прошлых лет'); ax.legend(frameon=False, fontsize=8)
ax.set_title('Самопроверка перед каждым годом: синий — модели доверяем, серый — отказ', fontsize=10); ax.spines[['top', 'right']].set_visible(False)
img_rel = fig64(f)

f, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot(ths, np.array(curve['p_model_cal']) / 1000, color=C1, label='Модель (калибр.)'); ax.plot(ths, np.array(curve['p_base']) / 1000, color=C3, label='База')
ax.axhline(real.sum() / 1000, color=C4, ls='--', lw=1, label='Факт (бурить всё)'); ax.axvline(thr_pre, color='k', ls=':', lw=1)
ax.text(thr_pre + .005, ax.get_ylim()[0] + .5, f'заранее заданный порог {thr_pre:.1%}', fontsize=8)
ax.set_xlabel('Порог бурения: бурить, если P > порога'); ax.set_ylabel('Результат, млрд $'); ax.legend(frameon=False, fontsize=8)
ax.set_title('Money curve: \\$3/барр., скважина \\$70 млн, S = 25 млн м³', fontsize=10); ax.spines[['top', 'right']].set_visible(False)
img_curve = fig64(f)

f, ax = plt.subplots(figsize=(7.2, 3.8))
for k, col in [('Факт (бурить всё)', C3), ('GeoVenture 3.1', C1)]:
    pcol = 'p_base' if k != 'GeoVenture 3.1' else 'p_geovent'
    sim = exante(pcol, dec[k], 3000, recent=True); ax.hist(sim / 1000, bins=60, alpha=.5, color=col, label=f'{k}: ожидаемое распределение')
    ax.axvline(risk[k]['realized'] / 1000, color=col, lw=2)
ax.axvline(0, color='k', lw=.6); ax.set_xlabel('Результат портфеля 1990–2005, млрд $ (линии — фактический итог)')
ax.legend(frameon=False, fontsize=8); ax.set_title('Инвестиционный риск до бурения (Монте-Карло) и что случилось на самом деле', fontsize=10); ax.spines[['top', 'right']].set_visible(False)
img_risk = fig64(f)

# ------------------------------------------------------------------ HTML
def fmt(x): return f'{x:,.0f}'.replace(',', ' ')
mm = M['main_money']['strategies']
rows_money = ''.join(f"<tr><td>{k}</td><td>{fmt(v['value_musd'])}</td><td>{v.get('drilled', 0):.0%}</td><td>{fmt(v['diff_vs_primary']) if 'diff_vs_primary' in v else '—'}</td><td>{(fmt(v['diff_CI90'][0]) + ' … ' + fmt(v['diff_CI90'][1])) if 'diff_CI90' in v else '—'}</td><td>{v.get('P_worse_than_primary', '—')}</td></tr>" for k, v in mm.items())
rows_cal = ''.join(f"<tr><td>{v['label']}</td><td>{v['AUC']}</td><td>{v['Brier']}</td><td>{v['ECE']}</td><td>{v['mean_p']:.1%}</td></tr>" for v in cal.values())
rows_grid = ''.join(f"<tr><td>{a}</td><td>{M['grid'][a]['beats_primary']}</td><td>{M['grid'][a]['beats_base_rule']}</td><td>{M['grid'][a]['beats_geologist_rule']}</td></tr>" for a in strats[1:])
rows_risk = ''.join(f"<tr><td>{k}</td><td>{v['ex_ante_P_loss']:.0%}</td><td>{fmt(v['ex_ante_p05'])} … {fmt(v['ex_ante_p95'])}</td><td>{fmt(v['realized'])}</td><td>{v['realized_percentile']:.0%}</td></tr>" for k, v in risk.items())
rows_bud = ''.join(f"<tr><td>{k}</td><td>{fmt(v)}</td><td>{(fmt(M['equal_budget_vs_random_CI90'][k][0][0]) + ' … ' + fmt(M['equal_budget_vs_random_CI90'][k][0][1]) + ' (P хуже случайного ' + str(M['equal_budget_vs_random_CI90'][k][1]) + ')') if k in M['equal_budget_vs_random_CI90'] else '—'}</td></tr>" for k, v in M['equal_budget_50pct'].items())
rows_nba = ''.join(f"<tr><td>{r['accuracy']:.0%}</td><td>{r['cost_musd']}</td><td>{r['buy_info']} / {r['drill']} / {r['skip']}</td><td>{fmt(r['simulated_median_musd'])}</td><td>{fmt(r['vs_primary'])}</td></tr>" for r in nba_grid)
nba = M['next_best_action']
html = f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><title>GeoVenture AI 3.1 — отчёт бэктеста</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;margin:32px auto;padding:0 16px;color:#1d2731;line-height:1.5}}
h1{{font-size:24px}}h2{{font-size:18px;margin-top:30px;border-top:1px solid #e3e6e8;padding-top:14px}}table{{border-collapse:collapse;margin:8px 0;font-size:14px}}
td,th{{border:1px solid #d9dee2;padding:5px 9px;text-align:left}}th{{background:#f3f5f6}}.note{{color:#5b6670;font-size:13px}}img{{max-width:100%}}.warn{{background:#fff6e5;padding:8px 12px;border-left:3px solid #e0a458}}</style></head><body>
<h1>GeoVenture AI 3.1 — воспроизводимый исторический бэктест</h1>
<p class="note">Сгенерировано <code>backtest31.py</code>. Данные: снимок NPD/Sodir 07.2015. Тест: {M['n_test']} разведочных скважин 1990–2005, новых коммерческих открытий {M['n_commercial']} ({M['commercial_rate']:.1%}). Дата решения — за год до бурения; обучение для года Y — на скважинах, завершённых до 1 января Y−1. Дизайн и основной сценарий зафиксированы заранее (шапка скрипта).</p>
<h2>1. Калибровка вероятностей</h2>
<table><tr><th>Модель</th><th>AUC</th><th>Brier ↓</th><th>ECE ↓</th><th>Средняя p (факт {M['commercial_rate']:.1%})</th></tr>{rows_cal}</table>
<p>P(коммерческое | открытие) теперь считается методом Каплана–Майера (горизонт {H_KM} лет), который учитывает, что статус свежих открытий ещё не известен. Модель нового открытия: AUC {M['discovery_model']['AUC_new_discovery']}, skill к базе {M['discovery_model']['BSS_new_discovery_vs_base']}.</p>
<img src="data:image/png;base64,{img_cal}">
<h2>2. Режим отказа (AI Abstention)</h2>
<p>Перед каждым годом модель проверяет себя на 5 прошлых годах (исход «новое открытие» известен сразу после бурения). Доверие — только если нижняя граница 80% интервала skill больше нуля. Модели доверяли в {M['abstention']['years_trusted']} из {M['abstention']['years_total']} лет; в остальных GeoVenture честно говорит «мой прогноз не лучше базы» и переходит на базовую оценку и Next Best Action.</p>
<img src="data:image/png;base64,{img_rel}">
<h2>3. Деньги: заранее заданный сценарий, каждая стратегия против факта</h2>
<table><tr><th>Стратегия</th><th>Результат, млн $</th><th>Бурит</th><th>Разница с фактом</th><th>90% интервал разницы*</th><th>P(хуже факта)*</th></tr>{rows_money}</table>
<p class="note">* Бутстреп по годам — это неопределённость самого бэктеста (насколько оценка зависит от состава лет), а не инвестиционный риск. Правило геолога: D = {Dg:g} км, подобрано на скважинах до {Y0 - 1} г.</p>
<p>Сетка {M['grid']['n']} сценариев (3 ценности барреля × 3 стоимости скважины × 3 размера S × 2 учёта запасов), число сценариев, где стратегия лучше:</p>
<table><tr><th>Стратегия</th><th>факта</th><th>правила по базе</th><th>правила геолога</th></tr>{rows_grid}</table>
<h2>4. Money curve</h2><p>Результат при разных порогах. Лучший порог задним числом: модель {M['money_curve']['best_threshold_model_in_hindsight']:.0%}, база {M['money_curve']['best_threshold_base_in_hindsight']:.0%} — выбирать его по тесту нельзя, это подгонка.</p>
<img src="data:image/png;base64,{img_curve}">
<h2>5. Одинаковый бюджет: каждый год бурим половину скважин</h2>
<table><tr><th>Как выбираем</th><th>Результат, млн $</th><th>90% интервал разницы со случайным выбором*</th></tr>{rows_bud}</table>
<h2>6. Инвестиционный риск до бурения против факта</h2>
<table><tr><th>Стратегия</th><th>P(убыток) до бурения</th><th>90% диапазон до бурения, млн $</th><th>Факт, млн $</th><th>Перцентиль факта</th></tr>{rows_risk}</table>
<p class="note">Монте-Карло из собственных вероятностей стратегии и размеров коммерческих открытий, известных на дату решения. Перцентиль факта около 50% — модель риска честна; у краёв — риск недооценён или переоценён. Первая версия (размеры всех исторических открытий) провалила проверку: факт ниже всех симуляций, потому что гиганты 1970-х завышают ожидания. Основная версия: (1) исправлена ошибка — в пул размеров не попадали коммерческие открытия с нулевыми запасами по правилу «запасы первому открытию», хотя исход измеряется по тому же правилу; (2) берутся открытия последних 10 лет. Обе правки сделаны после того, как мы увидели провал, и так и помечены.</p>
<img src="data:image/png;base64,{img_risk}">
<h2>7. Next Best Action и ценность информации</h2>
<p>Для каждой скважины: бурить / не бурить / сначала купить информацию. Информация — условное исследование с точностью {SURVEY['se']:.0%} и стоимостью ${SURVEY['cost']} млн (допущение). Рекомендации: бурить {nba['actions']['drill']}, купить информацию {nba['actions']['buy_info']}, не бурить {nba['actions']['skip']}. Средняя ценность идеальной информации — ${nba['mean_EVPI_per_well_musd']} млн на скважину; исследование окупается при точности от {nba['median_breakeven_accuracy']:.0%} (медиана), для {nba['share_wells_info_never_pays']:.0%} скважин не окупается ни при какой точности.</p>
<table><tr><th>Точность исследования</th><th>Цена, млн $</th><th>Купить инфо / бурить / нет</th><th>Симуляция, млн $</th><th>К факту</th></tr>{rows_nba}</table>
<p class="warn">Симуляция: сигнал исследования сгенерирован из реальных исходов с заданной точностью. Медианный результат {fmt(nba['simulated_value_musd']['median'])} млн $ ({fmt(nba['simulated_vs_primary_musd'])} к факту) показывает механику, а не доказанную ценность реальной сейсмики.</p>
<h2>8. Ограничения</h2><ul><li>Снимок 2015 г.; скважины 2014–2026 не проверены.</li><li>Запасы — оценки 2015 г., только для измерения результата; распределение запасов месторождения на первое открытие — допущение (вариант «поровну» в сетке).</li>
<li>Ценность барреля, стоимость скважины и точность исследования — сценарные параметры.</li><li>Бэктест видит только пробуренные скважины; ценность открытия нового района не учтена.</li></ul></body></html>"""
open(f'{OUT}/report.html', 'w').write(html)
print(json.dumps(M, ensure_ascii=False, indent=1)); print(YI.to_string(index=False))
