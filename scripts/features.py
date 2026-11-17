"""Build leakage-safe pre-drill features for NCS wildcat wells (decision date = spud/entry date)."""
import numpy as np, pandas as pd, sys

D = sys.argv[1] if len(sys.argv) > 1 else 'data'
LAG_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # delay before a neighbour's result is "known"
OUT = sys.argv[3] if len(sys.argv) > 3 else f'{D}/wildcats_features.csv'

w = pd.read_csv(f'{D}/wellbore_exploration_all.csv', low_memory=False)
disc = pd.read_csv(f'{D}/discovery.csv')
lic = pd.read_csv(f'{D}/licence.csv')

for c in ['wlbEntryDate', 'wlbCompletionDate']:
    w[c] = pd.to_datetime(w[c], errors='coerce')
HC = {'OIL', 'GAS', 'OIL/GAS', 'GAS/CONDENSATE', 'OIL/GAS/CONDENSATE'}
w['hc'] = w.wlbContent.isin(HC).astype(int)
w['main_bore'] = w.wlbWellboreName == w.wlbWell

# ---- all exploration main bores = the "known history" pool for neighbour features
pool = w[w.main_bore & w.wlbEntryDate.notna() & w.wlbCompletionDate.notna()].copy()
pool['known_date'] = pool.wlbCompletionDate + pd.Timedelta(days=LAG_DAYS)

# ---- target set: main-bore wells PLANNED as wildcat (planned purpose is pre-drill; final purpose can be reclassified)
t = pool[(pool.wlbPurposePlanned == 'WILDCAT')].copy()

# ---- commercial label: linked discovery that is producing / approved / in planning, or included in a field
dev = {'PRODUCING', 'SHUT DOWN', 'PDO APPROVED', 'PLANNING PHASE', 'DECIDED BY THE LICENSEES'}
disc['commercial'] = disc.dscCurrentActivityStatus.isin(dev) | disc.fldNpdidField.notna()
# follow "included in other discovery" to parent
name2comm = dict(zip(disc.dscName, disc.commercial))
inc = disc.dscCurrentActivityStatus == 'INCLUDED IN OTHER DISCOVERY'
disc.loc[inc, 'commercial'] = disc.loc[inc, 'dscResInclInDiscoveryName'].map(name2comm).fillna(False)
comm_by_well = disc.groupby('wlbNpdidWellbore').commercial.max()
comm_by_disc = dict(zip(disc.dscNpdidDiscovery, disc.commercial))
t['commercial'] = 0
m = t.hc == 1
c1 = t.loc[m, 'wlbNpdidWellbore'].map(comm_by_well)
c2 = t.loc[m, 'dscNpdidDiscovery'].map(comm_by_disc)
t.loc[m, 'commercial'] = (c1.fillna(False).astype(bool) | c2.fillna(False).astype(bool)).astype(int)

# ---- geometry
R = 6371.0
lat_p = np.radians(pool.wlbNsDecDeg.astype(float).values); lon_p = np.radians(pool.wlbEwDesDeg.astype(float).values)
kd_p = pool.known_date.values; hc_p = pool.hc.values; lic_p = pool.wlbProductionLicence.astype(str).values
id_p = pool.wlbNpdidWellbore.values

def hav(lat1, lon1, lat2, lon2):
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

rows = []
for _, r in t.iterrows():
    la, lo = np.radians(float(r.wlbNsDecDeg)), np.radians(float(r.wlbEwDesDeg))
    known = (kd_p < np.datetime64(r.wlbEntryDate)) & (id_p != r.wlbNpdidWellbore)
    d = hav(la, lo, lat_p[known], lon_p[known]); h = hc_p[known]
    f = {'wlbNpdidWellbore': r.wlbNpdidWellbore}
    for km in (10, 25, 50, 100):
        sel = d <= km
        f[f'n_wells_{km}km'] = int(sel.sum()); f[f'n_hc_{km}km'] = int(h[sel].sum())
    f['dist_nearest_well'] = float(d.min()) if len(d) else 500.0
    f['dist_nearest_hc'] = float(d[h == 1].min()) if (h == 1).any() else 500.0
    f['dist_nearest_dry'] = float(d[h == 0].min()) if (h == 0).any() else 500.0
    nn = np.argsort(d)[:1]
    f['nearest_well_hc'] = int(h[nn][0]) if len(d) else -1
    # Gaussian-kernel smoothed local success rate with Beta prior toward the global prior rate
    g = h.mean() if len(h) else 0.4
    wts = np.exp(-0.5 * (d / 20.0) ** 2)
    f['kernel_rate_20km'] = float((wts @ h + 2 * g) / (wts.sum() + 2))
    f['kernel_weight_20km'] = float(wts.sum())
    f['global_prior_rate'] = float(g)
    same_lic = known & (lic_p == str(r.wlbProductionLicence))
    f['n_wells_same_licence'] = int(same_lic.sum()); f['n_hc_same_licence'] = int(hc_p[same_lic].sum())
    rows.append(f)
F = pd.DataFrame(rows)
t = t.merge(F, on='wlbNpdidWellbore')

# licence age at spud (pre-drill)
lic['prlDateGranted'] = pd.to_datetime(lic.prlDateGranted, errors='coerce')
t = t.merge(lic[['prlNpdidLicence', 'prlDateGranted']], left_on='prlNpdidProductionLicence', right_on='prlNpdidLicence', how='left')
t['licence_age_yrs'] = (t.wlbEntryDate - t.prlDateGranted).dt.days / 365.25
t['round_type'] = np.where(t.wlbLicensingActivity.astype(str).str.contains('APA|TFO', regex=True), 'APA',
                    np.where(t.wlbLicensingActivity.astype(str).str.contains('SOKNADSRUNDE|ROUND|runde', case=False, regex=True), 'ROUND', 'OTHER'))
t['year'] = t.wlbEntryDate.dt.year
t['lat'] = t.wlbNsDecDeg.astype(float); t['lon'] = t.wlbEwDesDeg.astype(float)
t['water_depth'] = t.wlbWaterDepth.astype(float)
t['total_depth_md'] = t.wlbTotalDepth.astype(float)   # post-drill actual; used ONLY in a leakage sensitivity test
t.to_csv(OUT, index=False)
print(t.shape, 'hc rate', t.hc.mean().round(3), 'commercial rate', t.commercial.mean().round(3))
print(t.groupby(t.year // 5 * 5)[['hc', 'commercial']].agg(['mean', 'count']).round(2).to_string())
