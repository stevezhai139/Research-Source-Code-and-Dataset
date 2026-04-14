"""
HSM Validation on Real SDSS SkyServer Query Logs
=================================================
Replicates A8 validation using real query logs (upgrading from simulated SDSS data).
Computes HSM 5-dimensional similarity across real workload windows and reports
discrimination ratio (within-phase vs cross-phase), Mann-Whitney U, and ICC(2,1).

HSM Dimensions (per paper Section 3):
  S_R : SELECT-ratio similarity      = 1 - |ratio_A - ratio_B|
  S_V : Volume similarity            = exp(-|log(QPS_A) - log(QPS_B)|)
  S_T : Type angular similarity      = cosine([n_sel,n_upd,n_ins,n_del])
  S_A : Access-attribute overlap     = 0.5*Jaccard(tables) + 0.5*Jaccard(columns)
  S_P : Temporal pattern similarity  = DWT(db4,L=1) + SAX(α=4) + FastDTW(r=1)
        via wavedec → SAX encode each band → FastDTW distance → weighted avg

HSM = 0.2*S_R + 0.2*S_V + 0.2*S_T + 0.2*S_A + 0.2*S_P
"""

import re, csv, math, statistics, random, warnings
from collections import Counter, defaultdict
import numpy as np
from scipy import stats
import pywt
from fastdtw import fastdtw

# ── Config ────────────────────────────────────────────────────────────────────
NPTS          = 20          # queries per window (≈ TPC-H mean 19.5)
RANDOM_SEED   = 42
HSM_WEIGHTS   = [0.2, 0.2, 0.2, 0.2, 0.2]   # w_R, w_V, w_T, w_A, w_P
DWT_WAVELET   = 'db4'
DWT_LEVEL     = 1
BAND_WEIGHTS  = [0.40, 0.60]   # [cA1, cD1]
SAX_ALPHA     = 4
FASTDTW_R     = 1
N_SLOTS       = 8
FILE_PATH     = ('/sessions/awesome-eager-dijkstra/mnt/Research Papers/'
                 'Paper 3 /HSM/Experimental Code/data/SkyLog_Workload.csv')

# ── Step 1: Parse CSV (multi-line SQL) ────────────────────────────────────────
print("=" * 70)
print("  HSM Validation — Real SDSS SkyServer Query Logs")
print("=" * 70)
print("\n[1] Parsing SkyLog_Workload.csv ...")

record_end = re.compile(
    r',(\d+/\d+/\d{4}\s[\d:]+\s[AP]M),([\d.]+),([\d.]+),(\d+),([^,]*),(\d+)\s*$'
)

records = []
current_lines = []

with open(FILE_PATH, 'r', encoding='utf-8', errors='replace') as f:
    next(f)  # skip header
    for line in f:
        m = record_end.search(line.rstrip())
        if m:
            sql_part = line[:m.start()]
            current_lines.append(sql_part)
            full_sql = ' '.join(current_lines).strip()
            records.append({
                'sql'    : full_sql,
                'time'   : m.group(1),
                'elapsed': float(m.group(2)),
                'rows'   : int(m.group(4)),
                'dbname' : m.group(5).strip(),
                'error'  : m.group(6)
            })
            current_lines = []
        else:
            current_lines.append(line.rstrip())

print(f"  Parsed {len(records):,} query records")

# ── Step 2: Extract query features per record ─────────────────────────────────
print("\n[2] Extracting query features ...")

def extract_tables(sql):
    """Extract table/function names from FROM and JOIN clauses."""
    sql_upper = sql.upper()
    tables = set()
    for m in re.finditer(r'\bFROM\s+([A-Za-z][A-Za-z0-9_]*)', sql, re.I):
        tables.add(m.group(1).lower())
    for m in re.finditer(r'\bJOIN\s+([A-Za-z][A-Za-z0-9_]*)', sql, re.I):
        tables.add(m.group(1).lower())
    return tables

def extract_columns(sql):
    """Extract column names from WHERE / ORDER BY clauses (heuristic)."""
    cols = set()
    for m in re.finditer(r'\bWHERE\s+(\w+)\s*[=<>!]', sql, re.I):
        cols.add(m.group(1).lower())
    for m in re.finditer(r'\bAND\s+(\w+)\s*[=<>!]', sql, re.I):
        cols.add(m.group(1).lower())
    for m in re.finditer(r'\bORDER\s+BY\s+(\w+)', sql, re.I):
        cols.add(m.group(1).lower())
    return cols

def get_qtype(sql):
    s = sql.strip().upper()
    for t in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'EXEC']:
        if s.startswith(t):
            return t
    return 'OTHER'

for r in records:
    r['qtype']   = get_qtype(r['sql'])
    r['tables']  = extract_tables(r['sql'])
    r['columns'] = extract_columns(r['sql'])
    r['access']  = r['tables'] | r['columns']

print(f"  Features extracted for {len(records):,} records")

# ── Step 3: Define phases based on dominant table patterns ─────────────────────
print("\n[3] Identifying workload phases ...")

# Classify each query into a workload theme
PHASE_MAP = {
    'spatial'      : {'fgetnearbyobjeq', 'fgetnearbyapogeestareq',
                      'fgetnearbyprimobj', 'fgetnearestspec'},
    'photometric'  : {'photoobjall', 'photoprimary', 'phototag',
                      'photoobj', 'photoz', 'photozrf', 'field',
                      'galaxy', 'star'},
    'spectroscopic': {'apogeestar', 'aspcapstar', 'specobjall',
                      'sppparams', 'galspecline', 'galspecinfo'},
    'metadata'     : {'dbobjects', 'indexmap', 'dbo', 'systables'},
}

def classify_phase(tables):
    scores = {ph: len(tables & kw) for ph, kw in PHASE_MAP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'mixed'

for r in records:
    r['phase'] = classify_phase(r['tables'])

phase_counts = Counter(r['phase'] for r in records)
print("  Phase distribution:")
for ph, cnt in phase_counts.most_common():
    print(f"    {ph:15s}: {cnt:6,} ({100*cnt/len(records):.1f}%)")

# ── Step 4: Build windows ─────────────────────────────────────────────────────
print(f"\n[4] Building windows (Npts={NPTS} queries each) ...")

# Use only SELECT queries (99.1%) for cleaner analysis
select_records = [r for r in records if r['qtype'] == 'SELECT']
print(f"  Using {len(select_records):,} SELECT queries")

# Chunk into windows of NPTS queries
windows = []
for i in range(0, len(select_records) - NPTS, NPTS):
    chunk = select_records[i:i + NPTS]
    windows.append(chunk)

print(f"  Created {len(windows):,} windows of {NPTS} queries each")

# Assign window phase = majority phase in that window
def window_phase(w):
    phases = Counter(r['phase'] for r in w)
    return phases.most_common(1)[0][0]

for w in windows:
    pass  # phase assigned per-pair below

# ── Step 5: Compute HSM dimensions per window ─────────────────────────────────
print("\n[5] Computing HSM window features ...")

def compute_window_features(chunk):
    n = len(chunk)
    elapsed = [r['elapsed'] for r in chunk]

    # S_R: SELECT ratio (all are SELECT here, so vary by detecting sub-types)
    n_sel = sum(1 for r in chunk if r['qtype'] == 'SELECT')
    n_upd = sum(1 for r in chunk if r['qtype'] == 'UPDATE')
    n_ins = sum(1 for r in chunk if r['qtype'] == 'INSERT')
    n_del = sum(1 for r in chunk if r['qtype'] == 'DELETE')
    ratio_sel = n_sel / n

    # Volume: QPS = queries / total_elapsed  (avoid div-by-zero)
    total_elapsed = sum(max(e, 0.001) for e in elapsed)
    qps = n / total_elapsed

    # Access attribute set — keep tables and columns separate for S_A
    table_set = set()
    col_set   = set()
    for r in chunk:
        table_set.update(r['tables'])
        col_set.update(r['columns'])

    # Temporal pattern: build N_SLOTS=8 time-slot series (avg rate per slot)
    # Distribute queries evenly across N_SLOTS bins by sequential index
    rates = np.array([1.0/max(r['elapsed'], 0.001) for r in chunk])
    norm_s = np.arange(n, dtype=float) / max(n - 1, 1)
    slot_idx = np.floor(norm_s * N_SLOTS).clip(0, N_SLOTS - 1).astype(int)
    series = np.zeros(N_SLOTS)
    counts = np.zeros(N_SLOTS)
    for sl, rate in zip(slot_idx, rates):
        series[sl] += rate
        counts[sl] += 1
    counts[counts == 0] = 1
    qps_series = series / counts

    return {
        'ratio_sel' : ratio_sel,
        'qps'       : qps,
        'type_vec'  : np.array([n_sel, n_upd, n_ins, n_del], dtype=float),
        'tables'    : table_set,
        'columns'   : col_set,
        'ts'        : qps_series,
        'phase'     : window_phase(chunk),
    }

win_features = [compute_window_features(w) for w in windows]
print(f"  Computed features for {len(win_features)} windows")

# ── Step 6: HSM similarity computation ────────────────────────────────────────
print("\n[6] Computing pairwise HSM scores ...")

def s_r(fa, fb):
    return 1.0 - abs(fa['ratio_sel'] - fb['ratio_sel'])

def s_v(fa, fb):
    qa, qb = max(fa['qps'], 1e-9), max(fb['qps'], 1e-9)
    return math.exp(-abs(math.log(qa) - math.log(qb)))

def s_t(fa, fb):
    va, vb = fa['type_vec'], fb['type_vec']
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 1.0 if na < 1e-9 and nb < 1e-9 else 0.0
    cos = np.dot(va, vb) / (na * nb)
    return float(np.clip(cos, 0.0, 1.0))

def s_a(fa, fb):
    """S_A = 0.5*Jaccard(tables) + 0.5*Jaccard(columns)"""
    ta, tb = fa['tables'], fb['tables']
    ca, cb = fa['columns'], fb['columns']
    tj = len(ta & tb) / len(ta | tb) if (ta | tb) else 1.0
    cj = len(ca & cb) / len(ca | cb) if (ca | cb) else 1.0
    return 0.5 * tj + 0.5 * cj

def _sax_encode(arr):
    """SAX encode array into α=SAX_ALPHA symbols."""
    from scipy.stats import norm as _norm
    if arr.std() < 1e-9:
        return np.zeros(len(arr), dtype=float)
    z  = (arr - arr.mean()) / arr.std()
    bp = _norm.ppf(np.linspace(0, 1, SAX_ALPHA + 1)[1:-1])
    return np.digitize(z, bp).astype(float)

def _band_score(c1, c2):
    """FastDTW distance between two SAX-encoded DWT bands → similarity."""
    s1, s2 = _sax_encode(c1), _sax_encode(c2)
    dist, _ = fastdtw(s1.tolist(), s2.tolist(), radius=FASTDTW_R,
                      dist=lambda a, b: abs(float(a) - float(b)))
    max_dist = len(s1) * (SAX_ALPHA - 1)
    return 1.0 - dist / max(max_dist, 1.0)

def s_p(fa, fb):
    """S_P = DWT(db4,L=1) + SAX(α=4) + FastDTW(r=1), band-weighted average."""
    q1, q2 = fa['ts'], fb['ts']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c1 = pywt.wavedec(q1, DWT_WAVELET, level=DWT_LEVEL)
        c2 = pywt.wavedec(q2, DWT_WAVELET, level=DWT_LEVEL)
    n_bands = min(len(BAND_WEIGHTS), len(c1))
    score, w_sum = 0.0, sum(BAND_WEIGHTS[:n_bands])
    for i in range(n_bands):
        cb1, cb2 = c1[i], c2[i]
        if len(cb1) < 2:
            bs = max(0.0, 1.0 - abs(float(cb1[0]) - float(cb2[0])) /
                     max(abs(float(cb1[0])) + abs(float(cb2[0])), 1.0))
        else:
            bs = _band_score(cb1, cb2)
        score += BAND_WEIGHTS[i] * bs
    return float(np.clip(score / w_sum, 0.0, 1.0))

def hsm(fa, fb):
    sr = s_r(fa, fb)
    sv = s_v(fa, fb)
    st = s_t(fa, fb)
    sa = s_a(fa, fb)
    sp = s_p(fa, fb)
    score = 0.2*sr + 0.2*sv + 0.2*st + 0.2*sa + 0.2*sp
    return score, {'S_R': sr, 'S_V': sv, 'S_T': st, 'S_A': sa, 'S_P': sp}

# Compute consecutive window pairs
within_scores, cross_scores = [], []
within_dims  = defaultdict(list)
cross_dims   = defaultdict(list)

n_pairs = 0
for i in range(len(win_features) - 1):
    fa = win_features[i]
    fb = win_features[i + 1]
    score, dims = hsm(fa, fb)

    is_within = (fa['phase'] == fb['phase'])
    if is_within:
        within_scores.append(score)
        for k, v in dims.items():
            within_dims[k].append(v)
    else:
        cross_scores.append(score)
        for k, v in dims.items():
            cross_dims[k].append(v)
    n_pairs += 1

print(f"  Total pairs: {n_pairs}")
print(f"  Within-phase pairs: {len(within_scores)}")
print(f"  Cross-phase pairs:  {len(cross_scores)}")

# ── Step 7: Statistics ─────────────────────────────────────────────────────────
print("\n[7] Computing statistics ...")

w_mean  = statistics.mean(within_scores)
w_std   = statistics.stdev(within_scores)
c_mean  = statistics.mean(cross_scores)
c_std   = statistics.stdev(cross_scores)
dr      = w_mean / c_mean

# Mann-Whitney U
u_stat, p_val = stats.mannwhitneyu(within_scores, cross_scores, alternative='greater')
n1, n2 = len(within_scores), len(cross_scores)
r_biserial = 1 - 2*u_stat / (n1 * n2)

# 95% CI for DR (bootstrap)
rng = random.Random(RANDOM_SEED)
boot_drs = []
for _ in range(2000):
    s_w = [rng.choice(within_scores) for _ in range(n1)]
    s_c = [rng.choice(cross_scores)  for _ in range(n2)]
    m_c = statistics.mean(s_c)
    if m_c > 0:
        boot_drs.append(statistics.mean(s_w) / m_c)
boot_drs.sort()
ci_lo = boot_drs[int(0.025 * len(boot_drs))]
ci_hi = boot_drs[int(0.975 * len(boot_drs))]

# ICC(2,1) approximation
all_scores = within_scores + cross_scores
grand_mean = statistics.mean(all_scores)
ss_total = sum((x - grand_mean)**2 for x in all_scores)
ss_between = (n1*(w_mean - grand_mean)**2 + n2*(c_mean - grand_mean)**2)
ss_within = ss_total - ss_between
ms_between = ss_between / 1
ms_within  = ss_within / (len(all_scores) - 2)
icc = (ms_between - ms_within) / (ms_between + (2-1)*ms_within) if ms_between > ms_within else 0.0

# ── Step 8: Per-dimension analysis ────────────────────────────────────────────
print("\n[8] Per-dimension breakdown ...")

print(f"\n{'Dimension':<8}  {'Within':>8}  {'Cross':>8}  {'Delta':>8}")
print("-" * 45)
for dim in ['S_R', 'S_V', 'S_T', 'S_A', 'S_P']:
    wm = statistics.mean(within_dims[dim]) if within_dims[dim] else 0
    cm = statistics.mean(cross_dims[dim])  if cross_dims[dim]  else 0
    print(f"{dim:<8}  {wm:8.4f}  {cm:8.4f}  {wm-cm:+8.4f}")

# ── Step 9: Final Results ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  RESULTS: Real SDSS SkyServer Query Log Validation")
print("=" * 70)
print(f"\n  Windows         : {len(win_features):,} ({NPTS} queries each)")
print(f"  Within-phase    : {n1:,} pairs  (mean={w_mean:.4f}, σ={w_std:.4f})")
print(f"  Cross-phase     : {n2:,} pairs  (mean={c_mean:.4f}, σ={c_std:.4f})")
print(f"\n  Discrimination Ratio : {dr:.3f}  (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
print(f"  Mann-Whitney p       : {p_val:.3e}")
print(f"  Rank-biserial r      : {r_biserial:.3f}")
print(f"  ICC(2,1)             : {icc:.3f}")
print(f"\n  θ=0.75 separation    : ", end="")
below_thresh = sum(1 for s in cross_scores if s < 0.75)
above_thresh = sum(1 for s in within_scores if s >= 0.75)
print(f"{below_thresh}/{n2} cross-phase below θ  |  "
      f"{above_thresh}/{n1} within-phase above θ")

print(f"\n  Paper (simulated SDSS, A8) : DR=1.199, p=0.002")
print(f"  Real SDSS query logs       : DR={dr:.3f}, p={p_val:.3e}")
improvement = "BETTER" if dr > 1.199 else "COMPARABLE" if dr > 1.0 else "LOWER"
print(f"  Assessment                 : {improvement}")

print("\n" + "=" * 70)
print("  Phase transition examples (cross-phase pairs, lowest HSM score)")
print("=" * 70)
cross_pairs_detail = []
for i in range(len(win_features) - 1):
    fa = win_features[i]
    fb = win_features[i + 1]
    if fa['phase'] != fb['phase']:
        score, _ = hsm(fa, fb)
        cross_pairs_detail.append((score, fa['phase'], fb['phase']))
cross_pairs_detail.sort()
for score, pa, pb in cross_pairs_detail[:5]:
    print(f"  {pa:15s} → {pb:15s}  HSM={score:.4f}")
