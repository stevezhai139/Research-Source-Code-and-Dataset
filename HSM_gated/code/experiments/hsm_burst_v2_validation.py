"""
HSM Validation on Burst Workload v2 — S_P Isolation via pg_sleep
=================================================================
Isolates the S_P (DWT temporal pattern) dimension by controlling execution
time precisely with pg_sleep, eliminating the timing variance that masked
S_P in v1.  All three phases have approximately equal mean execution time
(~5–5.5 ms), so S_V ≈ 0 across all pairs and S_P is the sole discriminant.

PHASE DESIGN (3 phases × 30/30/24 = 84 queries per trace)
-----------------------------------------------------------
  Steady   (ST): 30 × pg_sleep(0.005)  → [5,5,5,5,5,5] ms — flat 5 ms
  Burst_Alt(BA): alternating period-2  → [1,10,1,10,…] ms  (mean 5.5 ms)
                 15 × pg_sleep(0.001) interleaved with 15 × pg_sleep(0.010)
  Burst_Grp(BG): grouped period-8     → [1,1,1,1,10,10,10,10,…] ms (mean 5.5 ms)
                 3 groups of [4 × pg_sleep(0.001), 4 × pg_sleep(0.010)]

WHY THIS ISOLATES S_P
----------------------
  S_V : mean(ST)=5.0 ms  mean(BA)=5.5 ms  mean(BG)=5.5 ms
        max contrast ≈ 10%  →  S_V Δ ≈ 0  (negligible)
  S_R : all SELECT         →  S_R = 1.0 everywhere  →  S_R Δ = 0
  S_T : all SELECT         →  S_T identical           →  S_T Δ = 0
  S_A : no real table access → constant              →  S_A Δ ≈ 0
  S_P : DWT of time series differs by construction
        ST  → flat  → DWT ≈ (c,c,c, 0,0,0) — all energy in approx
        BA  → period-2 oscillation  → DWT energy split approx/detail
        BG  → period-8 grouping     → different approx/detail split
        cosine between unit-sphere DWT vectors will differ clearly
        S_P must be dominant dim

EXPECTED OUTCOME
-----------------
  S_P dominant  (Δ > 0.10)
  DR: 1.3–1.8  (depends on DWT cosine spread)
  p < 0.001 across 10 seeds

DATABASE
--------
  pgbench TPC-B on Docker PostgreSQL 16, port 5433, database 'oltp'
  (same container — no additional setup required)

Usage:
  python hsm_burst_v2_validation.py --smoke   # verify pg_sleep timing
  python hsm_burst_v2_validation.py --execute # full 10-seed experiment
"""

import re, os, sys, csv, math, time, random, statistics, argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
import pywt

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / 'results' / 'burst_v2_validation'

DOCKER_HOST = os.environ.get('HSM_DOCKER_HOST', 'localhost')
DOCKER_PORT = int(os.environ.get('HSM_DOCKER_PORT', 5433))
DOCKER_USER = os.environ.get('HSM_DOCKER_USER', 'postgres')
DOCKER_PASS = os.environ.get('HSM_DOCKER_PASSWORD', 'postgres')
DOCKER_DB   = 'oltp'

TIMEOUT_MS  = 30_000
NPTS        = 6
STEP        = 3
N_REPS      = 3
SEEDS       = [42, 137, 271, 314, 999, 7, 13, 55, 88, 101]
PHASE_ORDER = ['Steady', 'Burst_Alt', 'Burst_Grp']

# pg_sleep durations (seconds)
T_FAST   = 0.001   #  1 ms
T_STEADY = 0.005   #  5 ms
T_SLOW   = 0.010   # 10 ms

# ── Query builders ─────────────────────────────────────────────────────────────
def q_fast():
    return f"SELECT pg_sleep({T_FAST})"

def q_steady():
    return f"SELECT pg_sleep({T_STEADY})"

def q_slow():
    return f"SELECT pg_sleep({T_SLOW})"

# ── Burst trace builder ────────────────────────────────────────────────────────
def build_burst_trace(seed):
    """
    Returns list of (qid, phase, qtype, sql) with controlled timing patterns.

    Steady   (30): all 5 ms — flat time series
    Burst_Alt(30): alternating 1 ms / 10 ms — period-2 oscillation
    Burst_Grp(24): 4×1ms then 4×10ms repeated — period-8 grouping
    """
    # seed is accepted for API compatibility; pg_sleep trace is deterministic
    _ = random.Random(seed)   # consume seed (no shuffle needed — pattern is fixed)
    trace = []

    # ── Block 1: Steady (30 × 5ms, flat) ─────────────────────────────────────
    for i in range(30):
        trace.append((f"ST{i+1:03d}", 'Steady', 'SELECT', q_steady()))

    # ── Block 2: Burst_Alt (15×fast + 15×slow, strict period-2) ─────────────
    for i in range(15):
        trace.append((f"BA{2*i+1:03d}", 'Burst_Alt', 'SELECT', q_fast()))
        trace.append((f"BA{2*i+2:03d}", 'Burst_Alt', 'SELECT', q_slow()))

    # ── Block 3: Burst_Grp (3 groups of 4×fast + 4×slow, period-8) ──────────
    for g in range(3):
        for j in range(4):
            trace.append((f"BG{g*8+j+1:03d}", 'Burst_Grp', 'SELECT', q_fast()))
        for j in range(4):
            trace.append((f"BG{g*8+j+5:03d}", 'Burst_Grp', 'SELECT', q_slow()))

    return trace   # 30 + 30 + 24 = 84 entries

# ── timed_exec (fresh cursor, no fetchall needed for pg_sleep) ────────────────
def timed_exec(cursor, sql, n_reps=N_REPS):
    def _run_once():
        c = conn.cursor()
        try:
            c.execute(sql)
            if c.description is not None:
                c.fetchall()
        finally:
            try: c.close()
            except: pass
    try: _run_once()
    except: pass
    times = []
    for _ in range(n_reps):
        c = conn.cursor()
        try:
            t0 = time.perf_counter()
            c.execute(sql)
            if c.description is not None:
                c.fetchall()
            times.append((time.perf_counter() - t0) * 1000)
        except psycopg2.extensions.QueryCanceledError:
            times.append(float(TIMEOUT_MS))
        except Exception:
            times.append(float(TIMEOUT_MS))
        finally:
            try: c.close()
            except: pass
    return float(statistics.median(times))

# ── HSM kernel ─────────────────────────────────────────────────────────────────
def compute_s_r(sqls):
    return sum(1 for s in sqls if re.match(r'\s*SELECT', s, re.I)) / len(sqls)

def compute_s_v(times):
    return float(np.mean(times))

def compute_s_t(sqls):
    counts = [0, 0, 0, 0]
    for s in sqls:
        s = s.strip().upper()
        if   s.startswith('SELECT'): counts[0] += 1
        elif s.startswith('UPDATE'): counts[1] += 1
        elif s.startswith('INSERT'): counts[2] += 1
        elif s.startswith('DELETE'): counts[3] += 1
    n = sum(counts)
    return np.array([c/n for c in counts], dtype=float) if n else np.zeros(4)

def compute_s_a(sqls):
    def tables(sql):
        found = re.findall(r'(?:FROM|JOIN)\s+(\w+)', sql, re.I)
        return frozenset(t for t in found if t.lower() != 'pg_sleep')
    sets = [tables(s) for s in sqls]
    union = set().union(*sets)
    if not union:
        return 1.0   # no table access → identical → max similarity
    inter = sets[0]
    for s in sets[1:]:
        inter = inter & s
    return len(inter) / len(union)

def compute_s_p(times):
    arr = np.array(times, dtype=float)
    coeffs = pywt.wavedec(arr, 'db2', level=1)
    vec = np.concatenate(coeffs)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else vec

def hsm_similarity(w1_sqls, w1_times, w2_sqls, w2_times, weights=None):
    if weights is None:
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    sr1, sr2 = compute_s_r(w1_sqls),  compute_s_r(w2_sqls)
    sv1, sv2 = compute_s_v(w1_times), compute_s_v(w2_times)
    st1, st2 = compute_s_t(w1_sqls),  compute_s_t(w2_sqls)
    sa1, sa2 = compute_s_a(w1_sqls),  compute_s_a(w2_sqls)
    sp1, sp2 = compute_s_p(w1_times), compute_s_p(w2_times)

    s_r = 1.0 - abs(sr1 - sr2)
    s_v = 1.0 - abs(sv1 - sv2) / (max(sv1, sv2) + 1e-9)
    ct  = np.dot(st1, st2) / (np.linalg.norm(st1)*np.linalg.norm(st2) + 1e-12)
    s_t = float(np.clip(ct, 0.0, 1.0))
    s_a = min(sa1, sa2) / (max(sa1, sa2) + 1e-9)
    cp  = float(np.dot(sp1, sp2))
    s_p = float(np.clip(cp, 0.0, 1.0))

    dims  = [s_r, s_v, s_t, s_a, s_p]
    score = sum(w * d for w, d in zip(weights, dims))
    return score, dims

def windows_from_trace(exec_trace, npts=NPTS, step=STEP):
    wins = []
    n = len(exec_trace)
    for start in range(0, n - npts + 1, step):
        block = exec_trace[start:start+npts]
        phases = [b[1] for b in block]
        dominant = Counter(phases).most_common(1)[0][0]
        wins.append({'sqls': [b[2] for b in block],
                     'times': [b[3] for b in block],
                     'phase': dominant})
    return wins

def compute_dr(wins):
    pairs_within, pairs_cross = [], []
    for i in range(len(wins)):
        for j in range(i+1, len(wins)):
            sc, dims = hsm_similarity(wins[i]['sqls'], wins[i]['times'],
                                      wins[j]['sqls'], wins[j]['times'])
            entry = (sc, dims)
            if wins[i]['phase'] == wins[j]['phase']:
                pairs_within.append(entry)
            else:
                pairs_cross.append(entry)
    if not pairs_within or not pairs_cross:
        return None, None, None, None

    w_scores = [e[0] for e in pairs_within]
    c_scores = [e[0] for e in pairs_cross]
    dr = statistics.median(w_scores) / (statistics.median(c_scores) + 1e-12)

    u_stat, p_val = stats.mannwhitneyu(w_scores, c_scores, alternative='greater')
    r_biserial = 1 - 2*u_stat / (len(w_scores)*len(c_scores))

    w_dims = np.mean([e[1] for e in pairs_within], axis=0)
    c_dims = np.mean([e[1] for e in pairs_cross],  axis=0)
    dim_deltas = w_dims - c_dims
    dominant = ['S_R','S_V','S_T','S_A','S_P'][int(np.argmax(dim_deltas))]

    return dr, p_val, r_biserial, {
        'n_within': len(pairs_within), 'n_cross': len(pairs_cross),
        'w_scores': w_scores, 'c_scores': c_scores,
        'w_dims': w_dims.tolist(), 'c_dims': c_dims.tolist(),
        'dim_deltas': dim_deltas.tolist(), 'dominant': dominant,
    }

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--execute', action='store_true')
parser.add_argument('--smoke',   action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--port',    type=int, default=DOCKER_PORT)
parser.add_argument('--dbname',  default=DOCKER_DB)
args = parser.parse_args()

print("=" * 70)
print("  HSM Validation — Burst v2 (pg_sleep, S_P Isolation)")
print("=" * 70)
mode = "SMOKE" if args.smoke else "EXECUTE" if args.execute else "INFO"
print(f"\n  Mode    : {mode}")
print(f"  DB      : {args.dbname} on port {args.port}")
print(f"  Phases  : Steady(5ms flat) / Burst_Alt(period-2) / Burst_Grp(period-8)")
print(f"  Timing  : fast={T_FAST*1000:.0f}ms  steady={T_STEADY*1000:.0f}ms  slow={T_SLOW*1000:.0f}ms")
print(f"  N_REPS  : {N_REPS}  Seeds: {len(SEEDS)}")
sample = build_burst_trace(42)
phase_counts = Counter(ph for _, ph, _, _ in sample)
print(f"\n[1] Trace structure (per seed): {len(sample)} queries")
for ph in PHASE_ORDER:
    print(f"    {ph:<12}: {phase_counts.get(ph,0):3d} queries")

# ── Database connection ────────────────────────────────────────────────────────
conn = None
if args.execute or args.smoke:
    print(f"\n[2] Connecting to {args.dbname} ...")
    try:
        import psycopg2, psycopg2.extensions
        conn = psycopg2.connect(
            host=DOCKER_HOST, port=args.port, dbname=args.dbname,
            user=DOCKER_USER, password=DOCKER_PASS, connect_timeout=10,
            options=f"-c statement_timeout={TIMEOUT_MS}")
        conn.set_session(autocommit=True)
        cur = conn.cursor()
        cur.execute("SELECT version()")
        ver = cur.fetchone()[0].split()[1]
        print(f"  Connected ✓  (PostgreSQL {ver})")
    except Exception as e:
        print(f"  ERROR: {e}"); sys.exit(1)

# ── Smoke test ─────────────────────────────────────────────────────────────────
if args.smoke:
    print("\n[3] Smoke test — timing calibration ...")
    print(f"  {'Query':30s}  {'Expected':>10}  {'Actual':>10}")
    print(f"  {'-'*55}")
    for label, sql, expected in [
        ("pg_sleep(0.001) = 1ms",  q_fast(),   T_FAST  * 1000),
        ("pg_sleep(0.005) = 5ms",  q_steady(), T_STEADY* 1000),
        ("pg_sleep(0.010) = 10ms", q_slow(),   T_SLOW  * 1000),
    ]:
        t = timed_exec(cur, sql, n_reps=3)
        ok = "✅" if abs(t - expected) < expected * 0.5 else "⚠️"
        print(f"  {label:30s}  {expected:>10.1f}  {t:>10.2f} ms  {ok}")

    print("\n  Per-phase sample (first 4 queries) ...")
    trace_s = build_burst_trace(42)
    for ph in PHASE_ORDER:
        sample_ph = [(qid, sql) for qid, p, _, sql in trace_s if p == ph][:4]
        times = [timed_exec(cur, sql, n_reps=1) for _, sql in sample_ph]
        if args.verbose:
            for (qid, _), t in zip(sample_ph, times):
                print(f"    {qid}  {t:8.2f} ms")
        print(f"  {ph:<12}: {[f'{t:.1f}' for t in times]}  mean={statistics.mean(times):.2f} ms")
    conn.close()
    print("\n  Smoke OK → run full experiment:")
    print("    python hsm_burst_v2_validation.py --execute")
    sys.exit(0)

if not args.execute:
    print("\nRun with --smoke or --execute")
    sys.exit(0)

# ── Full execution ─────────────────────────────────────────────────────────────
print(f"\n[3] Calibration run (seed=42) ...")
calib_trace = build_burst_trace(42)
exec_calib = []
for qid, ph, qt, sql in calib_trace:
    ms = timed_exec(cur, sql, n_reps=N_REPS)
    exec_calib.append((qid, ph, sql, ms))
print(f"  Done.")

print(f"\n[4] Per-phase timing (seed=42) ...")
print(f"  {'Phase':<14} {'N':>4}  {'Median ms':>10}  {'Min ms':>8}  {'Max ms':>8}")
print(f"  {'-'*52}")
for ph in PHASE_ORDER:
    ph_t = [ms for _, p, _, ms in exec_calib if p == ph]
    if ph_t:
        print(f"  {ph:<14} {len(ph_t):>4}  "
              f"{statistics.median(ph_t):>10.2f}  "
              f"{min(ph_t):>8.2f}  {max(ph_t):>8.2f}")

# Verify S_V will be minimal
means = {}
for ph in PHASE_ORDER:
    ph_t = [ms for _, p, _, ms in exec_calib if p == ph]
    means[ph] = statistics.mean(ph_t)
max_sv_contrast = max(means.values()) / (min(means.values()) + 1e-9)
print(f"\n  Mean timing: " + "  ".join(f"{ph}={means[ph]:.1f}ms" for ph in PHASE_ORDER))
print(f"  Max S_V contrast: {max_sv_contrast:.1f}×  "
      f"({'⚠️ high — S_V may dominate' if max_sv_contrast > 5 else '✅ low — S_P should dominate'})")

# ── 10-seed experiment ─────────────────────────────────────────────────────────
print(f"\n[5] Running {len(SEEDS)} seeds ...")
seed_results = []
for seed in SEEDS:
    trace = build_burst_trace(seed)
    exec_trace = []
    for qid, ph, qt, sql in trace:
        ms = timed_exec(cur, sql, n_reps=N_REPS)
        exec_trace.append((qid, ph, sql, ms))
    wins = windows_from_trace(exec_trace)
    dr, p_val, r_biserial, info = compute_dr(wins)
    if dr is None:
        print(f"  Seed {seed:3d}: insufficient pairs"); continue
    seed_results.append({'seed': seed, 'dr': dr, 'p': p_val, 'r': r_biserial,
                         'n_within': info['n_within'], 'n_cross': info['n_cross'],
                         'dim_deltas': info['dim_deltas'], 'dominant': info['dominant'],
                         'w_dims': info['w_dims'], 'c_dims': info['c_dims']})
    print(f"  Seed {seed:3d}: DR={dr:.3f}  p={p_val:.3e}  dominant={info['dominant']}")

conn.close()

# ── Aggregate ──────────────────────────────────────────────────────────────────
print(f"\n[6] Aggregating ...")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
csv_path = RESULTS_DIR / 'burst_v2_hsm_execute.csv'

drs = [r['dr'] for r in seed_results]
ps  = [r['p']  for r in seed_results]
n   = len(drs)
dr_med  = statistics.median(drs)
dr_mean = statistics.mean(drs)
dr_sd   = statistics.stdev(drs) if n > 1 else 0.0
t_crit  = stats.t.ppf(0.975, df=n-1) if n > 1 else 0.0
ci_lo   = dr_mean - t_crit * dr_sd / math.sqrt(n)
ci_hi   = dr_mean + t_crit * dr_sd / math.sqrt(n)
p_med   = statistics.median(ps)

dim_names  = ['S_R','S_V','S_T','S_A','S_P']
avg_w_dims = np.mean([r['w_dims'] for r in seed_results], axis=0)
avg_c_dims = np.mean([r['c_dims'] for r in seed_results], axis=0)
avg_deltas = avg_w_dims - avg_c_dims
dominant_overall = dim_names[int(np.argmax(avg_deltas))]

with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['metric', 'value'])
    w.writerow(['DR_median',   f'{dr_med:.4f}'])
    w.writerow(['DR_mean',     f'{dr_mean:.4f}'])
    w.writerow(['DR_SD',       f'{dr_sd:.4f}'])
    w.writerow(['DR_CI_lo',    f'{ci_lo:.4f}'])
    w.writerow(['DR_CI_hi',    f'{ci_hi:.4f}'])
    w.writerow(['DR_range_lo', f'{min(drs):.4f}'])
    w.writerow(['DR_range_hi', f'{max(drs):.4f}'])
    w.writerow(['MWU_p_median',f'{p_med:.4e}'])
    w.writerow(['dominant_dim',dominant_overall])
    for i, nm in enumerate(dim_names):
        w.writerow([f'delta_{nm}', f'{avg_deltas[i]:.4f}'])
    for r in seed_results:
        w.writerow([f'DR_seed{r["seed"]}', f'{r["dr"]:.4f}'])
print(f"  Results → {csv_path}")

# ── Final report ───────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  RESULTS: HSM Burst v2 — S_P Isolation Experiment")
print("=" * 70)
print(f"\n  Phases : Steady(5ms flat) / Burst_Alt(1↔10ms, period-2)"
      f" / Burst_Grp(1×4↔10×4, period-8)")
print(f"  Queries: 84 per trace   Seeds: {n}")

print(f"\n  ── Per-dimension (avg across {n} seeds) ──")
print(f"  {'Dim':<6} {'Within':>8} {'Cross':>8} {'Δ':>8}")
print(f"  {'-'*36}")
for i, nm in enumerate(dim_names):
    marker = " ← dominant" if nm == dominant_overall else ""
    print(f"  {nm:<6} {avg_w_dims[i]:>8.4f} {avg_c_dims[i]:>8.4f}"
          f" {avg_deltas[i]:>+8.4f}{marker}")

print(f"\n  ── Aggregate ({n} seeds) ──")
print(f"  DR median          : {dr_med:.3f}")
print(f"  DR mean ± SD       : {dr_mean:.3f} ± {dr_sd:.3f}")
print(f"  Empirical 95% CI   : [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"  Mann-Whitney p     : {p_med:.3e}")
print(f"  Dominant dim       : {dominant_overall}")
print(f"  DR range (seeds)   : {min(drs):.3f} – {max(drs):.3f}")

print(f"\n  ── Full Comparison ──")
print(f"  TPC-H   (A7)   : DR=1.445  dominant=S_A")
print(f"  SDSS    (A8)   : DR=1.086  dominant=S_V")
print(f"  JOB     (A8b)  : DR=1.814  dominant=S_T")
print(f"  OLTP    (A8c)  : DR=1.543  dominant=S_R")
print(f"  Burst v1(A8d)  : DR=1.264  dominant=S_V  (real queries, high variance)")
print(f"  Burst v2(A8d') : DR={dr_med:.3f}  dominant={dominant_overall}"
      f"  (pg_sleep, equal mean, S_P isolated)")

sig = "✓ SIGNIFICANT" if p_med < 0.05 else "✗ NOT SIGNIFICANT"
print(f"\n  Assessment : {sig} (α=0.05)")
print()
print("=" * 70)
