"""
HSM Validation on OLTP — pgbench TPC-B Schema
==============================================
Tests HSM's ability to discriminate workload phases in a write-heavy
OLTP setting, directly addressing the "OLAP-only" limitation of the
existing TPC-H, SDSS, and JOB/IMDB validations.

PHASE DESIGN (4 phases, ~30 queries each)
------------------------------------------
  ReadOnly  (RO): Pure SELECT — indexed point reads + range aggregates
                  S_R = 1.0,  S_T = [n, 0, 0, 0],  QPS high
  WriteHeavy(WH): UPDATE + INSERT dominant, minimal SELECT
                  S_R < 0.15, S_T = [few, many, many, 0], QPS low
  Mixed     (MX): Balanced SELECT / UPDATE / INSERT
                  S_R ≈ 0.5,  S_T = [mid, mid, mid, 0], QPS mid
  BulkRead  (BR): Analytical SELECT — full-table scans, GROUP BY, joins
                  S_R = 1.0,  S_T = [n, 0, 0, 0],  QPS low

HSM DISCRIMINATION PREDICTION
------------------------------
  S_T: RO vs WH → cosine ≈ 0.10–0.20 (strong cross-phase signal)
  S_R: RO/BR = 1.0 vs WH = 0.15 (strong contrast)
  S_V: BR slow (~200–500 ms) vs RO fast (~0.3–2 ms)
  S_A: RO/WH share same tables but different column access profiles
  Expected DR: 1.4–2.0

DATABASE
--------
  Schema  : pgbench TPC-B (pgbench_accounts 1M rows, pgbench_tellers 100,
            pgbench_branches 10, pgbench_history starts empty)
  Backend : Docker PostgreSQL 16, port 5433, database 'oltp'
  Write side-effects: UPDATEs and INSERTs commit permanently (autocommit).
  This is intentional — OLTP systems process real writes.

Usage:
  python hsm_oltp_validation.py            # static proxy mode
  python hsm_oltp_validation.py --execute  # real timing (recommended)
  python hsm_oltp_validation.py --smoke    # connectivity + per-phase sample
"""

import re, os, sys, csv, math, time, random, statistics, argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import pywt

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / 'results' / 'oltp_validation'

DOCKER_HOST = os.environ.get('HSM_DOCKER_HOST', 'localhost')
DOCKER_PORT = int(os.environ.get('HSM_DOCKER_PORT', 5433))
DOCKER_USER = os.environ.get('HSM_DOCKER_USER', 'postgres')
DOCKER_PASS = os.environ.get('HSM_DOCKER_PASSWORD', 'postgres')
DOCKER_DB   = 'oltp'

TIMEOUT_MS  = 30_000       # 30 s cap (OLTP queries should all finish < 5 s)
NPTS        = 6
STEP        = 3
N_REPS      = 3
SEEDS       = [42, 137, 271, 314, 999, 7, 13, 55, 88, 101]
PHASE_ORDER = ['ReadOnly', 'WriteHeavy', 'Mixed', 'BulkRead']

# ── Fixed Parameter Sets (deterministic → reproducible timing) ─────────────────
AIDS    = [1, 42, 1000, 5000, 10000, 50000, 100000,
           250000, 500000, 750000, 900000, 999999]
BIDS    = list(range(1, 11))           # 1–10
TIDS    = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
DELTAS  = [100, 250, 500, 1000, 2500]
RANGES  = [(1, 100000), (100001, 200000), (200001, 400000),
           (400001, 600000), (600001, 800000), (800001, 1000000)]

# ── Phase Query Pools ──────────────────────────────────────────────────────────
def build_query_pool():
    """
    Build ~30 queries per phase with varied parameters.
    Returns list of (qid, phase, qtype, sql) tuples.
    qtype ∈ {'SELECT','UPDATE','INSERT','DELETE'}
    """
    pool = []
    qnum = [0]

    def add(phase, qtype, sql):
        qnum[0] += 1
        pool.append((f"{phase[:2]}{qnum[0]:03d}", phase, qtype, sql))

    # ── ReadOnly: indexed point reads + small range aggregates ────────────────
    for aid in AIDS:
        add('ReadOnly', 'SELECT',
            f"SELECT aid, bid, abalance FROM pgbench_accounts WHERE aid = {aid}")
    for tid in TIDS[:6]:
        add('ReadOnly', 'SELECT',
            f"SELECT tid, bid, tbalance FROM pgbench_tellers WHERE tid = {tid}")
    for bid in BIDS[:5]:
        add('ReadOnly', 'SELECT',
            f"SELECT bid, bbalance FROM pgbench_branches WHERE bid = {bid}")
    for aid in AIDS[:7]:
        add('ReadOnly', 'SELECT',
            f"SELECT a.aid, a.abalance, b.bbalance "
            f"FROM pgbench_accounts a JOIN pgbench_branches b ON a.bid = b.bid "
            f"WHERE a.aid = {aid}")

    # ── WriteHeavy: UPDATE + INSERT dominant ──────────────────────────────────
    # Account balance updates
    for aid, delta in zip(AIDS, DELTAS * 3):
        add('WriteHeavy', 'UPDATE',
            f"UPDATE pgbench_accounts SET abalance = abalance + {delta} "
            f"WHERE aid = {aid}")
    # Teller balance updates
    for tid, delta in zip(TIDS[:8], DELTAS * 2):
        add('WriteHeavy', 'UPDATE',
            f"UPDATE pgbench_tellers SET tbalance = tbalance + {delta} "
            f"WHERE tid = {tid}")
    # Branch balance updates
    for bid, delta in zip(BIDS[:5], DELTAS):
        add('WriteHeavy', 'UPDATE',
            f"UPDATE pgbench_branches SET bbalance = bbalance + {delta} "
            f"WHERE bid = {bid}")
    # History inserts (core OLTP write)
    for i, (aid, delta) in enumerate(zip(AIDS[:8], DELTAS * 2)):
        tid  = TIDS[i % len(TIDS)]
        bid  = BIDS[i % len(BIDS)]
        add('WriteHeavy', 'INSERT',
            f"INSERT INTO pgbench_history(tid, bid, aid, delta, mtime) "
            f"VALUES ({tid}, {bid}, {aid}, {delta}, NOW())")
    # Minimal read-back (< 15% SELECT)
    for aid in AIDS[:3]:
        add('WriteHeavy', 'SELECT',
            f"SELECT abalance FROM pgbench_accounts WHERE aid = {aid}")

    # ── Mixed: balanced SELECT / UPDATE / INSERT ───────────────────────────────
    # Alternating read–then–update pattern (simulate TPC-B transaction steps)
    for aid, delta in zip(AIDS[:8], DELTAS * 2):
        add('Mixed', 'SELECT',
            f"SELECT abalance FROM pgbench_accounts WHERE aid = {aid}")
        add('Mixed', 'UPDATE',
            f"UPDATE pgbench_accounts SET abalance = abalance + {delta} "
            f"WHERE aid = {aid}")
    # Balance-check reads
    for tid in TIDS[:5]:
        add('Mixed', 'SELECT',
            f"SELECT tbalance FROM pgbench_tellers WHERE tid = {tid}")
    # Post-update inserts
    for i, aid in enumerate(AIDS[:5]):
        tid = TIDS[i % len(TIDS)]
        bid = BIDS[i % len(BIDS)]
        add('Mixed', 'INSERT',
            f"INSERT INTO pgbench_history(tid, bid, aid, delta, mtime) "
            f"VALUES ({tid}, {bid}, {aid}, 500, NOW())")

    # ── BulkRead: full-table scans, aggregates, analytical joins ──────────────
    for bid in BIDS:
        add('BulkRead', 'SELECT',
            f"SELECT COUNT(*), SUM(abalance), AVG(abalance), MAX(abalance) "
            f"FROM pgbench_accounts WHERE bid = {bid}")
    for lo, hi in RANGES:
        add('BulkRead', 'SELECT',
            f"SELECT COUNT(*), SUM(abalance) "
            f"FROM pgbench_accounts WHERE aid BETWEEN {lo} AND {hi}")
    # Full aggregate (heaviest — scans all 1M rows)
    for _ in range(3):
        add('BulkRead', 'SELECT',
            "SELECT bid, COUNT(*), SUM(abalance), AVG(abalance) "
            "FROM pgbench_accounts GROUP BY bid ORDER BY bid")
    # Top-N (sort 1M rows)
    for n in [10, 50, 100]:
        add('BulkRead', 'SELECT',
            f"SELECT aid, abalance FROM pgbench_accounts "
            f"ORDER BY abalance DESC LIMIT {n}")
    # Three-table join
    for bid in BIDS[:5]:
        add('BulkRead', 'SELECT',
            f"SELECT a.bid, COUNT(a.aid), SUM(a.abalance), b.bbalance "
            f"FROM pgbench_accounts a JOIN pgbench_branches b ON a.bid = b.bid "
            f"WHERE a.bid = {bid} GROUP BY a.bid, b.bbalance")

    return pool


# ── Argument Parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='HSM validation on OLTP (pgbench TPC-B schema).')
parser.add_argument('--execute', action='store_true')
parser.add_argument('--smoke',   action='store_true')
parser.add_argument('--port',    type=int, default=DOCKER_PORT)
parser.add_argument('--dbname',  default=DOCKER_DB)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--nreps',   type=int, default=N_REPS)
args = parser.parse_args()
N_REPS = args.nreps

print("=" * 70)
print("  HSM Validation — OLTP (pgbench TPC-B Schema)")
print("=" * 70)
mode = ("SMOKE" if args.smoke else
        "EXECUTE" if args.execute else "STATIC")
print(f"\n  Mode    : {mode}")
print(f"  DB      : {args.dbname} on port {args.port}")
print(f"  Phases  : ReadOnly / WriteHeavy / Mixed / BulkRead")
print(f"  N_REPS  : {N_REPS}  Seeds: {len(SEEDS)}")

# ── Build query pool ───────────────────────────────────────────────────────────
print("\n[1] Building query pool ...")
raw_pool = build_query_pool()

phase_counts = Counter(ph for _, ph, _, _ in raw_pool)
print(f"  Total queries : {len(raw_pool)}")
for ph in PHASE_ORDER:
    cnt = phase_counts.get(ph, 0)
    types = Counter(qt for _, p, qt, _ in raw_pool if p == ph)
    detail = "  ".join(f"{qt}={n}" for qt, n in sorted(types.items()))
    print(f"    {ph:<12}: {cnt:3d} queries  [{detail}]")

# ── Database connection ────────────────────────────────────────────────────────
conn = None
if args.execute or args.smoke:
    print(f"\n[2] Connecting to {args.dbname} ...")
    try:
        import psycopg2, psycopg2.extensions
        conn = psycopg2.connect(
            host=DOCKER_HOST, port=args.port, dbname=args.dbname,
            user=DOCKER_USER, password=DOCKER_PASS, connect_timeout=10,
            options=f"-c statement_timeout={TIMEOUT_MS}"
        )
        # IMPORTANT: autocommit=True so UPDATEs/INSERTs commit immediately
        conn.set_session(autocommit=True)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM pgbench_accounts")
        n_acc = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM pgbench_history")
        n_hist = cur.fetchone()[0]
        print(f"  Connected ✓  "
              f"(accounts: {n_acc:,}  history rows: {n_hist:,})")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)


def timed_exec(cursor, sql, n_reps=N_REPS):
    """Run query n_reps times; return median elapsed ms.
    Timeout → TIMEOUT_MS recorded (not excluded).
    First run is warm-up (discarded).
    Uses a fresh cursor per call to avoid state bleed from DML statements.
    Only calls fetchall() when cursor.description is not None (i.e. SELECT).
    """
    def _run_once():
        c = conn.cursor()
        try:
            c.execute(sql)
            if c.description is not None:   # SELECT returns rows; DML does not
                c.fetchall()
        finally:
            try:
                c.close()
            except Exception:
                pass

    # warm-up (discard result; ignore errors — e.g. first INSERT is fine)
    try:
        _run_once()
    except Exception:
        pass

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
            try:
                c.close()
            except Exception:
                pass
    return float(statistics.median(times))


# ── Smoke test ─────────────────────────────────────────────────────────────────
if args.smoke:
    print("\n[3] Smoke test — 3 queries per phase ...")
    for ph in PHASE_ORDER:
        sample = [(qid, qt, sql) for qid, p, qt, sql in raw_pool if p == ph][:3]
        times  = []
        for qid, qt, sql in sample:
            t = timed_exec(cur, sql, n_reps=1)
            times.append(t)
            if args.verbose:
                print(f"    {qid} ({qt:7s}) {t:8.2f} ms")
        med = statistics.median(times)
        print(f"  {ph:<12}: median={med:8.2f} ms")
    conn.close()
    print("\n  Smoke test complete.")
    print("  Run full experiment:")
    print("    python hsm_oltp_validation.py --execute")
    sys.exit(0)


# ── Execute all queries ────────────────────────────────────────────────────────
records = []
if args.execute:
    print(f"\n[3] Executing {len(raw_pool)} queries "
          f"(N_REPS={N_REPS}, timeout={TIMEOUT_MS//1000}s) ...")
    timeouts = 0
    for i, (qid, phase, qtype, sql) in enumerate(raw_pool):
        elapsed = timed_exec(cur, sql, N_REPS)
        if elapsed >= TIMEOUT_MS:
            timeouts += 1
        if args.verbose:
            flag = " TIMEOUT" if elapsed >= TIMEOUT_MS else ""
            print(f"  [{i+1:3d}/{len(raw_pool)}] "
                  f"{qid} ({phase:<12} {qtype:7s}) "
                  f"{elapsed:8.2f} ms{flag}")
        records.append({
            'qid': qid, 'phase': phase, 'qtype': qtype,
            'sql': sql, 'elapsed_ms': elapsed
        })
    conn.close()
    print(f"  Done. Timeouts: {timeouts}/{len(raw_pool)}")
else:
    # Static proxy: use known OLTP timing profiles
    proxy = {'ReadOnly': 1.0, 'WriteHeavy': 4.0, 'Mixed': 2.5, 'BulkRead': 200.0}
    for qid, phase, qtype, sql in raw_pool:
        records.append({
            'qid': qid, 'phase': phase, 'qtype': qtype,
            'sql': sql, 'elapsed_ms': proxy[phase]
        })
    print("\n[3] Static proxy mode (use --execute for real timing).")


# ── Per-phase timing summary ───────────────────────────────────────────────────
print("\n[4] Per-phase timing summary ...")
print(f"  {'Phase':<12}  {'N':>4}  {'Median ms':>10}  {'Min ms':>8}  {'Max ms':>8}")
print("  " + "-" * 50)
for ph in PHASE_ORDER:
    times = [r['elapsed_ms'] for r in records if r['phase'] == ph]
    if times:
        print(f"  {ph:<12}  {len(times):4d}  "
              f"{statistics.median(times):10.2f}  "
              f"{min(times):8.2f}  {max(times):8.2f}")

# Tables accessed per query (for S_A)
PHASE_TABLES = {
    'ReadOnly':   {'pgbench_accounts', 'pgbench_tellers',
                   'pgbench_branches'},
    'WriteHeavy': {'pgbench_accounts', 'pgbench_tellers',
                   'pgbench_branches', 'pgbench_history'},
    'Mixed':      {'pgbench_accounts', 'pgbench_tellers',
                   'pgbench_history'},
    'BulkRead':   {'pgbench_accounts', 'pgbench_branches'},
}
QTYPE_IDX = {'SELECT': 0, 'UPDATE': 1, 'INSERT': 2, 'DELETE': 3}

def extract_tables(sql):
    """Extract table names referenced in a query."""
    tables = set()
    for m in re.finditer(
            r'\b(pgbench_accounts|pgbench_tellers|pgbench_branches|pgbench_history)\b',
            sql, re.I):
        tables.add(m.group(1).lower())
    return tables


# ── Window features (same HSM framework as JOB) ───────────────────────────────
def compute_window_features(chunk):
    n       = len(chunk)
    elapsed = [r['elapsed_ms'] for r in chunk]
    total_s = sum(max(e, 0.01) for e in elapsed) / 1000.0
    qps     = n / total_s

    # S_R: SELECT ratio
    n_sel   = sum(1 for r in chunk if r['qtype'] == 'SELECT')
    ratio   = n_sel / n

    # S_T: operation-type vector [SELECT, UPDATE, INSERT, DELETE]
    type_vec = np.zeros(4)
    for r in chunk:
        idx = QTYPE_IDX.get(r['qtype'], 0)
        type_vec[idx] += 1

    # S_A: table + qtype access set
    access = set()
    for r in chunk:
        access.update(extract_tables(r['sql']))
        access.add(r['qtype'].lower())

    # S_P: DWT on per-query QPS series
    qps_series = np.array([1000.0 / max(e, 0.01) for e in elapsed])
    ts_len     = max(n, 8)
    qps_pad    = np.tile(qps_series, ts_len // n + 1)[:ts_len]

    return {
        'ratio'   : ratio,
        'qps'     : qps,
        'type_vec': type_vec,
        'access'  : access,
        'ts'      : qps_pad,
        'phase'   : Counter(r['phase'] for r in chunk).most_common(1)[0][0],
        'n'       : n,
    }

# ── HSM similarity functions ───────────────────────────────────────────────────
def s_r(a, b):
    return 1.0 - abs(a['ratio'] - b['ratio'])

def s_v(a, b):
    qa, qb = max(a['qps'], 1e-9), max(b['qps'], 1e-9)
    return math.exp(-abs(math.log(qa) - math.log(qb)))

def s_t(a, b):
    va, vb = a['type_vec'], b['type_vec']
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 1.0 if (na < 1e-9 and nb < 1e-9) else 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), 0.0, 1.0))

def s_a(a, b):
    aa, ab = a['access'], b['access']
    if not aa and not ab: return 1.0
    return len(aa & ab) / len(aa | ab)

def dwt_norm(ts):
    n   = len(ts)
    pad = 2 ** math.ceil(math.log2(max(n, 4)))
    tp  = np.pad(ts, (0, pad - n), mode='edge')
    ap  = pywt.dwt(tp, 'db2')[0]
    nm  = np.linalg.norm(ap)
    return ap / nm if nm > 1e-9 else np.zeros_like(ap)

def s_p(a, b):
    ta, tb  = a['ts'], b['ts']
    ml      = min(len(ta), len(tb))
    ha, hb  = dwt_norm(ta[:ml]), dwt_norm(tb[:ml])
    md      = min(len(ha), len(hb))
    return float(np.clip(1.0 - np.linalg.norm(ha[:md]-hb[:md]) / 2.0, 0.0, 1.0))

def hsm(a, b):
    sr, sv, st, sa, sp = s_r(a,b), s_v(a,b), s_t(a,b), s_a(a,b), s_p(a,b)
    return 0.2*sr + 0.2*sv + 0.2*st + 0.2*sa + 0.2*sp, \
           {'S_R':sr,'S_V':sv,'S_T':st,'S_A':sa,'S_P':sp}


# ── Phase-block trace + windows ────────────────────────────────────────────────
def build_trace(recs, seed):
    rng = random.Random(seed)
    trace = []
    for ph in PHASE_ORDER:
        members = [r for r in recs if r['phase'] == ph]
        rng.shuffle(members)
        trace.extend(members)
    return trace

def build_windows(trace, npts=NPTS, step=STEP):
    return [trace[i:i+npts]
            for i in range(0, len(trace)-npts+1, step)
            if len(trace[i:i+npts]) == npts]


# ── Multi-seed experiment ──────────────────────────────────────────────────────
print(f"\n[5] Running {len(SEEDS)} seeds ...")
all_results = []

for seed in SEEDS:
    trace   = build_trace(records, seed)
    windows = build_windows(trace)
    if len(windows) < 4:
        print(f"  Seed {seed}: only {len(windows)} windows — SKIP")
        continue

    feats = [compute_window_features(w) for w in windows]

    within_s, cross_s = [], []
    within_d = defaultdict(list)
    cross_d  = defaultdict(list)

    for i in range(len(feats)):
        for j in range(i+1, len(feats)):
            score, dims = hsm(feats[i], feats[j])
            is_w = (feats[i]['phase'] == feats[j]['phase'])
            if is_w:
                within_s.append(score)
                for k,v in dims.items(): within_d[k].append(v)
            else:
                cross_s.append(score)
                for k,v in dims.items(): cross_d[k].append(v)

    if not within_s or not cross_s:
        print(f"  Seed {seed}: no valid pairs — SKIP"); continue

    w_mean = statistics.mean(within_s)
    c_mean = statistics.mean(cross_s)
    dr     = w_mean / c_mean if c_mean > 0 else float('inf')
    n1, n2 = len(within_s), len(cross_s)
    u_stat, p_val = stats.mannwhitneyu(within_s, cross_s, alternative='greater')
    r_bis  = 1 - 2*u_stat/(n1*n2)

    rng_b  = random.Random(seed+1)
    bds    = sorted(statistics.mean(rng_b.choice(within_s) for _ in range(n1)) /
                    max(statistics.mean(rng_b.choice(cross_s) for _ in range(n2)), 1e-9)
                    for _ in range(2000))
    ci_lo  = bds[int(0.025*len(bds))]
    ci_hi  = bds[int(0.975*len(bds))]

    dim_delta = {d: (statistics.mean(within_d[d]) if within_d[d] else 0) -
                    (statistics.mean(cross_d[d])  if cross_d[d]  else 0)
                 for d in ['S_R','S_V','S_T','S_A','S_P']}
    dom = max(dim_delta, key=dim_delta.get)

    all_results.append({
        'seed':seed,'n_windows':len(feats),
        'n1':n1,'n2':n2,
        'w_mean':w_mean,'c_mean':c_mean,
        'DR':dr,'ci_lo':ci_lo,'ci_hi':ci_hi,
        'p_val':p_val,'r_bis':r_bis,
        'dom':dom,'dim_delta':dim_delta,
        'within_s':within_s,'cross_s':cross_s,
        'within_d':within_d,'cross_d':cross_d,
    })
    print(f"  Seed {seed:3d}: DR={dr:.3f} "
          f"(CI [{ci_lo:.3f},{ci_hi:.3f}])  "
          f"p={p_val:.3e}  dominant={dom}")

if not all_results:
    print("\n  ERROR: No valid results.")
    sys.exit(1)

# ── Aggregate ──────────────────────────────────────────────────────────────────
print("\n[6] Aggregating ...")
all_drs  = [r['DR']    for r in all_results]
all_ps   = [r['p_val'] for r in all_results]
all_rb   = [r['r_bis'] for r in all_results]
med_dr   = statistics.median(all_drs)
med_p    = statistics.median(all_ps)
med_rb   = statistics.median(all_rb)
med_ci_lo= statistics.median(r['ci_lo'] for r in all_results)
med_ci_hi= statistics.median(r['ci_hi'] for r in all_results)
emp_mean = statistics.mean(all_drs)
emp_sd   = statistics.stdev(all_drs) if len(all_drs) > 1 else 0.0
t_crit   = 2.262 if len(all_drs) == 10 else 2.0
emp_ci_lo= emp_mean - t_crit * emp_sd / math.sqrt(len(all_drs))
emp_ci_hi= emp_mean + t_crit * emp_sd / math.sqrt(len(all_drs))

dom_cnt  = Counter(r['dom'] for r in all_results)
final_dom= dom_cnt.most_common(1)[0][0]
ref      = all_results[0]

# ── Save results ───────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
mode_sfx = 'execute' if args.execute else 'static'
out_file = RESULTS_DIR / f'oltp_hsm_{mode_sfx}.csv'

with open(out_file, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['metric','value'])
    w.writerow(['design',        'oltp_pgbench_tpcb'])
    w.writerow(['mode',          mode_sfx])
    w.writerow(['n_queries',     len(records)])
    w.writerow(['npts',          NPTS])
    w.writerow(['n_seeds',       len(all_results)])
    w.writerow(['n_windows_ref', ref['n_windows']])
    w.writerow(['n_within_ref',  ref['n1']])
    w.writerow(['n_cross_ref',   ref['n2']])
    w.writerow(['within_mean',   round(ref['w_mean'], 6)])
    w.writerow(['cross_mean',    round(ref['c_mean'], 6)])
    w.writerow(['DR_median',     round(med_dr,    4)])
    w.writerow(['DR_mean',       round(emp_mean,  4)])
    w.writerow(['DR_sd',         round(emp_sd,    4)])
    w.writerow(['DR_CI_lo_emp',  round(emp_ci_lo, 4)])
    w.writerow(['DR_CI_hi_emp',  round(emp_ci_hi, 4)])
    w.writerow(['DR_CI_lo_boot', round(med_ci_lo, 4)])
    w.writerow(['DR_CI_hi_boot', round(med_ci_hi, 4)])
    w.writerow(['MWU_p_median',  f'{med_p:.3e}'])
    w.writerow(['r_biserial',    round(med_rb, 4)])
    w.writerow(['dominant_dim',  final_dom])
    for d in ['S_R','S_V','S_T','S_A','S_P']:
        wm = statistics.mean(ref['within_d'][d]) if ref['within_d'][d] else 0.0
        cm = statistics.mean(ref['cross_d'][d])  if ref['cross_d'][d]  else 0.0
        w.writerow([f'delta_{d}', round(wm-cm, 6)])
    for r in all_results:
        w.writerow([f'DR_seed{r["seed"]}', round(r['DR'], 4)])

print(f"  Results → {out_file}")

# ── Final summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  RESULTS: HSM OLTP Validation — pgbench TPC-B")
print("=" * 70)
print(f"\n  Phases    : ReadOnly / WriteHeavy / Mixed / BulkRead")
print(f"  Queries   : {len(records)} total")
print(f"  Seeds     : {len(all_results)}")
print(f"\n  ── Per-dimension (seed {ref['seed']}) ──")
print(f"  {'Dim':<6}  {'Within':>8}  {'Cross':>8}  {'Δ':>8}")
print("  " + "-" * 38)
for d in ['S_R','S_V','S_T','S_A','S_P']:
    wm = statistics.mean(ref['within_d'][d]) if ref['within_d'][d] else 0.0
    cm = statistics.mean(ref['cross_d'][d])  if ref['cross_d'][d]  else 0.0
    flag = " ← dominant" if d == final_dom else ""
    print(f"  {d:<6}  {wm:8.4f}  {cm:8.4f}  {wm-cm:+8.4f}{flag}")

print(f"\n  ── Aggregate ({len(all_results)} seeds) ──")
print(f"  DR median          : {med_dr:.3f}")
print(f"  DR mean ± SD       : {emp_mean:.3f} ± {emp_sd:.3f}")
print(f"  Empirical 95% CI   : [{emp_ci_lo:.3f}, {emp_ci_hi:.3f}]")
print(f"  Bootstrap 95% CI   : [{med_ci_lo:.3f}, {med_ci_hi:.3f}]")
print(f"  Mann-Whitney p     : {med_p:.3e}")
print(f"  |r_biserial|       : {abs(med_rb):.3f}")
print(f"  Dominant dim       : {final_dom}")
print(f"  DR range (seeds)   : {min(all_drs):.3f} – {max(all_drs):.3f}")

print(f"\n  ── Comparison ──")
print(f"  TPC-H  (A7)  : DR=1.445,  p<0.001")
print(f"  SDSS   (A8)  : DR=1.086,  p=1.6e-70")
print(f"  JOB    (A8b) : DR=1.814,  p=2.8e-85")
print(f"  OLTP   (A8c) : DR={med_dr:.3f},  p={med_p:.3e}")

sig = "✓ SIGNIFICANT" if med_p < 0.05 else "✗ NOT SIGNIFICANT"
print(f"\n  Assessment : {sig} (α=0.05)")
print("\n" + "=" * 70)
