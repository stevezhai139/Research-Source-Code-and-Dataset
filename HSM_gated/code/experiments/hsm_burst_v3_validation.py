"""
HSM Validation — Burst v3 (STEP=period alignment, S_P isolation)
=================================================================
Fixes the phase-misalignment bug in v2 by setting STEP = burst period.

ROOT CAUSE OF v2 FAILURE
--------------------------
  v2 used STEP=3 with burst period=2.  A sliding window of NPTS=6 starting
  at an even position sees [F,S,F,S,F,S], but starting at an odd position
  sees [S,F,S,F,S,F].  After unit-sphere DWT normalization these vectors
  point in almost opposite directions → cosine ≈ 0 → within-phase S_P
  similarity is ALSO low, not just cross-phase → Δ ≈ 0.

FIX: STEP = period
------------------
  With STEP=2 and period=2, every window starts at an even index:
    pos 0 → [F,S,F,S,F,S]   ← same DWT pattern
    pos 2 → [F,S,F,S,F,S]   ← same DWT pattern
    pos 4 → [F,S,F,S,F,S]   ← same DWT pattern
  All within-burst windows are IDENTICAL → S_P within ≈ 1.0
  Steady windows are all [5,5,5,5,5,5]  → S_P within ≈ 1.0
  Cross-phase: [5,5,5,5,5,5] vs [F,S,F,S,F,S] → clearly different DWT

PHASE DESIGN (2 phases, 30+30 = 60 queries per trace)
------------------------------------------------------
  Steady   (ST): 30 × pg_sleep(0.005) → [5,5,5,5,5,5] ms, flat
  Burst_Alt(BA): alternating pg_sleep(0.001)/pg_sleep(0.010), period-2
                 → [1,10,1,10,1,10] ms, oscillating

  NPTS=6, STEP=2  (STEP=period ensures alignment)

  With actual Docker overhead (~3 ms), actual timings:
    fast  (pg_sleep 1ms)   → ~4 ms
    steady(pg_sleep 5ms)   → ~8 ms
    slow  (pg_sleep 10ms)  → ~14 ms
  Both phases have mean ≈ 8-9 ms → S_V contrast ≈ 1.1×  → S_P must dominate

EXPECTED
---------
  S_P: dominant, Δ > 0.10
  S_V: near-zero (means equal)
  DR : 1.3–2.0
"""

import re, os, sys, csv, math, time, random, statistics, argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
import pywt

# ── Config ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / 'results' / 'burst_v3_validation'

DOCKER_HOST = os.environ.get('HSM_DOCKER_HOST', 'localhost')
DOCKER_PORT = int(os.environ.get('HSM_DOCKER_PORT', 5433))
DOCKER_USER = os.environ.get('HSM_DOCKER_USER', 'postgres')
DOCKER_PASS = os.environ.get('HSM_DOCKER_PASSWORD', 'postgres')
DOCKER_DB   = 'oltp'

TIMEOUT_MS  = 30_000
NPTS        = 6
STEP        = 2          # ← KEY FIX: STEP = burst period = 2
N_REPS      = 3
SEEDS       = [42, 137, 271, 314, 999, 7, 13, 55, 88, 101]
PHASE_ORDER = ['Steady', 'Burst_Alt']

T_FAST   = 0.001   #  ~4 ms actual
T_STEADY = 0.005   #  ~8 ms actual
T_SLOW   = 0.010   # ~14 ms actual

# ── Trace ──────────────────────────────────────────────────────────────────────
def build_trace(seed):
    _ = random.Random(seed)
    trace = []
    # Steady: 30 × pg_sleep(5ms)
    for i in range(30):
        trace.append((f"ST{i+1:03d}", 'Steady',    'SELECT', f"SELECT pg_sleep({T_STEADY})"))
    # Burst_Alt: strict alternating F,S,F,S,... (period-2)
    for i in range(15):
        trace.append((f"BA{2*i+1:03d}", 'Burst_Alt', 'SELECT', f"SELECT pg_sleep({T_FAST})"))
        trace.append((f"BA{2*i+2:03d}", 'Burst_Alt', 'SELECT', f"SELECT pg_sleep({T_SLOW})"))
    return trace  # 60 queries total

# ── timed_exec ─────────────────────────────────────────────────────────────────
def timed_exec(cursor, sql, n_reps=N_REPS):
    def _run():
        c = conn.cursor()
        try:
            c.execute(sql)
            if c.description is not None: c.fetchall()
        finally:
            try: c.close()
            except: pass
    try: _run()
    except: pass
    times = []
    for _ in range(n_reps):
        c = conn.cursor()
        try:
            t0 = time.perf_counter()
            c.execute(sql)
            if c.description is not None: c.fetchall()
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
def s_r(sqls):
    return sum(1 for s in sqls if re.match(r'\s*SELECT', s, re.I)) / len(sqls)

def s_v(times):
    return float(np.mean(times))

def s_t(sqls):
    c = [0,0,0,0]
    for s in sqls:
        u = s.strip().upper()
        if   u.startswith('SELECT'): c[0]+=1
        elif u.startswith('UPDATE'): c[1]+=1
        elif u.startswith('INSERT'): c[2]+=1
        elif u.startswith('DELETE'): c[3]+=1
    n=sum(c); return np.array([x/n for x in c]) if n else np.zeros(4)

def s_a(sqls):
    def tbls(sql):
        f=re.findall(r'(?:FROM|JOIN)\s+(\w+)',sql,re.I)
        return frozenset(t for t in f if t.lower()!='pg_sleep')
    sets=[tbls(s) for s in sqls]
    union=set().union(*sets)
    if not union: return 1.0
    inter=sets[0]
    for s in sets[1:]: inter=inter&s
    return len(inter)/len(union)

def s_p(times):
    arr=np.array(times,dtype=float)
    coeffs=pywt.wavedec(arr,'db2',level=1)
    vec=np.concatenate(coeffs)
    norm=np.linalg.norm(vec)
    return vec/norm if norm>1e-12 else vec

def hsm(w1s,w1t,w2s,w2t,W=None):
    if W is None: W=[.2,.2,.2,.2,.2]
    sr1,sr2=s_r(w1s),s_r(w2s)
    sv1,sv2=s_v(w1t),s_v(w2t)
    st1,st2=s_t(w1s),s_t(w2s)
    sa1,sa2=s_a(w1s),s_a(w2s)
    sp1,sp2=s_p(w1t),s_p(w2t)
    _sr=1-abs(sr1-sr2)
    _sv=1-abs(sv1-sv2)/(max(sv1,sv2)+1e-9)
    ct=np.dot(st1,st2)/(np.linalg.norm(st1)*np.linalg.norm(st2)+1e-12)
    _st=float(np.clip(ct,0,1))
    _sa=min(sa1,sa2)/(max(sa1,sa2)+1e-9)
    cp=float(np.dot(sp1,sp2))
    _sp=float(np.clip(cp,0,1))
    dims=[_sr,_sv,_st,_sa,_sp]
    return sum(w*d for w,d in zip(W,dims)), dims

def windows(exec_trace):
    wins=[]
    n=len(exec_trace)
    for s in range(0,n-NPTS+1,STEP):
        blk=exec_trace[s:s+NPTS]
        ph=Counter(b[1] for b in blk).most_common(1)[0][0]
        wins.append({'sqls':[b[2] for b in blk],
                     'times':[b[3] for b in blk],'phase':ph})
    return wins

def compute_dr(wins):
    pw,pc=[],[]
    for i in range(len(wins)):
        for j in range(i+1,len(wins)):
            sc,dims=hsm(wins[i]['sqls'],wins[i]['times'],
                        wins[j]['sqls'],wins[j]['times'])
            (pw if wins[i]['phase']==wins[j]['phase'] else pc).append((sc,dims))
    if not pw or not pc: return None,None,None,None
    ws=[e[0] for e in pw]; cs=[e[0] for e in pc]
    dr=statistics.median(ws)/(statistics.median(cs)+1e-12)
    u,p=stats.mannwhitneyu(ws,cs,alternative='greater')
    r=1-2*u/(len(ws)*len(cs))
    wd=np.mean([e[1] for e in pw],axis=0)
    cd=np.mean([e[1] for e in pc],axis=0)
    delta=wd-cd
    dom=['S_R','S_V','S_T','S_A','S_P'][int(np.argmax(delta))]
    return dr,p,r,{'n_within':len(pw),'n_cross':len(pc),
                   'w_dims':wd.tolist(),'c_dims':cd.tolist(),
                   'dim_deltas':delta.tolist(),'dominant':dom}

# ── CLI ────────────────────────────────────────────────────────────────────────
parser=argparse.ArgumentParser()
parser.add_argument('--execute',action='store_true')
parser.add_argument('--smoke',  action='store_true')
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--port',   type=int,default=DOCKER_PORT)
parser.add_argument('--dbname', default=DOCKER_DB)
args=parser.parse_args()

print("="*70)
print("  HSM Validation — Burst v3 (STEP=period, S_P Isolation)")
print("="*70)
sample=build_trace(42)
ph_cnt=Counter(p for _,p,_,_ in sample)
print(f"\n  Mode    : {'SMOKE' if args.smoke else 'EXECUTE' if args.execute else 'INFO'}")
print(f"  NPTS={NPTS}  STEP={STEP}  N_REPS={N_REPS}  Seeds={len(SEEDS)}")
print(f"  Phases  : Steady(5ms flat) / Burst_Alt(1↔10ms period-2)")
print(f"  KEY FIX : STEP=2 = burst period → every window sees same pattern")
print(f"\n[1] Trace: {len(sample)} queries")
for ph in PHASE_ORDER:
    print(f"    {ph}: {ph_cnt.get(ph,0)} queries  "
          f"windows≈{max(0,(ph_cnt.get(ph,0)-NPTS)//STEP+1)}")

conn=None
if args.execute or args.smoke:
    print(f"\n[2] Connecting ...")
    try:
        import psycopg2,psycopg2.extensions
        conn=psycopg2.connect(host=DOCKER_HOST,port=args.port,dbname=args.dbname,
                              user=DOCKER_USER,password=DOCKER_PASS,connect_timeout=10,
                              options=f"-c statement_timeout={TIMEOUT_MS}")
        conn.set_session(autocommit=True)
        cur=conn.cursor(); cur.execute("SELECT version()")
        print(f"  Connected ✓  ({cur.fetchone()[0].split()[1]})")
    except Exception as e:
        print(f"  ERROR: {e}"); sys.exit(1)

if args.smoke:
    print("\n[3] Timing calibration ...")
    for lbl,sql,exp in [("pg_sleep(0.001)",f"SELECT pg_sleep({T_FAST})",T_FAST*1000),
                        ("pg_sleep(0.005)",f"SELECT pg_sleep({T_STEADY})",T_STEADY*1000),
                        ("pg_sleep(0.010)",f"SELECT pg_sleep({T_SLOW})",T_SLOW*1000)]:
        t=timed_exec(cur,sql,n_reps=3)
        print(f"  {lbl}: expected={exp:.0f}ms  actual={t:.2f}ms  "
              f"{'✅' if abs(t-exp)<exp*2 else '⚠️'}")

    print("\n  Window alignment check (first 3 Burst windows) ...")
    tr=build_trace(42)
    et=[(qid,ph,sql,timed_exec(cur,sql,n_reps=1)) for qid,ph,_,sql in tr]
    ws=windows(et)
    burst_wins=[w for w in ws if w['phase']=='Burst_Alt'][:3]
    for i,w in enumerate(burst_wins):
        pattern='alt' if w['times'][0]<w['times'][1] else 'inv-alt'
        print(f"  Burst window {i}: {[f'{t:.1f}' for t in w['times']]} "
              f"→ {pattern} {'✅ aligned' if pattern=='alt' else '⚠️ phase-shifted'}")
    conn.close(); sys.exit(0)

if not args.execute:
    print("\nRun with --smoke or --execute"); sys.exit(0)

# ── Full run ───────────────────────────────────────────────────────────────────
print(f"\n[3] Calibration (seed=42) ...")
calib=build_trace(42)
ec=[(qid,ph,sql,timed_exec(cur,sql,N_REPS)) for qid,ph,_,sql in calib]
print(f"  Done.")
print(f"\n[4] Phase timing ...")
for ph in PHASE_ORDER:
    t=[ms for _,p,_,ms in ec if p==ph]
    print(f"  {ph:<12}: n={len(t)}  median={statistics.median(t):.2f}ms"
          f"  mean={statistics.mean(t):.2f}ms  "
          f"  range=[{min(t):.2f},{max(t):.2f}]")
means={ph:statistics.mean([ms for _,p,_,ms in ec if p==ph]) for ph in PHASE_ORDER}
sv_contrast=max(means.values())/(min(means.values())+1e-9)
print(f"\n  S_V contrast: {sv_contrast:.2f}×  "
      f"{'✅ low → S_P should dominate' if sv_contrast<2 else '⚠️ high'}")

print(f"\n[5] 10 seeds ...")
results=[]
for seed in SEEDS:
    tr=build_trace(seed)
    et=[(qid,ph,sql,timed_exec(cur,sql,N_REPS)) for qid,ph,_,sql in tr]
    ws=windows(et)
    dr,p,r,info=compute_dr(ws)
    if dr is None:
        print(f"  Seed {seed:3d}: insufficient pairs"); continue
    results.append({'seed':seed,'dr':dr,'p':p,'r':r,
                    'w_dims':info['w_dims'],'c_dims':info['c_dims'],
                    'dim_deltas':info['dim_deltas'],'dominant':info['dominant']})
    print(f"  Seed {seed:3d}: DR={dr:.3f}  p={p:.3e}  dominant={info['dominant']}")

conn.close()

# ── Aggregate & save ───────────────────────────────────────────────────────────
print(f"\n[6] Aggregating ...")
RESULTS_DIR.mkdir(parents=True,exist_ok=True)
csv_path=RESULTS_DIR/'burst_v3_hsm_execute.csv'

drs=[r['dr'] for r in results]; ps=[r['p'] for r in results]; n=len(drs)
dr_med=statistics.median(drs); dr_mean=statistics.mean(drs)
dr_sd=statistics.stdev(drs) if n>1 else 0
tc=stats.t.ppf(0.975,df=n-1) if n>1 else 0
ci_lo=dr_mean-tc*dr_sd/math.sqrt(n); ci_hi=dr_mean+tc*dr_sd/math.sqrt(n)
p_med=statistics.median(ps)

dim_names=['S_R','S_V','S_T','S_A','S_P']
avg_w=np.mean([r['w_dims'] for r in results],axis=0)
avg_c=np.mean([r['c_dims'] for r in results],axis=0)
avg_d=avg_w-avg_c
dom=dim_names[int(np.argmax(avg_d))]

with open(csv_path,'w',newline='') as f:
    w=csv.writer(f)
    for k,v in [('DR_median',f'{dr_med:.4f}'),('DR_mean',f'{dr_mean:.4f}'),
                ('DR_SD',f'{dr_sd:.4f}'),('DR_CI_lo',f'{ci_lo:.4f}'),
                ('DR_CI_hi',f'{ci_hi:.4f}'),('DR_range_lo',f'{min(drs):.4f}'),
                ('DR_range_hi',f'{max(drs):.4f}'),('MWU_p_median',f'{p_med:.4e}'),
                ('dominant_dim',dom),('NPTS',str(NPTS)),('STEP',str(STEP))]:
        w.writerow([k,v])
    for i,nm in enumerate(dim_names):
        w.writerow([f'delta_{nm}',f'{avg_d[i]:.4f}'])
    for r in results:
        w.writerow([f'DR_seed{r["seed"]}',f'{r["dr"]:.4f}'])
print(f"  Results → {csv_path}")

# ── Report ─────────────────────────────────────────────────────────────────────
print()
print("="*70)
print("  RESULTS: HSM Burst v3 — STEP=period Alignment Fix")
print("="*70)
print(f"\n  Phases : Steady(~8ms) / Burst_Alt(~4ms↔14ms, period-2)")
print(f"  NPTS={NPTS}  STEP={STEP}  Queries=60  Seeds={n}")
print(f"\n  ── Per-dimension ──")
print(f"  {'Dim':<6}{'Within':>8}{'Cross':>8}{'Δ':>8}")
print(f"  {'-'*32}")
for i,nm in enumerate(dim_names):
    mark=" ← dominant" if nm==dom else ""
    print(f"  {nm:<6}{avg_w[i]:>8.4f}{avg_c[i]:>8.4f}{avg_d[i]:>+8.4f}{mark}")
print(f"\n  DR median        : {dr_med:.3f}")
print(f"  DR mean ± SD     : {dr_mean:.3f} ± {dr_sd:.3f}")
print(f"  Empirical 95% CI : [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"  Mann-Whitney p   : {p_med:.3e}")
print(f"  Dominant dim     : {dom}")
print(f"  DR range         : {min(drs):.3f} – {max(drs):.3f}")

print(f"\n  ── All validations ──")
rows=[("TPC-H",  "A7",  1.445,"S_A","p<0.001"),
      ("SDSS",   "A8",  1.086,"S_V","p=1.6e-70"),
      ("JOB",    "A8b", 1.814,"S_T","p=2.8e-85"),
      ("OLTP",   "A8c", 1.543,"S_R","p=1.459e-65"),
      ("Burst v1","A8d",1.264,"S_V","p=8.3e-17 (real queries, timing variance)"),
      (f"Burst v3","A8d'",dr_med,dom,f"p={p_med:.3e} (pg_sleep, STEP=period)")]
for name,tag,dr,d,note in rows:
    print(f"  {name:<10} {tag:<5}: DR={dr:.3f}  dom={d:<4}  {note}")

print(f"\n  Assessment: {'✓ SIGNIFICANT' if p_med<0.05 else '✗ NOT SIGNIFICANT'} (α=0.05)")
print()
print("="*70)
