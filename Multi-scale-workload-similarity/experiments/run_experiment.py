"""
HSM Paper 3A — TPC-H Real Experiment (All-in-One)
===================================================
รันบนเครื่องของคุณ (MacBook Pro M4 + PostgreSQL 16)

สิ่งที่ script นี้ทำ:
  1. สร้าง TPC-H SF0.1 schema และ data จริงใน PostgreSQL
  2. รัน 4 workload phases (32 windows × ~15 queries)
  3. วัด execution time จริงทุก query
  4. คำนวณ HSM 5 dimensions จาก trace จริง
  5. Export ผลลัพธ์เป็น CSV 3 ไฟล์

ใช้งาน:
  pip install psycopg2-binary pandas numpy scipy
  python run_experiment.py

แล้ว upload ไฟล์ผลลัพธ์ให้ Claude:
  - tpch_trace.csv
  - tpch_hsm.csv
  - tpch_stats.csv
"""

import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from scipy.stats import spearmanr  # kept for reference; sr() now uses angular distance
from collections import Counter
import time, os, random, math, sys

# ══════════════════════════════════════════════════════
#  CONFIG — แก้ตรงนี้ถ้า PostgreSQL ใช้ password
# ══════════════════════════════════════════════════════
PG = dict(host="localhost", port=5432,
          user="postgres", password="", dbname="postgres")
TARGET_DB  = "tpch_exp"
SCALE      = 0.05        # SF0.05 = ~300K lineitem rows (เร็ว ~2 นาที)
                         # เปลี่ยนเป็น 0.1 หรือ 1.0 ถ้าต้องการข้อมูลมากขึ้น
N_WINDOWS  = 32          # 4 phases × 8 windows
QUERIES_PW = 15          # queries ต่อ window
SEED       = 2024
# ══════════════════════════════════════════════════════

random.seed(SEED)
np.random.seed(SEED)

def connect(db=None):
    cfg = {**PG, "dbname": db or PG["dbname"]}
    return psycopg2.connect(**cfg)

# ─── STEP 1: Create Database ──────────────────────────
print("="*55)
print("TPC-H Experiment — PostgreSQL 16 (Homebrew)")
print("="*55)
print(f"\n[1] Creating database '{TARGET_DB}'...")
try:
    conn = connect(); conn.autocommit = True
    cur  = conn.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS {TARGET_DB}")
    cur.execute(f"CREATE DATABASE {TARGET_DB}")
    conn.close()
    print(f"    ✓ '{TARGET_DB}' created")
except Exception as e:
    print(f"    ✗ {e}"); sys.exit(1)

conn = connect(TARGET_DB); conn.autocommit = True
cur  = conn.cursor()

# ─── STEP 2: Create Schema ────────────────────────────
print("[2] Creating TPC-H schema...")
cur.execute("""
CREATE TABLE region(
  r_regionkey INTEGER PRIMARY KEY,
  r_name      CHAR(25), r_comment VARCHAR(152));

CREATE TABLE nation(
  n_nationkey INTEGER PRIMARY KEY,
  n_name      CHAR(25), n_regionkey INTEGER, n_comment VARCHAR(152));

CREATE TABLE supplier(
  s_suppkey   INTEGER PRIMARY KEY,
  s_name      CHAR(25), s_address VARCHAR(40),
  s_nationkey INTEGER, s_phone CHAR(15),
  s_acctbal   NUMERIC(15,2), s_comment VARCHAR(101));

CREATE TABLE customer(
  c_custkey    BIGINT PRIMARY KEY,
  c_name       VARCHAR(25), c_address VARCHAR(40),
  c_nationkey  INTEGER, c_phone CHAR(15),
  c_acctbal    NUMERIC(15,2), c_mktsegment CHAR(10),
  c_comment    VARCHAR(117));

CREATE TABLE part(
  p_partkey     BIGINT PRIMARY KEY,
  p_name        VARCHAR(55), p_mfgr CHAR(25),
  p_brand       CHAR(10), p_type VARCHAR(25),
  p_size        INTEGER, p_container CHAR(10),
  p_retailprice NUMERIC(15,2), p_comment VARCHAR(23));

CREATE TABLE orders(
  o_orderkey      BIGINT PRIMARY KEY,
  o_custkey       BIGINT, o_orderstatus CHAR(1),
  o_totalprice    NUMERIC(15,2), o_orderdate DATE,
  o_orderpriority CHAR(15), o_clerk CHAR(15),
  o_shippriority  INTEGER, o_comment VARCHAR(79));

CREATE TABLE partsupp(
  ps_partkey    BIGINT, ps_suppkey INTEGER,
  ps_availqty   INTEGER, ps_supplycost NUMERIC(15,2),
  ps_comment    VARCHAR(199),
  PRIMARY KEY(ps_partkey, ps_suppkey));

CREATE TABLE lineitem(
  l_orderkey      BIGINT, l_partkey BIGINT,
  l_suppkey       INTEGER, l_linenumber INTEGER,
  l_quantity      NUMERIC(15,2), l_extendedprice NUMERIC(15,2),
  l_discount      NUMERIC(15,2), l_tax NUMERIC(15,2),
  l_returnflag    CHAR(1), l_linestatus CHAR(1),
  l_shipdate      DATE, l_commitdate DATE, l_receiptdate DATE,
  l_shipinstruct  CHAR(25), l_shipmode CHAR(10),
  l_comment       VARCHAR(44),
  PRIMARY KEY(l_orderkey, l_linenumber));
""")
print("    ✓ Schema created (8 tables)")

# ─── STEP 3: Generate & Load Data ────────────────────
print(f"[3] Generating TPC-H SF{SCALE} data...")

REGIONS    = [(i, n, "comment") for i,n in enumerate(
               ["AFRICA","AMERICA","ASIA","EUROPE","MIDDLE EAST"])]
NATIONS    = [(i,n,r,"c") for i,(n,r) in enumerate([
               ("ALGERIA",0),("ARGENTINA",1),("BRAZIL",1),("CANADA",1),
               ("EGYPT",4),("ETHIOPIA",0),("FRANCE",3),("GERMANY",3),
               ("INDIA",2),("INDONESIA",2),("IRAN",4),("IRAQ",4),
               ("JAPAN",2),("JORDAN",4),("KENYA",0),("MOROCCO",0),
               ("MOZAMBIQUE",0),("PERU",1),("CHINA",2),("ROMANIA",3),
               ("SAUDI ARABIA",4),("VIETNAM",2),("RUSSIA",3),
               ("UNITED KINGDOM",3),("UNITED STATES",1)])]

N_SUPP     = max(1, int(10000 * SCALE))
N_CUST     = max(1, int(150000 * SCALE))
N_PART     = max(1, int(200000 * SCALE))
N_ORDER    = max(1, int(1500000 * SCALE))
N_LINEITEM = max(1, int(6000000 * SCALE))

MKTSEGS  = ['AUTOMOBILE','BUILDING','FURNITURE','HOUSEHOLD','MACHINERY']
OPRIOS   = ['1-URGENT','2-HIGH','3-MEDIUM','4-NOT SPECIFIED','5-LOW']
SHIPMODES= ['AIR','FOB','MAIL','RAIL','REG AIR','SHIP','TRUCK']
BRANDS   = [f'Brand#{i}{j}' for i in range(1,6) for j in range(1,6)]
PTYPES   = ['STANDARD ANODIZED TIN','LARGE BRUSHED BRASS','SMALL POLISHED NICKEL',
            'PROMO BURNISHED STEEL','ECONOMY PLATED COPPER']
PCONT    = ['SM BOX','MED BOX','LG BOX','SM PACK','MED PACK','LG PACK',
            'SM CASE','MED CASE','LG CASE','WRAP CASE','JUMBO CASE','MED BAG']

def rstr(n): return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ',k=n))
def rdate(y0=1992,y1=1998):
    d = random.randint(0,365*6);
    from datetime import date, timedelta
    return date(y0,1,1)+timedelta(days=d)

# Load region & nation (tiny)
cur.executemany("INSERT INTO region VALUES(%s,%s,%s)", REGIONS)
cur.executemany("INSERT INTO nation VALUES(%s,%s,%s,%s)", NATIONS)

# Suppliers
print(f"    Loading {N_SUPP:,} suppliers...", end=' ', flush=True)
data = [(i+1, f'Supplier#{i+1:09d}', rstr(25),
         random.randint(0,24), f'{random.randint(10,99)}-{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}',
         round(random.uniform(-999,9999),2), rstr(50))
        for i in range(N_SUPP)]
psycopg2.extras.execute_values(cur,
    "INSERT INTO supplier VALUES %s", data, page_size=1000)
print("✓")

# Customers
print(f"    Loading {N_CUST:,} customers...", end=' ', flush=True)
data = [(i+1, f'Customer#{i+1:09d}', rstr(25),
         random.randint(0,24), f'{random.randint(10,99)}-{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}',
         round(random.uniform(-999,9999),2),
         random.choice(MKTSEGS), rstr(50))
        for i in range(N_CUST)]
psycopg2.extras.execute_values(cur,
    "INSERT INTO customer VALUES %s", data, page_size=1000)
print("✓")

# Parts
print(f"    Loading {N_PART:,} parts...", end=' ', flush=True)
data = [(i+1, rstr(20), f'Manufacturer#{random.randint(1,5)}',
         random.choice(BRANDS), random.choice(PTYPES),
         random.randint(1,50), random.choice(PCONT),
         round(900+i*0.01, 2), rstr(20))
        for i in range(N_PART)]
psycopg2.extras.execute_values(cur,
    "INSERT INTO part VALUES %s", data, page_size=1000)
print("✓")

# PartSupp
print(f"    Loading partsupp...", end=' ', flush=True)
ps_data = []
for p in range(1, N_PART+1):
    for s_off in range(4):
        s = (p + s_off * (N_SUPP // 4 + 1)) % N_SUPP + 1
        ps_data.append((p, s, random.randint(1,9999),
                        round(random.uniform(1,1000),2), rstr(50)))
psycopg2.extras.execute_values(cur,
    "INSERT INTO partsupp VALUES %s", ps_data, page_size=2000)
del ps_data
print("✓")

# Orders + Lineitems
print(f"    Loading {N_ORDER:,} orders + lineitems...", end=' ', flush=True)
ord_data, li_data = [], []
li_count = 0
for o in range(1, N_ORDER+1):
    cust = random.randint(1, N_CUST)
    odate = rdate()
    ord_data.append((o, cust, 'O' if random.random()<0.5 else 'F',
                     round(random.uniform(1000,200000),2),
                     odate, random.choice(OPRIOS),
                     f'Clerk#{random.randint(1,1000):09d}',
                     0, rstr(30)))
    n_lines = random.randint(1,7)
    for ln in range(1, n_lines+1):
        p = random.randint(1, N_PART)
        s = random.randint(1, N_SUPP)
        qty = random.randint(1,50)
        ep  = round(qty * random.uniform(10,100), 2)
        sdate = rdate(1992, 1998)
        li_data.append((o, p, s, ln, qty, ep,
                        round(random.uniform(0,0.1),2),
                        round(random.uniform(0,0.08),2),
                        random.choice(['A','N','R']),
                        random.choice(['O','F']),
                        sdate, sdate, sdate,
                        random.choice(['DELIVER IN PERSON','COLLECT COD','NONE','TAKE BACK RETURN']),
                        random.choice(SHIPMODES), rstr(20)))
        li_count += 1

    # Batch insert every 5000 orders
    if len(ord_data) >= 5000:
        psycopg2.extras.execute_values(cur,
            "INSERT INTO orders VALUES %s", ord_data, page_size=2000)
        psycopg2.extras.execute_values(cur,
            "INSERT INTO lineitem VALUES %s", li_data, page_size=2000)
        ord_data, li_data = [], []

if ord_data:
    psycopg2.extras.execute_values(cur,
        "INSERT INTO orders VALUES %s", ord_data, page_size=2000)
    psycopg2.extras.execute_values(cur,
        "INSERT INTO lineitem VALUES %s", li_data, page_size=2000)
print(f"✓  ({li_count:,} lineitems)")

# Basic indexes for query performance
print("    Creating indexes...", end=' ', flush=True)
for ddl in [
    "CREATE INDEX ON lineitem(l_shipdate)",
    "CREATE INDEX ON lineitem(l_orderkey)",
    "CREATE INDEX ON orders(o_orderdate)",
    "CREATE INDEX ON orders(o_custkey)",
    "CREATE INDEX ON customer(c_nationkey)",
    "CREATE INDEX ON supplier(s_nationkey)",
]:
    cur.execute(ddl)
print("✓")

# Verify
cur.execute("SELECT COUNT(*) FROM lineitem")
n_li = cur.fetchone()[0]
print(f"\n    ✓ TPC-H data loaded: {n_li:,} lineitem rows\n")

# ─── STEP 4: Define Queries & Phases ─────────────────
QUERIES = {
"Q1": """SELECT l_returnflag, l_linestatus,
  SUM(l_quantity), SUM(l_extendedprice),
  SUM(l_extendedprice*(1-l_discount)), COUNT(*)
FROM lineitem WHERE l_shipdate <= '1998-09-02'
GROUP BY l_returnflag, l_linestatus ORDER BY 1,2""",

"Q3": """SELECT l_orderkey,
  SUM(l_extendedprice*(1-l_discount)) AS rev,
  o_orderdate, o_shippriority
FROM customer JOIN orders ON c_custkey=o_custkey
JOIN lineitem ON l_orderkey=o_orderkey
WHERE c_mktsegment='BUILDING'
  AND o_orderdate<'1995-03-15' AND l_shipdate>'1995-03-15'
GROUP BY l_orderkey,o_orderdate,o_shippriority
ORDER BY rev DESC LIMIT 10""",

"Q4": """SELECT o_orderpriority, COUNT(*) AS cnt
FROM orders WHERE o_orderdate>='1993-07-01'
  AND o_orderdate<'1993-10-01'
  AND EXISTS(SELECT 1 FROM lineitem
             WHERE l_orderkey=o_orderkey
               AND l_commitdate<l_receiptdate)
GROUP BY o_orderpriority ORDER BY 1""",

"Q5": """SELECT n_name,
  SUM(l_extendedprice*(1-l_discount)) AS rev
FROM customer JOIN orders ON c_custkey=o_custkey
JOIN lineitem ON l_orderkey=o_orderkey
JOIN supplier ON l_suppkey=s_suppkey AND s_nationkey=c_nationkey
JOIN nation ON n_nationkey=s_nationkey
JOIN region ON r_regionkey=n_regionkey
WHERE r_name='ASIA'
  AND o_orderdate>='1994-01-01' AND o_orderdate<'1995-01-01'
GROUP BY n_name ORDER BY rev DESC""",

"Q6": """SELECT SUM(l_extendedprice*l_discount) AS rev
FROM lineitem
WHERE l_shipdate>='1994-01-01' AND l_shipdate<'1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity<24""",

"Q7": """SELECT n1.n_name, n2.n_name,
  EXTRACT(year FROM l_shipdate),
  SUM(l_extendedprice*(1-l_discount))
FROM supplier JOIN lineitem ON s_suppkey=l_suppkey
JOIN orders ON o_orderkey=l_orderkey
JOIN customer ON c_custkey=o_custkey
JOIN nation n1 ON s_nationkey=n1.n_nationkey
JOIN nation n2 ON c_nationkey=n2.n_nationkey
WHERE ((n1.n_name='FRANCE' AND n2.n_name='GERMANY')
    OR (n1.n_name='GERMANY' AND n2.n_name='FRANCE'))
  AND l_shipdate BETWEEN '1995-01-01' AND '1996-12-31'
GROUP BY 1,2,3 ORDER BY 1,2,3""",

"Q10": """SELECT c_custkey, c_name,
  SUM(l_extendedprice*(1-l_discount)), c_acctbal, n_name
FROM customer JOIN orders ON c_custkey=o_custkey
JOIN lineitem ON l_orderkey=o_orderkey
JOIN nation ON c_nationkey=n_nationkey
WHERE o_orderdate>='1993-10-01' AND o_orderdate<'1994-01-01'
  AND l_returnflag='R'
GROUP BY c_custkey,c_name,c_acctbal,c_phone,n_name,c_address,c_comment
ORDER BY 3 DESC LIMIT 20""",

"Q11": """SELECT ps_partkey,
  SUM(ps_supplycost*ps_availqty) AS val
FROM partsupp JOIN supplier ON ps_suppkey=s_suppkey
JOIN nation ON s_nationkey=n_nationkey
WHERE n_name='GERMANY'
GROUP BY ps_partkey
HAVING SUM(ps_supplycost*ps_availqty)>(
  SELECT SUM(ps_supplycost*ps_availqty)*0.0001
  FROM partsupp JOIN supplier ON ps_suppkey=s_suppkey
  JOIN nation ON s_nationkey=n_nationkey
  WHERE n_name='GERMANY')
ORDER BY val DESC LIMIT 20""",

"Q12": """SELECT l_shipmode,
  SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END),
  SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END)
FROM orders JOIN lineitem ON o_orderkey=l_orderkey
WHERE l_shipmode IN ('MAIL','SHIP')
  AND l_commitdate<l_receiptdate AND l_shipdate<l_commitdate
  AND l_receiptdate>='1994-01-01' AND l_receiptdate<'1995-01-01'
GROUP BY l_shipmode ORDER BY l_shipmode""",

"Q14": """SELECT 100.0*SUM(CASE WHEN p_type LIKE 'PROMO%'
  THEN l_extendedprice*(1-l_discount) ELSE 0 END)
  /SUM(l_extendedprice*(1-l_discount))
FROM lineitem JOIN part ON l_partkey=p_partkey
WHERE l_shipdate>='1995-09-01' AND l_shipdate<'1995-10-01'""",

"Q17": """SELECT SUM(l_extendedprice)/7.0
FROM lineitem JOIN part ON p_partkey=l_partkey
WHERE p_brand='Brand#23' AND p_container='MED BOX'
  AND l_quantity<(SELECT 0.2*AVG(l_quantity)
                  FROM lineitem WHERE l_partkey=p_partkey)""",

"Q18": """SELECT c_name,c_custkey,o_orderkey,o_orderdate,
  o_totalprice, SUM(l_quantity)
FROM customer JOIN orders ON c_custkey=o_custkey
JOIN lineitem ON o_orderkey=l_orderkey
WHERE o_orderkey IN(
  SELECT l_orderkey FROM lineitem
  GROUP BY l_orderkey HAVING SUM(l_quantity)>300)
GROUP BY 1,2,3,4,5 ORDER BY o_totalprice DESC LIMIT 10""",
}

QUERY_COLS = {
"Q1":  {"lineitem.l_shipdate","lineitem.l_returnflag","lineitem.l_linestatus",
        "lineitem.l_quantity","lineitem.l_extendedprice","lineitem.l_discount","lineitem.l_tax"},
"Q3":  {"customer.c_mktsegment","customer.c_custkey","orders.o_custkey",
        "orders.o_orderdate","orders.o_shippriority","lineitem.l_orderkey",
        "lineitem.l_shipdate","lineitem.l_extendedprice","lineitem.l_discount"},
"Q4":  {"orders.o_orderdate","orders.o_orderpriority","lineitem.l_commitdate","lineitem.l_receiptdate"},
"Q5":  {"customer.c_custkey","customer.c_nationkey","orders.o_orderdate",
        "lineitem.l_extendedprice","lineitem.l_discount","lineitem.l_suppkey",
        "supplier.s_nationkey","nation.n_name","region.r_name"},
"Q6":  {"lineitem.l_shipdate","lineitem.l_discount","lineitem.l_quantity","lineitem.l_extendedprice"},
"Q7":  {"supplier.s_nationkey","lineitem.l_shipdate","lineitem.l_extendedprice",
        "lineitem.l_discount","orders.o_custkey","customer.c_nationkey","nation.n_name"},
"Q10": {"customer.c_custkey","customer.c_name","customer.c_acctbal","customer.c_nationkey",
        "orders.o_orderdate","lineitem.l_returnflag","lineitem.l_extendedprice",
        "lineitem.l_discount","nation.n_name"},
"Q11": {"partsupp.ps_suppkey","partsupp.ps_supplycost","partsupp.ps_availqty",
        "supplier.s_nationkey","nation.n_name"},
"Q12": {"lineitem.l_shipmode","lineitem.l_commitdate","lineitem.l_receiptdate",
        "lineitem.l_shipdate","orders.o_orderpriority"},
"Q14": {"lineitem.l_partkey","lineitem.l_shipdate","lineitem.l_extendedprice",
        "lineitem.l_discount","part.p_type"},
"Q17": {"lineitem.l_partkey","lineitem.l_quantity","lineitem.l_extendedprice",
        "part.p_brand","part.p_container"},
"Q18": {"customer.c_name","customer.c_custkey","orders.o_orderkey",
        "orders.o_orderdate","orders.o_totalprice","lineitem.l_quantity"},
}

# 4 Phases — query sets are DISTINCT (S_A จะต่างกันชัดที่ phase boundary)
PHASES = [
  {"name":"Reporting",    "qs":["Q1","Q6","Q14","Q12","Q4"],    "w":[.35,.25,.20,.10,.10], "n":15},
  {"name":"Join-Heavy",   "qs":["Q3","Q5","Q7","Q10","Q18"],    "w":[.30,.25,.20,.15,.10], "n":20},
  {"name":"Aggregation",  "qs":["Q11","Q17","Q1","Q6","Q14"],   "w":[.35,.25,.20,.10,.10], "n":12},
  {"name":"Multi-Join",   "qs":["Q7","Q5","Q3","Q11","Q17"],    "w":[.30,.25,.20,.15,.10], "n":18},
]
WIN_PER_PHASE = N_WINDOWS // len(PHASES)  # 8

# ─── STEP 5: Run Workload ─────────────────────────────
print("[4] Running real TPC-H workload on PostgreSQL...")
print(f"    {N_WINDOWS} windows × {QUERIES_PW} queries = ~{N_WINDOWS*QUERIES_PW} total queries\n")

trace = []
for win in range(N_WINDOWS):
    ph   = PHASES[win // WIN_PER_PHASE]
    wts  = np.array(ph["w"]) / sum(ph["w"])
    n_q  = ph["n"]
    qnames = np.random.choice(ph["qs"], size=n_q, p=wts, replace=True).tolist()

    win_ms = 0.0
    for seq, qname in enumerate(qnames):
        t0 = time.perf_counter()
        try:
            cur.execute(QUERIES[qname])
            rows = cur.fetchall()
            n_rows = len(rows)
            ok = True
        except Exception as e:
            conn.rollback()
            n_rows, ok = 0, False
        ms = (time.perf_counter() - t0) * 1000
        win_ms += ms
        trace.append({"window":win, "phase":ph["name"], "seq":seq,
                      "query":qname, "exec_ms":round(ms,3),
                      "n_rows":n_rows, "ok":ok})

    avg_ms = win_ms / n_q
    print(f"    W{win:02d} [{ph['name']:11s}] {n_q:2d}q  "
          f"total={win_ms:6.0f}ms  avg={avg_ms:5.0f}ms/q")

conn.close()

df = pd.DataFrame(trace)
ok_df = df[df["ok"]]
print(f"\n    ✓ {len(ok_df):,} successful queries "
      f"({(df.ok==False).sum()} errors)")
df.to_csv("tpch_trace.csv", index=False)
print(f"    ✓ Saved: tpch_trace.csv")

# ─── STEP 6: Compute Real HSM ────────────────────────
print("\n[5] Computing HSM scores...")

ALL_Q = sorted(df["query"].unique())

def sr(w1, w2):
    """S_R: query-mix similarity via angular distance on frequency vectors.

    v28 fix: Spearman rank-correlation distance (1-r)/2 is NOT a proven metric
    (triangle inequality fails in general). Angular distance arccos(cosine)/π
    IS a metric on the unit sphere (Deza & Deza, 2009).
    """
    c1 = Counter(w1["query"]); c2 = Counter(w2["query"])
    v1 = np.array([c1.get(q, 0) / max(len(w1), 1) for q in ALL_Q])
    v2 = np.array([c2.get(q, 0) / max(len(w2), 1) for q in ALL_Q])
    d1, d2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if d1 > 0 and d2 > 0:
        cos_val = np.clip(np.dot(v1, v2) / (d1 * d2), -1.0, 1.0)
        ang_dist = np.arccos(cos_val) / np.pi  # ∈ [0,1], IS a metric
        return float(1.0 - ang_dist)
    return 1.0 if (d1 < 1e-9 and d2 < 1e-9) else 0.5

def sv(w1, w2):
    n1, n2 = len(w1), len(w2)
    return min(n1,n2)/max(n1,n2) if max(n1,n2) > 0 else 1.0

def st(w1, w2, K=8):
    """S_T: temporal-pattern similarity via angular distance on timing histograms.

    v28 fix: cosine similarity gives d_T = 1 - cos(u,v), which violates the
    triangle inequality and is NOT a metric. Angular distance arccos(cos)/π
    IS a metric on the unit sphere.
    """
    def h(w):
        s = w["seq"].values; mx = max(s.max(), 1)
        idx = (s / mx * K).clip(0, K - 1).astype(int)
        v = np.bincount(idx, minlength=K).astype(float)
        return v / (v.sum() + 1e-9)
    h1, h2 = h(w1), h(w2)
    d1, d2 = np.linalg.norm(h1), np.linalg.norm(h2)
    if d1 > 0 and d2 > 0:
        cos_val = np.clip(np.dot(h1, h2) / (d1 * d2), -1.0, 1.0)
        ang_dist = np.arccos(cos_val) / np.pi  # ∈ [0,1], IS a metric
        return float(1.0 - ang_dist)
    return 1.0

def sa(w1, w2):
    def cols(w):
        s=set()
        for q in w["query"]: s|=QUERY_COLS.get(q,set())
        return s
    a1,a2=cols(w1),cols(w2); u=a1|a2
    return len(a1&a2)/len(u) if u else 1.0

def sp(w1, w2):
    t1=set(w1["query"]); t2=set(w2["query"]); u=t1|t2
    return len(t1&t2)/len(u) if u else 1.0

wins = sorted(df["window"].unique())
hsm_rows = []
for i in range(len(wins)-1):
    w1 = ok_df[ok_df["window"]==wins[i]]
    w2 = ok_df[ok_df["window"]==wins[i+1]]
    p1 = w1["phase"].iloc[0]; p2 = w2["phase"].iloc[0]
    cross = p1 != p2
    sR,sV,sT,sA,sP = sr(w1,w2),sv(w1,w2),st(w1,w2),sa(w1,w2),sp(w1,w2)
    hsm = .25*sR + .20*sV + .20*sT + .20*sA + .15*sP
    hsm_rows.append({
        "w_from":wins[i],"w_to":wins[i+1],
        "phase_from":p1,"phase_to":p2,"cross_phase":cross,
        "n1":len(w1),"n2":len(w2),
        "avg_ms_w1":round(w1["exec_ms"].mean(),2),
        "avg_ms_w2":round(w2["exec_ms"].mean(),2),
        "S_R":round(sR,4),"S_V":round(sV,4),"S_T":round(sT,4),
        "S_A":round(sA,4),"S_P":round(sP,4),"HSM":round(hsm,4),
    })
    flag = "*** TRANSITION ***" if cross else ""
    print(f"    W{wins[i]:02d}→W{wins[i+1]:02d} [{p1[:8]:8s}→{p2[:8]:8s}] "
          f"HSM={hsm:.3f} S_R={sR:.3f} S_V={sV:.3f} "
          f"S_A={sA:.3f} S_T={sT:.3f}  {flag}")

hdf = pd.DataFrame(hsm_rows)
hdf.to_csv("tpch_hsm.csv", index=False)

# Window statistics
wdf = ok_df.groupby(["window","phase"]).agg(
    n_queries=("query","count"), avg_ms=("exec_ms","mean"),
    total_ms=("exec_ms","sum"), n_qtypes=("query","nunique")
).reset_index()
wdf.to_csv("tpch_stats.csv", index=False)

# ─── STEP 7: Print Summary ────────────────────────────
within = hdf[~hdf["cross_phase"]]
across = hdf[hdf["cross_phase"]]
print("\n" + "="*55)
print("RESULTS SUMMARY")
print("="*55)
print(f"  Total queries run (real PostgreSQL): {len(ok_df):,}")
print(f"  Execution time range: {ok_df.exec_ms.min():.0f}–{ok_df.exec_ms.max():.0f} ms")
print(f"  Within-phase  HSM: {within.HSM.mean():.4f}  (std={within.HSM.std():.4f})")
print(f"  Cross-phase   HSM: {across.HSM.mean():.4f}  (std={across.HSM.std():.4f})")
print(f"  Discrimination:    {within.HSM.mean()/max(across.HSM.mean(),0.001):.2f}×")
print(f"  S_A within:  {within.S_A.mean():.4f}    S_A across: {across.S_A.mean():.4f}")
print(f"\n  Min HSM: {hdf.HSM.min():.3f}  at W{hdf.loc[hdf.HSM.idxmin(),'w_from']}→W{hdf.loc[hdf.HSM.idxmin(),'w_to']}")
print(f"  Max HSM: {hdf.HSM.max():.3f}  at W{hdf.loc[hdf.HSM.idxmax(),'w_from']}→W{hdf.loc[hdf.HSM.idxmax(),'w_to']}")

print(f"""
{'='*55}
ไฟล์ที่ต้อง upload ให้ Claude (ลาก & วางใน chat):
  1. tpch_trace.csv   — raw query trace
  2. tpch_hsm.csv     — HSM scores
  3. tpch_stats.csv   — window statistics
{'='*55}
""")