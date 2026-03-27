"""
HSM Paper 3A — Priority 3: Scale Experiment
============================================
Measures ACTUAL index rebuild time T_A(N) and HSM gating savings
at configurable TPC-H scale factors (SF=0.2, 1, 3, 10).

Usage
─────
  # Single scale factor, 5 runs (recommended for paper):
  python run_scale_experiment.py --scale 1.0 --runs 5

  # Multiple scale factors (run sequentially):
  python run_scale_experiment.py --scale 1.0 3.0 10.0 --runs 5

  # Quick smoke-test:
  python run_scale_experiment.py --scale 0.05

  # Verbose mode (shows per-window detail):
  python run_scale_experiment.py --scale 1.0 --runs 5 --verbose

Estimated runtimes per run (M4 MacBook Pro)
────────────────────────────────────────────
  SF=0.05  →  ~2 min/run    (smoke-test)
  SF=0.2   →  ~8 min/run    (main paper experiments — already done)
  SF=1.0   →  ~25 min/run   × 5 runs ≈ 2 hr
  SF=3.0   →  ~70 min/run   × 5 runs ≈ 6 hr
  SF=10.0  →  ~250 min/run  × 5 runs ≈ 21 hr (run overnight)

Output (saved to expresults/ subfolder)
─────────────────────────────────────────
  expresults/scale_sf{n}_trace.csv          — per-query trace
  expresults/scale_sf{n}_index_timing.csv   — T_A per window per run
  expresults/scale_sf{n}_hsm.csv            — HSM scores for window pairs
  expresults/scale_results_summary.csv      — appended summary row per SF

Install dependencies
────────────────────
  pip install psycopg2-binary pandas numpy scipy PyWavelets

Configure PostgreSQL credentials in the CONFIG block below.
"""

import sys, os, time, random, math, argparse
import psycopg2, psycopg2.extras
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
try:
    import pywt
except ImportError:
    print("ERROR: PyWavelets not installed.  Run: pip install PyWavelets")
    sys.exit(1)
# NOTE: fastdtw removed (v28 fix). S_P now uses L2 on DWT approximation
# coefficients — a true metric. No extra install needed.

# ══════════════════════════════════════════════════════════════════════
#  CONFIG  — edit to match your PostgreSQL setup
# ══════════════════════════════════════════════════════════════════════
PG = dict(
    host     = "localhost",
    port     = 5432,
    user     = "arunreungsinkonkarn",   # ← your macOS username
    password = "",
    dbname   = "postgres",
)

N_PHASES     = 4
WINS_PHASE   = 8     # 32 windows total per run
N_RUNS       = 3     # repetitions per SF (for statistical stability)
HSM_THETA    = 0.75  # gating threshold (stable if HSM > theta)
SEED         = 2024
INDEX_COLS   = [     # indexes to measure T_A on
    ("lineitem", "l_shipdate"),
    ("lineitem", "l_orderkey"),
    ("orders",   "o_orderdate"),
]

# ══════════════════════════════════════════════════════════════════════
#  Parse arguments
# ══════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="HSM Scale Experiment — Priority 3")
parser.add_argument("--scale", type=float, nargs="+", default=[1.0],
                    help="TPC-H scale factor(s) to run (e.g. --scale 1.0 3.0)")
# Accept both --runs and --n-runs as synonyms
parser.add_argument("--runs", "--n-runs", dest="n_runs", type=int, default=N_RUNS,
                    help=f"Repetitions per SF (default: {N_RUNS})")
parser.add_argument("--verbose", "-v", action="store_true",
                    help="Print detailed progress for each window pair")
args = parser.parse_args()
SCALE_FACTORS = args.scale
N_RUNS = args.n_runs
VERBOSE = args.verbose

# Output dir = expresults/ subfolder next to this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, "expresults")
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY_CSV = os.path.join(OUT_DIR, "scale_results_summary.csv")


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════
def connect(db=None):
    cfg = {**PG, "dbname": db or PG["dbname"]}
    return psycopg2.connect(**cfg)

def rstr(n):
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=n))

def rdate(y0=1992, y1=1998):
    from datetime import date, timedelta
    return date(y0, 1, 1) + timedelta(days=random.randint(0, 365 * (y1 - y0)))

def safe_name(sf):
    """Convert 1.0 → 'sf1_0' for use in DB/file names."""
    return f"sf{str(sf).replace('.', '_')}"


# ══════════════════════════════════════════════════════════════════════
#  TPC-H schema creation
# ══════════════════════════════════════════════════════════════════════
SCHEMA_SQL = """
CREATE TABLE region(
  r_regionkey INTEGER PRIMARY KEY,
  r_name CHAR(25), r_comment VARCHAR(152));

CREATE TABLE nation(
  n_nationkey INTEGER PRIMARY KEY,
  n_name CHAR(25), n_regionkey INTEGER, n_comment VARCHAR(152));

CREATE TABLE supplier(
  s_suppkey INTEGER PRIMARY KEY,
  s_name CHAR(25), s_address VARCHAR(40), s_nationkey INTEGER,
  s_phone CHAR(15), s_acctbal NUMERIC(15,2), s_comment VARCHAR(101));

CREATE TABLE customer(
  c_custkey BIGINT PRIMARY KEY,
  c_name VARCHAR(25), c_address VARCHAR(40), c_nationkey INTEGER,
  c_phone CHAR(15), c_acctbal NUMERIC(15,2),
  c_mktsegment CHAR(10), c_comment VARCHAR(117));

CREATE TABLE part(
  p_partkey BIGINT PRIMARY KEY,
  p_name VARCHAR(55), p_mfgr CHAR(25), p_brand CHAR(10),
  p_type VARCHAR(25), p_size INTEGER, p_container CHAR(10),
  p_retailprice NUMERIC(15,2), p_comment VARCHAR(23));

CREATE TABLE orders(
  o_orderkey BIGINT PRIMARY KEY,
  o_custkey BIGINT, o_orderstatus CHAR(1), o_totalprice NUMERIC(15,2),
  o_orderdate DATE, o_orderpriority CHAR(15), o_clerk CHAR(15),
  o_shippriority INTEGER, o_comment VARCHAR(79));

CREATE TABLE partsupp(
  ps_partkey BIGINT, ps_suppkey INTEGER,
  ps_availqty INTEGER, ps_supplycost NUMERIC(15,2),
  ps_comment VARCHAR(199),
  PRIMARY KEY(ps_partkey, ps_suppkey));

CREATE TABLE lineitem(
  l_orderkey BIGINT, l_partkey BIGINT, l_suppkey INTEGER,
  l_linenumber INTEGER, l_quantity NUMERIC(15,2),
  l_extendedprice NUMERIC(15,2), l_discount NUMERIC(15,2),
  l_tax NUMERIC(15,2), l_returnflag CHAR(1), l_linestatus CHAR(1),
  l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE,
  l_shipinstruct CHAR(25), l_shipmode CHAR(10), l_comment VARCHAR(44),
  PRIMARY KEY(l_orderkey, l_linenumber));
"""

def create_schema(cur):
    cur.execute(SCHEMA_SQL)
    print("    ✓ Schema created (8 tables)")


# ══════════════════════════════════════════════════════════════════════
#  TPC-H data generation
# ══════════════════════════════════════════════════════════════════════
REGIONS  = [(i, n, "c") for i, n in enumerate(
            ["AFRICA","AMERICA","ASIA","EUROPE","MIDDLE EAST"])]
NATIONS  = [(i, n, r, "c") for i, (n, r) in enumerate([
            ("ALGERIA",0),("ARGENTINA",1),("BRAZIL",1),("CANADA",1),
            ("EGYPT",4),("ETHIOPIA",0),("FRANCE",3),("GERMANY",3),
            ("INDIA",2),("INDONESIA",2),("IRAN",4),("IRAQ",4),
            ("JAPAN",2),("JORDAN",4),("KENYA",0),("MOROCCO",0),
            ("MOZAMBIQUE",0),("PERU",1),("CHINA",2),("ROMANIA",3),
            ("SAUDI ARABIA",4),("VIETNAM",2),("RUSSIA",3),
            ("UNITED KINGDOM",3),("UNITED STATES",1)])]

MKTSEGS  = ['AUTOMOBILE','BUILDING','FURNITURE','HOUSEHOLD','MACHINERY']
OPRIOS   = ['1-URGENT','2-HIGH','3-MEDIUM','4-NOT SPECIFIED','5-LOW']
SHIPMODES= ['AIR','FOB','MAIL','RAIL','REG AIR','SHIP','TRUCK']
BRANDS   = [f'Brand#{i}{j}' for i in range(1,6) for j in range(1,6)]
PTYPES   = ['STANDARD ANODIZED TIN','LARGE BRUSHED BRASS',
            'SMALL POLISHED NICKEL','PROMO BURNISHED STEEL','ECONOMY PLATED COPPER']
PCONT    = ['SM BOX','MED BOX','LG BOX','SM PACK','MED PACK',
            'LG PACK','SM CASE','MED CASE','LG CASE','WRAP CASE']


def generate_data(cur, scale):
    N_SUPP  = max(1, int(10_000  * scale))
    N_CUST  = max(1, int(150_000 * scale))
    N_PART  = max(1, int(200_000 * scale))
    N_ORDER = max(1, int(1_500_000 * scale))

    cur.executemany("INSERT INTO region VALUES(%s,%s,%s)", REGIONS)
    cur.executemany("INSERT INTO nation VALUES(%s,%s,%s,%s)", NATIONS)

    print(f"    Loading {N_SUPP:,} suppliers...", end=' ', flush=True)
    data = [(i+1, f'Supplier#{i+1:09d}', rstr(25), random.randint(0,24),
             f'{random.randint(10,99)}-{random.randint(100,999)}-{random.randint(1000,9999)}',
             round(random.uniform(-999,9999),2), rstr(50))
            for i in range(N_SUPP)]
    psycopg2.extras.execute_values(cur, "INSERT INTO supplier VALUES %s",
                                    data, page_size=1000)
    print("✓")

    print(f"    Loading {N_CUST:,} customers...", end=' ', flush=True)
    data = [(i+1, f'Customer#{i+1:09d}', rstr(25), random.randint(0,24),
             f'{random.randint(10,99)}-{random.randint(100,999)}-{random.randint(1000,9999)}',
             round(random.uniform(-999,9999),2), random.choice(MKTSEGS), rstr(50))
            for i in range(N_CUST)]
    psycopg2.extras.execute_values(cur, "INSERT INTO customer VALUES %s",
                                    data, page_size=1000)
    print("✓")

    print(f"    Loading {N_PART:,} parts...", end=' ', flush=True)
    data = [(i+1, rstr(20), f'Manufacturer#{random.randint(1,5)}',
             random.choice(BRANDS), random.choice(PTYPES),
             random.randint(1,50), random.choice(PCONT),
             round(900 + i * 0.01, 2), rstr(20))
            for i in range(N_PART)]
    psycopg2.extras.execute_values(cur, "INSERT INTO part VALUES %s",
                                    data, page_size=1000)
    print("✓")

    print(f"    Loading partsupp...", end=' ', flush=True)
    ps_data = []
    for p in range(1, N_PART + 1):
        for s_off in range(4):
            s = (p + s_off * (N_SUPP // 4 + 1)) % N_SUPP + 1
            ps_data.append((p, s, random.randint(1,9999),
                            round(random.uniform(1,1000),2), rstr(50)))
        if len(ps_data) >= 10000:
            psycopg2.extras.execute_values(cur, "INSERT INTO partsupp VALUES %s",
                                            ps_data, page_size=2000)
            ps_data = []
    if ps_data:
        psycopg2.extras.execute_values(cur, "INSERT INTO partsupp VALUES %s",
                                        ps_data, page_size=2000)
    del ps_data
    print("✓")

    print(f"    Loading {N_ORDER:,} orders + lineitems...", end=' ', flush=True)
    ord_data, li_data, li_count = [], [], 0
    for o in range(1, N_ORDER + 1):
        cust  = random.randint(1, N_CUST)
        odate = rdate()
        ord_data.append((o, cust, 'O' if random.random() < 0.5 else 'F',
                         round(random.uniform(1000,200000),2),
                         odate, random.choice(OPRIOS),
                         f'Clerk#{random.randint(1,1000):09d}', 0, rstr(30)))
        for ln in range(1, random.randint(1,7) + 1):
            p = random.randint(1, N_PART); s = random.randint(1, N_SUPP)
            qty = random.randint(1, 50); ep = round(qty * random.uniform(10,100), 2)
            sd  = rdate(1992, 1998)
            li_data.append((o, p, s, ln, qty, ep,
                            round(random.uniform(0,0.1),2),
                            round(random.uniform(0,0.08),2),
                            random.choice(['A','N','R']),
                            random.choice(['O','F']),
                            sd, sd, sd,
                            random.choice(['DELIVER IN PERSON','COLLECT COD','NONE']),
                            random.choice(SHIPMODES), rstr(20)))
            li_count += 1
        if len(ord_data) >= 5000:
            psycopg2.extras.execute_values(cur, "INSERT INTO orders VALUES %s",
                                            ord_data, page_size=2000)
            psycopg2.extras.execute_values(cur, "INSERT INTO lineitem VALUES %s",
                                            li_data, page_size=2000)
            ord_data, li_data = [], []
    if ord_data:
        psycopg2.extras.execute_values(cur, "INSERT INTO orders VALUES %s",
                                        ord_data, page_size=2000)
        psycopg2.extras.execute_values(cur, "INSERT INTO lineitem VALUES %s",
                                        li_data, page_size=2000)
    print(f"✓  ({li_count:,} lineitems)")
    return N_CUST, li_count


# ══════════════════════════════════════════════════════════════════════
#  Query / DML definitions (same as run_experiment_v2.py)
# ══════════════════════════════════════════════════════════════════════
SELECT_QUERIES = {
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
GROUP BY l_orderkey,o_orderdate,o_shippriority ORDER BY rev DESC LIMIT 10""",
"Q4": """SELECT o_orderpriority, COUNT(*) AS cnt
FROM orders WHERE o_orderdate>='1993-07-01' AND o_orderdate<'1993-10-01'
  AND EXISTS(SELECT 1 FROM lineitem
             WHERE l_orderkey=o_orderkey AND l_commitdate<l_receiptdate)
GROUP BY o_orderpriority ORDER BY 1""",
"Q5": """SELECT n_name, SUM(l_extendedprice*(1-l_discount)) AS rev
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
"Q11": """SELECT ps_partkey, SUM(ps_supplycost*ps_availqty) AS val
FROM partsupp JOIN supplier ON ps_suppkey=s_suppkey
JOIN nation ON s_nationkey=n_nationkey
WHERE n_name='GERMANY'
GROUP BY ps_partkey
HAVING SUM(ps_supplycost*ps_availqty)>(
  SELECT SUM(ps_supplycost*ps_availqty)*0.0001
  FROM partsupp JOIN supplier ON ps_suppkey=s_suppkey
  JOIN nation ON s_nationkey=n_nationkey WHERE n_name='GERMANY')
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
"Q17": """WITH avg_qty AS (
  SELECT l_partkey, 0.2*AVG(l_quantity) AS threshold
  FROM lineitem GROUP BY l_partkey)
SELECT SUM(l.l_extendedprice)/7.0
FROM lineitem l JOIN part p ON p.p_partkey=l.l_partkey
JOIN avg_qty a ON a.l_partkey=l.l_partkey
WHERE p.p_brand='Brand#23' AND p.p_container='MED BOX'
  AND l.l_quantity < a.threshold""",
"Q18": """SELECT c_name, c_custkey, o_orderkey, o_orderdate,
  o_totalprice, SUM(l_quantity)
FROM customer JOIN orders ON c_custkey=o_custkey
JOIN lineitem ON o_orderkey=l_orderkey
WHERE o_orderkey IN(
  SELECT l_orderkey FROM lineitem
  GROUP BY l_orderkey HAVING SUM(l_quantity)>300)
GROUP BY 1,2,3,4,5 ORDER BY o_totalprice DESC LIMIT 10""",
}

QUERY_TABLES = {
    "Q1": {"lineitem"}, "Q3": {"customer","orders","lineitem"},
    "Q4": {"orders","lineitem"},
    "Q5": {"customer","orders","lineitem","supplier","nation","region"},
    "Q6": {"lineitem"}, "Q7": {"supplier","lineitem","orders","customer","nation"},
    "Q10":{"customer","orders","lineitem","nation"},
    "Q11":{"partsupp","supplier","nation"}, "Q12":{"orders","lineitem"},
    "Q14":{"lineitem","part"}, "Q17":{"lineitem","part"},
    "Q18":{"customer","orders","lineitem"},
}
QUERY_COLS = {
    "Q1": {"lineitem.l_shipdate","lineitem.l_returnflag","lineitem.l_linestatus",
           "lineitem.l_quantity","lineitem.l_extendedprice","lineitem.l_discount","lineitem.l_tax"},
    "Q3": {"customer.c_mktsegment","orders.o_orderdate","orders.o_shippriority",
           "lineitem.l_orderkey","lineitem.l_shipdate","lineitem.l_extendedprice","lineitem.l_discount"},
    "Q4": {"orders.o_orderdate","orders.o_orderpriority",
           "lineitem.l_commitdate","lineitem.l_receiptdate"},
    "Q5": {"customer.c_nationkey","orders.o_orderdate","lineitem.l_extendedprice",
           "lineitem.l_discount","lineitem.l_suppkey","supplier.s_nationkey",
           "nation.n_name","region.r_name"},
    "Q6": {"lineitem.l_shipdate","lineitem.l_discount","lineitem.l_quantity","lineitem.l_extendedprice"},
    "Q7": {"supplier.s_nationkey","lineitem.l_shipdate","lineitem.l_extendedprice",
           "lineitem.l_discount","orders.o_custkey","customer.c_nationkey","nation.n_name"},
    "Q10":{"customer.c_custkey","customer.c_name","customer.c_acctbal","customer.c_nationkey",
           "orders.o_orderdate","lineitem.l_returnflag","lineitem.l_extendedprice",
           "lineitem.l_discount","nation.n_name"},
    "Q11":{"partsupp.ps_suppkey","partsupp.ps_supplycost","partsupp.ps_availqty",
           "supplier.s_nationkey","nation.n_name"},
    "Q12":{"lineitem.l_shipmode","lineitem.l_commitdate","lineitem.l_receiptdate",
           "lineitem.l_shipdate","orders.o_orderpriority"},
    "Q14":{"lineitem.l_partkey","lineitem.l_shipdate","lineitem.l_extendedprice",
           "lineitem.l_discount","part.p_type"},
    "Q17":{"lineitem.l_partkey","lineitem.l_quantity","lineitem.l_extendedprice",
           "part.p_brand","part.p_container"},
    "Q18":{"customer.c_name","customer.c_custkey","orders.o_orderkey",
           "orders.o_orderdate","orders.o_totalprice","lineitem.l_quantity"},
}
DML_TABLES = {"U_orders":{"orders"},"I_customer":{"customer"},
              "D_orders":{"orders"},"U_customer":{"customer"}}
DML_COLS   = {"U_orders":{"orders.o_comment"},
              "I_customer":{"customer.c_name","customer.c_address","customer.c_comment"},
              "D_orders":{"orders.o_orderkey"},
              "U_customer":{"customer.c_comment","customer.c_acctbal"}}

PHASES = [
    {"name":"Reporting",   "sel_qs":["Q1","Q6","Q14","Q12","Q4"],
     "sel_wts":[0.35,0.25,0.20,0.10,0.10], "dml_type":"U_orders",
     "dml_frac":0.20, "n_q":30, "pattern":"uniform"},
    {"name":"JoinHeavy",   "sel_qs":["Q3","Q5","Q7","Q10","Q18"],
     "sel_wts":[0.30,0.25,0.20,0.15,0.10], "dml_type":"I_customer",
     "dml_frac":0.45, "n_q":40, "pattern":"frontload"},
    {"name":"Procurement", "sel_qs":["Q11","Q17","Q6","Q1","Q14"],
     "sel_wts":[0.35,0.25,0.20,0.10,0.10], "dml_type":"D_orders",
     "dml_frac":0.15, "n_q":25, "pattern":"backload"},
    {"name":"Customer",    "sel_qs":["Q10","Q18","Q3","Q5","Q7"],
     "sel_wts":[0.30,0.25,0.25,0.10,0.10], "dml_type":"U_customer",
     "dml_frac":0.50, "n_q":20, "pattern":"oscillate"},
]
N_WINDOWS = N_PHASES * WINS_PHASE


def make_window_query_list(phase):
    n_total = phase["n_q"]
    n_dml   = max(1, round(n_total * phase["dml_frac"]))
    n_sel   = n_total - n_dml
    wts     = np.array(phase["sel_wts"]); wts /= wts.sum()
    sel_list = [(q, "SELECT") for q in np.random.choice(
                 phase["sel_qs"], size=n_sel, p=wts, replace=True)]
    dml_list = [(phase["dml_type"],
                 "UPDATE" if phase["dml_type"].startswith("U") else
                 "INSERT" if phase["dml_type"].startswith("I") else "DELETE")
                for _ in range(n_dml)]
    pat = phase["pattern"]
    if pat == "uniform":
        combined = sel_list + dml_list; random.shuffle(combined)
    elif pat == "frontload":
        combined = sel_list + dml_list
    elif pat == "backload":
        combined = dml_list + sel_list
    elif pat == "oscillate":
        combined, i, j, toggle = [], 0, 0, True
        while i < len(sel_list) or j < len(dml_list):
            if toggle and i < len(sel_list):
                combined.append(sel_list[i]); i += 1
            elif j < len(dml_list):
                combined.append(dml_list[j]); j += 1
            elif i < len(sel_list):
                combined.append(sel_list[i]); i += 1
            toggle = not toggle
    else:
        combined = sel_list + dml_list; random.shuffle(combined)
    return combined


# ══════════════════════════════════════════════════════════════════════
#  HSM computation (same as run_experiment_v2.py)
# ══════════════════════════════════════════════════════════════════════
DWT_WAVELET = 'db2'   # db2 (filter len 4) avoids boundary-effect warnings with 8-slot series;
                      # db4 (prev.) triggers pywt "too high" warning for N_SLOTS=8 at level 1.
DWT_LEVEL   = 1
N_SLOTS     = 8
# SAX_ALPHA, BAND_WEIGHTS, FASTDTW_R removed — S_P now uses L2 on DWT
# approximation coefficients (a true metric) instead of FastDTW on SAX symbols.

def _build_qps_series(w, n_slots=N_SLOTS):
    n = len(w)
    if n == 0:
        return np.zeros(n_slots)
    seqs   = w["seq"].values.astype(float)
    ms     = w["exec_ms"].values
    norm_s = (seqs - seqs.min()) / max(seqs.max() - seqs.min(), 1)
    slots  = np.floor(norm_s * n_slots).clip(0, n_slots - 1).astype(int)
    series = np.zeros(n_slots); counts = np.zeros(n_slots)
    for sl, m in zip(slots, ms):
        series[sl] += m; counts[sl] += 1
    counts[counts == 0] = 1
    return series / counts

def compute_sp(w1, w2):
    """S_P: performance-pattern similarity via L2 on normalized DWT approx coefficients.

    Algorithm (v28, Option A fix):
      1. Build per-window avg-latency-per-slot series (8 slots).
      2. DWT decompose with db2 wavelet; keep approximation coefficients (band 0).
      3. Normalize each coefficient vector to the unit L2-sphere.
      4. d_P = ||a1_unit - a2_unit||_2  ∈ [0, 2]  (L2 distance — IS a metric).
      5. S_P = 1 - d_P/2  ∈ [0, 1].

    Why this is a proper metric:
      - L2 distance on R^n satisfies all four metric axioms.
      - Normalizing to the unit sphere and halving are bijective/scaling transforms
        that preserve metric properties.
      - FastDTW (previous) lacked a triangle-inequality proof; now removed.
    """
    q1 = _build_qps_series(w1)
    q2 = _build_qps_series(w2)
    # DWT approximation coefficients only (coarsest-scale representation)
    a1 = pywt.wavedec(q1, DWT_WAVELET, level=DWT_LEVEL)[0]
    a2 = pywt.wavedec(q2, DWT_WAVELET, level=DWT_LEVEL)[0]
    n1, n2 = np.linalg.norm(a1), np.linalg.norm(a2)
    if n1 < 1e-9 and n2 < 1e-9:
        return 1.0   # both series are flat-zero → identical patterns
    a1_unit = a1 / (n1 + 1e-9)
    a2_unit = a2 / (n2 + 1e-9)
    l2_dist = np.linalg.norm(a1_unit - a2_unit)   # ∈ [0, 2] for unit vectors
    return float(np.clip(1.0 - l2_dist / 2.0, 0.0, 1.0))

def compute_hsm(w1, w2):
    def sel_ratio(w):
        n = len(w); return (w["op_type"] == "SELECT").sum() / n if n > 0 else 0.0
    sR = max(0.0, 1.0 - abs(sel_ratio(w1) - sel_ratio(w2)))
    n1, n2 = max(len(w1),1), max(len(w2),1)
    sV = math.exp(-abs(math.log10(n1) - math.log10(n2)))
    def op_vec(w):
        return np.array([(w["op_type"]==t).sum() for t in
                         ["SELECT","UPDATE","INSERT","DELETE"]], dtype=float)
    v1, v2 = op_vec(w1), op_vec(w2)
    d1, d2 = np.linalg.norm(v1), np.linalg.norm(v2)
    # S_T (v28 fix): angular distance on unit sphere IS a proper metric.
    # 1 - cosine is NOT metric (triangle inequality violated); arccos/π is.
    if d1 > 0 and d2 > 0:
        cos_val = np.clip(np.dot(v1, v2) / (d1 * d2), -1.0, 1.0)
        ang_dist = np.arccos(cos_val) / np.pi   # ∈ [0,1], satisfies all metric axioms
        sT = float(1.0 - ang_dist)
    else:
        sT = 1.0 if np.array_equal(v1, v2) else 0.0
    def get_sets(w):
        tables, cols = set(), set()
        for q in w["query"]:
            tables |= (QUERY_TABLES.get(q,set()) | DML_TABLES.get(q,set()))
            cols   |= (QUERY_COLS.get(q,set()) | DML_COLS.get(q,set()))
        return tables, cols
    t1,c1 = get_sets(w1); t2,c2 = get_sets(w2)
    tj = len(t1&t2)/len(t1|t2) if (t1|t2) else 1.0
    cj = len(c1&c2)/len(c1|c2) if (c1|c2) else 1.0
    sA = 0.5*tj + 0.5*cj
    sP = compute_sp(w1, w2)
    # Weights per paper definition: S_R=0.25, S_V=0.20, S_T=0.20, S_A=0.20, S_P=0.15
    # (v28 fix: was 0.20 equal for all — incorrect)
    hsm = 0.25*sR + 0.20*sV + 0.20*sT + 0.20*sA + 0.15*sP
    return sR, sV, sT, sA, sP, hsm


# ══════════════════════════════════════════════════════════════════════
#  Index rebuild timing measurement
# ══════════════════════════════════════════════════════════════════════
def measure_index_rebuild(conn, n_warmup=2):
    """
    Drop and recreate each index in INDEX_COLS, time the CREATE INDEX.
    Returns list of (table, col, rebuild_ms).
    Returns mean T_A across all indexes.
    """
    results = []
    conn.autocommit = True
    cur = conn.cursor()
    # Drop all non-PK indexes on target tables
    for tbl, col in INDEX_COLS:
        idx_name = f"idx_{tbl}_{col}"
        cur.execute(f"DROP INDEX IF EXISTS {idx_name}")

    # Warmup + timed runs
    for _ in range(n_warmup):
        for tbl, col in INDEX_COLS:
            idx_name = f"idx_{tbl}_{col}"
            cur.execute(f"DROP INDEX IF EXISTS {idx_name}")
            cur.execute(f"CREATE INDEX {idx_name} ON {tbl}({col})")
            cur.execute(f"DROP INDEX IF EXISTS {idx_name}")

    # Timed measurement (3 reps, take median)
    for tbl, col in INDEX_COLS:
        idx_name = f"idx_{tbl}_{col}"
        times = []
        for _ in range(3):
            cur.execute(f"DROP INDEX IF EXISTS {idx_name}")
            t0 = time.perf_counter()
            cur.execute(f"CREATE INDEX {idx_name} ON {tbl}({col})")
            ms = (time.perf_counter() - t0) * 1000
            times.append(ms)
        median_ms = float(np.median(times))
        results.append({"table": tbl, "col": col, "rebuild_ms": round(median_ms, 2)})
        print(f"      T_A({tbl}.{col}): {median_ms:.1f} ms")
    cur.close()
    return results


# ══════════════════════════════════════════════════════════════════════
#  Main experiment loop — one scale factor at a time
# ══════════════════════════════════════════════════════════════════════
def run_one_sf(scale):
    random.seed(SEED); np.random.seed(SEED)
    sfname  = safe_name(scale)
    db_name = f"tpch_scale_{sfname}"

    print("\n" + "="*65)
    print(f"  HSM Scale Experiment — SF={scale}  (N_RUNS={N_RUNS})")
    print("="*65)

    # ── Create / recreate database ────────────────────────────────────
    print(f"\n[1] Creating database '{db_name}'...")
    try:
        conn0 = connect(); conn0.autocommit = True
        cur0  = conn0.cursor()
        cur0.execute(f"DROP DATABASE IF EXISTS {db_name}")
        cur0.execute(f"CREATE DATABASE {db_name}")
        conn0.close()
        print("    ✓ database created")
    except Exception as e:
        print(f"    ✗ {e}"); sys.exit(1)

    conn = connect(db_name); conn.autocommit = True
    cur  = conn.cursor()

    # ── Schema + data ─────────────────────────────────────────────────
    print(f"\n[2] Creating schema...")
    create_schema(cur)

    print(f"\n[3] Generating TPC-H SF={scale} data...")
    t_gen_start = time.perf_counter()
    N_CUST, n_lineitem = generate_data(cur, scale)
    t_gen = time.perf_counter() - t_gen_start
    print(f"\n    ✓ Data generation complete: {n_lineitem:,} lineitems in {t_gen:.1f}s")

    # Collect valid PKs for DML
    cur.execute("SELECT o_orderkey FROM orders ORDER BY RANDOM() LIMIT 5000")
    ORDER_PKS = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT c_custkey FROM customer ORDER BY RANDOM() LIMIT 5000")
    CUST_PKS  = [r[0] for r in cur.fetchall()]
    _next_cust_key = [N_CUST + 1]  # mutable for closure

    def run_dml(cur, dml_type):
        t0 = time.perf_counter()
        try:
            if dml_type == "U_orders":
                ok_key = random.choice(ORDER_PKS)
                cur.execute("UPDATE orders SET o_comment=%s WHERE o_orderkey=%s",
                            (rstr(20), ok_key))
            elif dml_type == "I_customer":
                nk = _next_cust_key[0]; _next_cust_key[0] += 1
                cur.execute("INSERT INTO customer VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
                            (nk, f'Customer#{nk:09d}', rstr(20),
                             random.randint(0,24),
                             f'{random.randint(10,99)}-{random.randint(100,999)}-{random.randint(1000,9999)}',
                             round(random.uniform(-999,9999),2),
                             random.choice(MKTSEGS), rstr(40)))
            elif dml_type == "D_orders":
                ok_key = random.choice(ORDER_PKS)
                cur.execute("DELETE FROM orders WHERE o_orderkey=%s AND o_orderstatus='F'",
                            (ok_key,))
            elif dml_type == "U_customer":
                ck = random.choice(CUST_PKS)
                cur.execute("UPDATE customer SET c_comment=%s WHERE c_custkey=%s",
                            (rstr(30), ck))
            ms = (time.perf_counter() - t0) * 1000
            return True, ms
        except Exception:
            conn.rollback()
            return False, (time.perf_counter() - t0) * 1000

    # ── N runs of workload ────────────────────────────────────────────
    all_trace    = []
    all_hsm      = []
    all_idx_time = []

    for run_id in range(1, N_RUNS + 1):
        print(f"\n[4.{run_id}] Run {run_id}/{N_RUNS} — workload ({N_WINDOWS} windows)...")
        trace = []
        for win in range(N_WINDOWS):
            ph    = PHASES[win // WINS_PHASE]
            qlist = make_window_query_list(ph)
            win_ms = 0.0
            for seq, (qname, op_type) in enumerate(qlist):
                if op_type == "SELECT":
                    t0 = time.perf_counter()
                    try:
                        cur.execute(SELECT_QUERIES[qname])
                        cur.fetchall(); ok = True
                    except Exception:
                        conn.rollback(); ok = False
                    ms = (time.perf_counter() - t0) * 1000
                else:
                    ok, ms = run_dml(cur, qname)
                win_ms += ms
                trace.append({"run":run_id, "window":win, "phase":ph["name"],
                               "seq":seq, "query":qname, "op_type":op_type,
                               "exec_ms":round(ms,4), "ok":ok})
            if VERBOSE:
                print(f"    Run{run_id} W{win:02d}[{ph['name'][:8]:8s}]"
                      f"  {len(qlist):2d}q  total={win_ms:7.1f}ms  avg={win_ms/len(qlist):6.1f}ms/q")
            else:
                print(f"    Run{run_id} W{win:02d}", end="\r", flush=True)

        df = pd.DataFrame(trace)
        ok_df = df[df["ok"]]
        all_trace.append(df)

        # ── Compute HSM for this run ───────────────────────────────
        wins = sorted(df["window"].unique())
        for i in range(len(wins)-1):
            w1 = ok_df[ok_df["window"]==wins[i]]
            w2 = ok_df[ok_df["window"]==wins[i+1]]
            p1 = w1["phase"].iloc[0]; p2 = w2["phase"].iloc[0]
            sR,sV,sT,sA,sP,hsm = compute_hsm(w1, w2)
            all_hsm.append({"run":run_id,"w_from":wins[i],"w_to":wins[i+1],
                             "phase_from":p1,"phase_to":p2,"cross_phase":(p1!=p2),
                             "S_R":round(sR,4),"S_V":round(sV,4),"S_T":round(sT,4),
                             "S_A":round(sA,4),"S_P":round(sP,4),"HSM":round(hsm,4)})

        # ── Measure index rebuild time (T_A) ──────────────────────────
        print(f"\n[4.{run_id}] Measuring T_A (index rebuild) at N={n_lineitem:,}...")
        idx_rows = measure_index_rebuild(conn)
        for r in idx_rows:
            r.update({"run": run_id, "sf": scale, "n_lineitem": n_lineitem})
            all_idx_time.append(r)

    conn.close()

    # ── Save output files ─────────────────────────────────────────────
    trace_csv  = os.path.join(OUT_DIR, f"scale_{sfname}_trace.csv")
    hsm_csv    = os.path.join(OUT_DIR, f"scale_{sfname}_hsm.csv")
    idx_csv    = os.path.join(OUT_DIR, f"scale_{sfname}_index_timing.csv")

    pd.concat(all_trace).to_csv(trace_csv, index=False)
    hdf = pd.DataFrame(all_hsm)
    hdf.to_csv(hsm_csv, index=False)
    pd.DataFrame(all_idx_time).to_csv(idx_csv, index=False)
    print(f"\n    ✓ Saved: {trace_csv}")
    print(f"    ✓ Saved: {hsm_csv}")
    print(f"    ✓ Saved: {idx_csv}")

    # ── Summary statistics ────────────────────────────────────────────
    within = hdf[~hdf["cross_phase"]]
    across = hdf[ hdf["cross_phase"]]

    # Discrimination ratio + Mann-Whitney U
    w_scores = within["HSM"].values
    c_scores = across["HSM"].values
    mean_within = float(w_scores.mean()) if len(w_scores) > 0 else 0.0
    mean_cross  = float(c_scores.mean()) if len(c_scores) > 0 else 0.0
    dr = mean_within / max(mean_cross, 1e-9)
    try:
        stat, p_val = mannwhitneyu(w_scores, c_scores, alternative='greater')
    except Exception:
        p_val = float('nan')

    # p_stable = fraction of within-phase transitions where HSM > theta
    p_stable = float((w_scores > HSM_THETA).mean()) if len(w_scores) > 0 else 0.0

    # T_A mean across indexes and runs
    idf = pd.DataFrame(all_idx_time)
    t_a_mean = float(idf["rebuild_ms"].mean())
    t_a_std  = float(idf["rebuild_ms"].std())

    # Gating savings calculation (Theorem 9)
    # T_total_no_gating   = n_windows × T_A_mean
    # T_total_with_gating = n_windows × T_A_mean × (1 - p_stable)
    savings_pct = p_stable * 100.0
    speedup     = 1.0 / max(1.0 - p_stable, 1e-9)

    summary = {
        "sf": scale, "n_lineitem": n_lineitem,
        "n_runs": N_RUNS,
        "T_A_mean_ms": round(t_a_mean, 2),
        "T_A_std_ms":  round(t_a_std, 2),
        "DR_HSM":      round(dr, 4),
        "p_value":     round(p_val, 6) if not math.isnan(p_val) else "nan",
        "p_stable":    round(p_stable, 4),
        "savings_pct": round(savings_pct, 1),
        "speedup_x":   round(speedup, 2),
    }

    # Append to summary CSV
    sumdf = pd.DataFrame([summary])
    if os.path.exists(SUMMARY_CSV):
        existing = pd.read_csv(SUMMARY_CSV)
        existing = existing[existing["sf"] != scale]  # overwrite same SF
        sumdf = pd.concat([existing, sumdf], ignore_index=True)
    sumdf.to_csv(SUMMARY_CSV, index=False)
    print(f"    ✓ Saved/updated: {SUMMARY_CSV}")

    print(f"""
{'='*65}
  RESULTS — SF={scale}
{'='*65}
  N_lineitem       : {n_lineitem:,}
  T_A (mean)       : {t_a_mean:.1f} ms  (σ={t_a_std:.1f})
  HSM DR           : {dr:.4f}
  p-value          : {p_val:.4f}
  p_stable (θ={HSM_THETA}) : {p_stable:.1%}
  Index savings    : {savings_pct:.1f}%
  Speedup          : {speedup:.2f}×  (theory: {1/(1-0.9):.1f}× at p=0.90)
{'='*65}""")

    return summary


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*65)
    print("  HSM Paper 3A — Priority 3: Scale Analysis")
    print(f"  Scale factors: {SCALE_FACTORS}")
    print(f"  Runs per SF  : {N_RUNS}")
    print("="*65)

    all_summaries = []
    for sf in SCALE_FACTORS:
        summary = run_one_sf(sf)
        all_summaries.append(summary)

    # ── Final combined table ──────────────────────────────────────────
    print("\n" + "="*65)
    print("  FINAL SCALE RESULTS TABLE (all SFs run this session)")
    print("="*65)
    print(f"  {'SF':>5}  {'N_lineitem':>12}  {'T_A_ms':>9}  {'Savings%':>9}  {'Speedup':>8}  {'DR':>7}  {'p-value':>9}")
    print("  " + "-"*65)
    for s in all_summaries:
        p_str = f"{s['p_value']:.4f}" if s['p_value'] != "nan" else "  nan"
        print(f"  {s['sf']:>5}  {s['n_lineitem']:>12,}  {s['T_A_mean_ms']:>9.1f}"
              f"  {s['savings_pct']:>8.1f}%  {s['speedup_x']:>7.2f}×"
              f"  {s['DR_HSM']:>7.4f}  {p_str:>9}")
    print("="*65)
    print(f"\n  Full summary CSV: {SUMMARY_CSV}")
    print("  Upload scale_results_summary.csv + scale_sf*_*.csv to Claude.")
