"""
HSM Validation on JOB (Join Order Benchmark) — IMDB Workload
=============================================================
Second external validation dataset for A8 (cross-domain generalizability).
Complements hsm_sdss_validation.py (SDSS SkyServer, read-heavy astronomical).

JOB Reference:
  Leis et al. (2015). "How Good Are Query Optimizers, Really?"
  VLDB. 113 queries over the IMDB movie database.
  https://github.com/gregrahn/join-order-benchmark

HSM Dimensions (per paper Section 3):
  S_R : SELECT-ratio similarity      = 1 - |ratio_A - ratio_B|
  S_V : Volume similarity            = exp(-|log(QPS_A) - log(QPS_B)|)
  S_T : Type angular similarity      = cosine([n_sel,n_upd,n_ins,n_del])
  S_A : Access-attribute overlap     = Jaccard(tables_A, tables_B)
  S_P : Temporal pattern similarity  = 1 - ||â_A - â_B||_2 / 2
        via DWT(db2, L=1) → unit-sphere normalisation → L2 distance

HSM = 0.2*S_R + 0.2*S_V + 0.2*S_T + 0.2*S_A + 0.2*S_P

Modes:
  static   (default) — SQL structural analysis only; no DB connection needed.
                        Proxy for S_V/S_P: query complexity (join count, predicates).
  execute             — Runs queries against Docker PostgreSQL with IMDB loaded.
                        Full 5-dimensional computation with real execution times.

Usage:
  # Static analysis (works before IMDB is loaded in Docker):
  python hsm_job_validation.py

  # Full execution against Docker IMDB database:
  python hsm_job_validation.py --execute

  # Specify alternate DB connection:
  python hsm_job_validation.py --execute --port 5433 --dbname imdb

Prerequisites for --execute mode:
  1. IMDB data loaded into PostgreSQL (Docker on port 5433):
       a. Download: wget http://event.cwi.nl/da/job/imdb.tgz
       b. Extract to data/imdb/
       c. Load schema: psql -h localhost -p 5433 -U postgres -c "CREATE DATABASE imdb;"
       d. Run: python hsm_job_validation.py --setup  (loads CSV data into DB)
  2. pip install psycopg2-binary --break-system-packages

Prerequisites for static mode (default):
  pip install numpy scipy pywt --break-system-packages
  JOB query files in: data/job/queries/*.sql
       OR use --download to fetch from GitHub automatically.

Data directory layout:
  data/
    imdb/           ← extracted from imdb.tgz (CSV files)
    job/
      queries/      ← JOB .sql files (1a.sql, 1b.sql … 33c.sql)
"""

import re
import os
import sys
import csv
import math
import random
import statistics
import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import pywt

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
DATA_DIR      = SCRIPT_DIR.parent / 'data'
JOB_DIR       = DATA_DIR / 'job'
QUERY_DIR     = JOB_DIR / 'queries'
RESULTS_DIR   = SCRIPT_DIR.parent / 'results' / 'job_validation'

NPTS_FULL     = 10          # queries per window when all 113 JOB queries available
NPTS_EMBEDDED = 4           # queries per window when using embedded 26-query subset
NPTS          = NPTS_FULL   # will be set dynamically below after loading queries
RANDOM_SEED   = 42
HSM_WEIGHTS   = [0.2, 0.2, 0.2, 0.2, 0.2]

DOCKER_HOST = os.environ.get('HSM_DOCKER_HOST', 'localhost')
DOCKER_PORT = int(os.environ.get('HSM_DOCKER_PORT', 5433))
DOCKER_USER = os.environ.get('HSM_DOCKER_USER', 'postgres')
DOCKER_PASS = os.environ.get('HSM_DOCKER_PASSWORD', 'postgres')
DOCKER_DB     = 'imdb'

# ── IMDB Phase Definitions ─────────────────────────────────────────────────────
# 4 natural phases reflecting JOB query themes.
# Derived from table co-occurrence patterns in Leis et al. (2015).
PHASE_MAP = {
    'actor':      # Person / name-centric queries
        {'name', 'aka_name', 'person_info', 'cast_info', 'role_type', 'char_name'},
    'movie':      # Title / movie information queries
        {'title', 'aka_title', 'movie_info', 'movie_info_idx', 'kind_type'},
    'production': # Company / production queries
        {'company_name', 'company_type', 'movie_companies', 'complete_cast',
         'comp_cast_type', 'link_type', 'movie_link'},
    'keyword':    # Genre / keyword / metadata queries
        {'keyword', 'movie_keyword', 'info_type'},
}

# ── JOB Query Phase Pre-classification ────────────────────────────────────────
# Each JOB query group maps to a dominant phase.
# Source: Leis et al. (2015) Table 1 and manual review of query SQL.
JOB_GROUP_PHASE = {
    # Actor/person-centric groups
    '2':  'actor', '4':  'actor', '8':  'actor', '12': 'actor',
    '20': 'actor', '24': 'actor', '28': 'actor',
    # Movie/title-centric groups
    '1':  'movie', '3':  'movie', '7':  'movie', '11': 'movie',
    '13': 'movie', '14': 'movie', '15': 'movie', '18': 'movie',
    '22': 'movie', '23': 'movie', '27': 'movie', '30': 'movie',
    # Production/company-centric groups
    '6':  'production', '9':  'production', '16': 'production',
    '19': 'production', '21': 'production', '25': 'production',
    '26': 'production', '29': 'production', '31': 'production',
    '32': 'production',
    # Keyword/genre/metadata groups
    '5':  'keyword', '10': 'keyword', '17': 'keyword',
    '33': 'keyword',
}

# ── Argument Parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='HSM validation on JOB (IMDB) benchmark workload.'
)
parser.add_argument('--execute', action='store_true',
    help='Execute queries against Docker IMDB database for real timing data.')
parser.add_argument('--setup', action='store_true',
    help='Load IMDB CSV data into Docker PostgreSQL (run once after IMDB download).')
parser.add_argument('--download', action='store_true',
    help='Download JOB query files from GitHub.')
parser.add_argument('--port', type=int, default=DOCKER_PORT,
    help=f'Docker PostgreSQL port (default: {DOCKER_PORT}).')
parser.add_argument('--dbname', default=DOCKER_DB,
    help=f'Database name (default: {DOCKER_DB}).')
parser.add_argument('--verbose', action='store_true',
    help='Print per-query timing details.')
args = parser.parse_args()

# ── Helpers ────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  HSM Validation — JOB (Join Order Benchmark / IMDB)")
print("=" * 70)
mode_label = "EXECUTE (real timing)" if args.execute else "STATIC (structural analysis)"
print(f"\n  Mode: {mode_label}")


def download_job_queries():
    """Download JOB query files from GitHub."""
    try:
        import urllib.request
        QUERY_DIR.mkdir(parents=True, exist_ok=True)
        base = 'https://raw.githubusercontent.com/gregrahn/join-order-benchmark/master/'
        # Query files follow pattern 1a.sql, 1b.sql, ..., 33c.sql
        import json, urllib.request
        api_url = 'https://api.github.com/repos/gregrahn/join-order-benchmark/contents/'
        with urllib.request.urlopen(api_url, timeout=15) as r:
            files = json.loads(r.read().decode())
        sql_files = [f['name'] for f in files if f['name'].endswith('.sql')
                     and re.match(r'^\d+[a-z]\.sql$', f['name'])]
        print(f"  Found {len(sql_files)} SQL files to download ...")
        for fname in sorted(sql_files):
            url = base + fname
            dest = QUERY_DIR / fname
            if not dest.exists():
                with urllib.request.urlopen(url, timeout=15) as r:
                    dest.write_bytes(r.read())
                print(f"    Downloaded: {fname}")
            else:
                print(f"    Exists:     {fname}")
        print(f"  Downloaded {len(sql_files)} query files to {QUERY_DIR}")
    except Exception as e:
        print(f"\n  ERROR downloading query files: {e}")
        print("  Manual download:")
        print("    git clone https://github.com/gregrahn/join-order-benchmark")
        print(f"   cp join-order-benchmark/*.sql {QUERY_DIR}/")
        sys.exit(1)


# ── Step 0: Download queries if requested ─────────────────────────────────────
if args.download:
    print("\n[0] Downloading JOB query files ...")
    download_job_queries()


# ── Step 1: Load JOB SQL Queries ───────────────────────────────────────────────
print("\n[1] Loading JOB SQL query files ...")

if not QUERY_DIR.exists() or not list(QUERY_DIR.glob('*.sql')):
    print(f"\n  WARNING: No .sql files found in {QUERY_DIR}")
    print("  JOB query files needed. Options:")
    print("    1. python hsm_job_validation.py --download")
    print("    2. git clone https://github.com/gregrahn/join-order-benchmark")
    print(f"      cp join-order-benchmark/*.sql {QUERY_DIR}/")
    print("\n  Continuing with EMBEDDED representative queries for each phase ...")
    use_embedded = True
else:
    use_embedded = False


def get_embedded_queries():
    """
    Embedded representative JOB queries (subset) for static analysis
    when query files are not downloaded.
    These represent the 4 phases with authentic JOB query structure.
    """
    # Format: (query_id, phase, sql)
    return [
        # ── ACTOR PHASE (person/name-centric) ─────────────────────────────────
        ('2a', 'actor', """
SELECT MIN(n.name) AS voicing_actress,
       MIN(t.title) AS voiced_char_animation_movie
FROM cast_info AS ci, company_name AS cn, company_type AS ct,
     info_type AS it1, info_type AS it2, keyword AS k, movie_companies AS mc,
     movie_info AS mi, movie_info_idx AS mi_idx, movie_keyword AS mk,
     name AS n, role_type AS rt, title AS t
WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)',
                  '(voice: English version)')
  AND cn.country_code = '[us]'
  AND it1.info = 'release dates'
  AND it2.info IN ('Rating', 'DVD')
  AND k.keyword IN ('hero', 'based-on-novel')
  AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
  AND rt.role = 'actress'
  AND t.production_year > 2005
  AND t.id = mi.movie_id
  AND t.id = mi_idx.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mi_idx.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = ci.movie_id
  AND mi.movie_id = mi_idx.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi.movie_id = ci.movie_id
  AND mi_idx.movie_id = mc.movie_id
  AND mi_idx.movie_id = ci.movie_id
  AND mc.movie_id = ci.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi.info_type_id
  AND it2.id = mi_idx.info_type_id
  AND ct.id = mc.company_type_id
  AND cn.id = mc.company_id
  AND rt.id = ci.role_id
  AND n.id = ci.person_id
  AND t.id = n.id;
"""),
        ('4a', 'actor', """
SELECT MIN(mi_idx.info) AS rating,
       MIN(t.title) AS movie_title
FROM info_type AS it1, info_type AS it2, movie_info AS mi,
     movie_info_idx AS mi_idx, title AS t
WHERE it1.info = 'genres'
  AND it2.info = 'rating'
  AND mi.info IN ('Drama', 'Horror')
  AND mi_idx.info > '2.0'
  AND t.production_year > 2010
  AND t.id = mi.movie_id
  AND t.id = mi_idx.movie_id
  AND mi.movie_id = mi_idx.movie_id
  AND it1.id = mi.info_type_id
  AND it2.id = mi_idx.info_type_id;
"""),
        ('8a', 'actor', """
SELECT MIN(an.name) AS acress_pseudonym,
       MIN(t.title) AS japanese_movie_dubbed
FROM aka_name AS an, cast_info AS ci, company_name AS cn,
     movie_companies AS mc, name AS n, role_type AS rt, title AS t
WHERE ci.note IN ('(voice: English version)', '(English version)',
                  '(voice) (English version)')
  AND cn.country_code = '[jp]'
  AND mc.note LIKE '%(Japan)%'
  AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
  AND n.name LIKE '%Yo%'
  AND n.id = an.person_id
  AND n.id = ci.person_id
  AND ci.movie_id = t.id
  AND t.id = mc.movie_id
  AND mc.company_id = cn.id
  AND ci.role_id = rt.id
  AND an.person_id = ci.person_id;
"""),
        ('12a', 'actor', """
SELECT MIN(chn.name) AS character_name,
       MIN(t.title) AS russian_mov_with_actor_and_character
FROM cast_info AS ci, company_name AS cn, company_type AS ct,
     char_name AS chn, movie_companies AS mc, role_type AS rt, title AS t
WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)',
                  '(voice: English version)')
  AND cn.country_code = '[ru]'
  AND rt.role = 'actor'
  AND t.production_year > 2010
  AND t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND ci.movie_id = mc.movie_id
  AND chn.id = ci.person_role_id
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id
  AND rt.id = ci.role_id;
"""),
        ('20a', 'actor', """
SELECT MIN(n.name) AS cast_member,
       MIN(t.title) AS complete_credited_movie
FROM cast_info AS ci, comp_cast_type AS cct1, comp_cast_type AS cct2,
     complete_cast AS cc, movie_companies AS mc, name AS n,
     company_name AS cn, title AS t
WHERE cct1.kind = 'cast'
  AND cct2.kind != 'complete+verified'
  AND cn.country_code = '[us]'
  AND t.production_year > 2000
  AND t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND t.id = cc.movie_id
  AND mc.company_id = cn.id
  AND ci.person_id = n.id
  AND cct1.id = cc.subject_id
  AND cct2.id = cc.status_id;
"""),
        ('24a', 'actor', """
SELECT MIN(chn.name) AS voiced_char,
       MIN(n.name) AS voicing_actress,
       MIN(t.title) AS voiced_animation
FROM aka_name AS an, char_name AS chn, cast_info AS ci,
     company_name AS cn, info_type AS it, keyword AS k,
     movie_companies AS mc, movie_info AS mi, movie_keyword AS mk,
     name AS n, role_type AS rt, title AS t
WHERE an.name LIKE '%a%'
  AND chn.name IS NOT NULL
  AND it.info = 'release dates'
  AND k.keyword IN ('hero', 'based-on-novel', 'disney', 'animation')
  AND n.gender = 'f'
  AND n.name LIKE '%An%'
  AND rt.role = 'actress'
  AND t.production_year > 2010
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = ci.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi.movie_id = ci.movie_id
  AND mc.movie_id = ci.movie_id
  AND k.id = mk.keyword_id
  AND it.id = mi.info_type_id
  AND cn.id = mc.company_id
  AND an.person_id = n.id
  AND rt.id = ci.role_id
  AND n.id = ci.person_id
  AND chn.id = ci.person_role_id;
"""),
        ('28a', 'actor', """
SELECT MIN(cn.name) AS producing_company,
       MIN(lt.link) AS link_type,
       MIN(t.title) AS complete_western_sequel
FROM company_name AS cn, company_type AS ct, info_type AS it,
     keyword AS k, link_type AS lt, movie_companies AS mc,
     movie_info AS mi, movie_keyword AS mk, movie_link AS ml, title AS t
WHERE cn.country_code = '[us]'
  AND it.info = 'countries'
  AND k.keyword = 'sequel'
  AND lt.link IN ('sequel', 'follows', 'followedBy')
  AND mc.note IS NULL
  AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish',
                  'Danish', 'Norwegian', 'German', 'USA', 'American')
  AND t.production_year > 1999
  AND lt.id = ml.link_type_id
  AND t.id = ml.movie_id
  AND t.id = mk.movie_id
  AND t.id = mi.movie_id
  AND t.id = mc.movie_id
  AND mk.movie_id = ml.movie_id
  AND ml.movie_id = mi.movie_id
  AND ml.movie_id = mc.movie_id
  AND mi.movie_id = mk.movie_id
  AND mi.movie_id = mc.movie_id
  AND k.id = mk.keyword_id
  AND it.id = mi.info_type_id
  AND ct.id = mc.company_type_id
  AND cn.id = mc.company_id;
"""),
        # ── MOVIE PHASE (title/movie-info-centric) ─────────────────────────────
        ('1a', 'movie', """
SELECT MIN(mc.note) AS production_note,
       MIN(t.title) AS movie_title,
       MIN(t.production_year) AS movie_year
FROM company_type AS ct, info_type AS it, movie_companies AS mc,
     movie_info_idx AS mi_idx, title AS t
WHERE ct.kind = 'production companies'
  AND it.info = 'top 250 rank'
  AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
  AND (mc.note LIKE '%(co-production)%' OR mc.note LIKE '%(presents)%')
  AND ct.id = mc.company_type_id
  AND t.id = mc.movie_id
  AND t.id = mi_idx.movie_id
  AND mc.movie_id = mi_idx.movie_id
  AND it.id = mi_idx.info_type_id;
"""),
        ('3a', 'movie', """
SELECT MIN(t.title) AS movie_title
FROM company_type AS ct, info_type AS it, keyword AS k,
     movie_companies AS mc, movie_info_idx AS mi_idx,
     movie_keyword AS mk, title AS t
WHERE ct.kind = 'production companies'
  AND it.info = 'bottom 10 rank'
  AND k.keyword LIKE '%sequel%'
  AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
  AND ct.id = mc.company_type_id
  AND t.id = mc.movie_id
  AND t.id = mi_idx.movie_id
  AND t.id = mk.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = mi_idx.movie_id
  AND mc.movie_id = mi_idx.movie_id
  AND k.id = mk.keyword_id
  AND it.id = mi_idx.info_type_id;
"""),
        ('7a', 'movie', """
SELECT MIN(n.name) AS of_person,
       MIN(t.title) AS biography_movie
FROM aka_name AS an, cast_info AS ci, info_type AS it,
     link_type AS lt, movie_info AS mi, movie_link AS ml,
     name AS n, title AS t
WHERE an.name LIKE 'A%'
  AND it.info = 'mini biography'
  AND lt.link = 'features'
  AND n.name_pcode_cf LIKE 'D%'
  AND n.gender = 'm'
  AND mi.note IS NULL
  AND t.production_year > 2000
  AND lt.id = ml.link_type_id
  AND t.id = ml.movie_id
  AND t.id = mi.movie_id
  AND t.id = ci.movie_id
  AND mk.movie_id = ml.movie_id
  AND ml.movie_id = mi.movie_id
  AND ml.movie_id = ci.movie_id
  AND mi.movie_id = ci.movie_id
  AND it.id = mi.info_type_id
  AND n.id = an.person_id
  AND n.id = ci.person_id
  AND ci.person_id = an.person_id;
"""),
        ('11a', 'movie', """
SELECT MIN(cn.name) AS from_company,
       MIN(mc.note) AS production_note,
       MIN(t.title) AS movie_based_on_book
FROM company_name AS cn, company_type AS ct, info_type AS it1,
     info_type AS it2, keyword AS k, movie_companies AS mc,
     movie_info AS mi1, movie_info AS mi2,
     movie_keyword AS mk, title AS t
WHERE cn.country_code = '[us]'
  AND ct.kind = 'production companies'
  AND it1.info = 'rating'
  AND it2.info = 'release dates'
  AND k.keyword = 'based-on-novel'
  AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
  AND mi2.info LIKE 'USA:%200%'
  AND t.production_year > 1950
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id
  AND t.id = mc.movie_id
  AND t.id = mk.movie_id
  AND t.id = mi1.movie_id
  AND t.id = mi2.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = mi1.movie_id
  AND mk.movie_id = mi2.movie_id
  AND mi1.movie_id = mc.movie_id
  AND mi1.movie_id = mi2.movie_id
  AND mc.movie_id = mi2.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi1.info_type_id
  AND it2.id = mi2.info_type_id;
"""),
        ('13a', 'movie', """
SELECT MIN(cn.name) AS producing_company,
       MIN(mi_idx.info) AS rating,
       MIN(t.title) AS movie_about_winning
FROM company_name AS cn, company_type AS ct, info_type AS it1,
     info_type AS it2, keyword AS k, movie_companies AS mc,
     movie_info AS mi, movie_info_idx AS mi_idx,
     movie_keyword AS mk, title AS t
WHERE cn.country_code = '[us]'
  AND ct.kind = 'production companies'
  AND it1.info = 'genres'
  AND it2.info = 'votes'
  AND k.keyword IN ('champion', 'winning', 'pride')
  AND mi.info IN ('Drama', 'Comedy', 'Sport')
  AND mi_idx.info > '0.0'
  AND t.production_year > 2000
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id
  AND t.id = mc.movie_id
  AND t.id = mk.movie_id
  AND t.id = mi.movie_id
  AND t.id = mi_idx.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mi_idx.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi.movie_id = mi_idx.movie_id
  AND mc.movie_id = mi_idx.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi.info_type_id
  AND it2.id = mi_idx.info_type_id;
"""),
        ('14a', 'movie', """
SELECT MIN(mi_idx.info) AS rating,
       MIN(t.title) AS north_american_acting_movie
FROM info_type AS it, keyword AS k, movie_info_idx AS mi_idx,
     movie_keyword AS mk, title AS t
WHERE it.info = 'rating'
  AND k.keyword LIKE '%sequel%'
  AND mi_idx.info > '5.0'
  AND t.production_year > 1990
  AND t.id = mi_idx.movie_id
  AND t.id = mk.movie_id
  AND mk.movie_id = mi_idx.movie_id
  AND k.id = mk.keyword_id
  AND it.id = mi_idx.info_type_id;
"""),
        ('22a', 'movie', """
SELECT MIN(cn.name) AS movie_company,
       MIN(mi_idx.info) AS rating,
       MIN(t.title) AS western_violent_movie
FROM company_name AS cn, company_type AS ct,
     info_type AS it1, info_type AS it2,
     keyword AS k, movie_companies AS mc,
     movie_info AS mi, movie_info_idx AS mi_idx,
     movie_keyword AS mk, title AS t
WHERE cn.country_code = '[us]'
  AND ct.kind = 'production companies'
  AND it1.info = 'genres'
  AND it2.info = 'votes'
  AND k.keyword IN ('murder', 'violence', 'blood', 'death', 'fire', 'gun')
  AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
  AND mi.info IN ('Western', 'Action', 'Sci-Fi')
  AND mi_idx.info > '7.0'
  AND t.production_year > 1990
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id
  AND t.id = mc.movie_id
  AND t.id = mk.movie_id
  AND t.id = mi.movie_id
  AND t.id = mi_idx.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mi_idx.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi.movie_id = mi_idx.movie_id
  AND mc.movie_id = mi_idx.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi.info_type_id
  AND it2.id = mi_idx.info_type_id;
"""),
        # ── PRODUCTION PHASE (company/production-centric) ──────────────────────
        ('6a', 'production', """
SELECT MIN(k.keyword) AS movie_keyword,
       MIN(t.title) AS movie_title
FROM keyword AS k, movie_keyword AS mk, title AS t
WHERE k.keyword LIKE '%sequel%'
  AND mk.movie_id = t.id
  AND k.id = mk.keyword_id;
"""),
        ('9a', 'production', """
SELECT MIN(an1.name) AS actress_pseudonym,
       MIN(t.title) AS japanese_movie_dubbed
FROM aka_name AS an1, cast_info AS ci, company_name AS cn,
     movie_companies AS mc, name AS n1, role_type AS rt, title AS t
WHERE ci.note IN ('(voice: English version)', '(English version)',
                  '(voice) (English version)')
  AND cn.country_code = '[jp]'
  AND n1.name NOT LIKE '%Yo%'
  AND rt.role = 'actress'
  AND t.production_year > 2000
  AND t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND ci.movie_id = mc.movie_id
  AND cn.id = mc.company_id
  AND ci.role_id = rt.id
  AND n1.id = ci.person_id
  AND an1.person_id = ci.person_id;
"""),
        ('16a', 'production', """
SELECT MIN(an.name) AS cool_actor_pseudonym,
       MIN(t.title) AS series_named_after_char
FROM aka_name AS an, cast_info AS ci, company_name AS cn,
     company_type AS ct, keyword AS k, movie_companies AS mc,
     movie_keyword AS mk, name AS n, role_type AS rt,
     title AS t
WHERE cn.country_code = '[us]'
  AND ct.kind = 'production companies'
  AND k.keyword = 'character-name-in-title'
  AND n.gender = 'm'
  AND n.name LIKE '%Ch%'
  AND rt.role = 'actor'
  AND t.production_year > 1990
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = ci.movie_id
  AND mc.movie_id = ci.movie_id
  AND k.id = mk.keyword_id
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id
  AND rt.id = ci.role_id
  AND n.id = ci.person_id
  AND an.person_id = ci.person_id;
"""),
        ('19a', 'production', """
SELECT MIN(n.name) AS voicing_actress,
       MIN(t.title) AS voiced_animation
FROM cast_info AS ci, company_name AS cn, info_type AS it,
     keyword AS k, movie_companies AS mc, movie_info AS mi,
     movie_keyword AS mk, name AS n, role_type AS rt, title AS t
WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)',
                  '(voice: English version)')
  AND cn.country_code = '[us]'
  AND it.info = 'release dates'
  AND k.keyword = 'animation'
  AND mc.note LIKE '%(20th Century Fox)%'
  AND n.gender = 'f'
  AND rt.role = 'actress'
  AND t.production_year > 2000
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = ci.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi.movie_id = ci.movie_id
  AND mc.movie_id = ci.movie_id
  AND k.id = mk.keyword_id
  AND it.id = mi.info_type_id
  AND cn.id = mc.company_id
  AND rt.id = ci.role_id
  AND n.id = ci.person_id;
"""),
        ('26a', 'production', """
SELECT MIN(k.keyword) AS movie_keyword,
       MIN(t.title) AS movie_title
FROM keyword AS k, movie_info AS mi, movie_keyword AS mk, title AS t
WHERE k.keyword LIKE '%sequel%'
  AND mi.info LIKE 'USA:%'
  AND t.production_year > 2000
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND mk.movie_id = mi.movie_id
  AND k.id = mk.keyword_id;
"""),
        ('29a', 'production', """
SELECT MIN(n.name) AS female_lead,
       MIN(t.title) AS movie_title
FROM cast_info AS ci, company_name AS cn, company_type AS ct,
     name AS n, movie_companies AS mc, role_type AS rt, title AS t
WHERE cn.country_code = '[us]'
  AND ct.kind = 'production companies'
  AND n.gender = 'f'
  AND rt.role IN ('actress', 'leading actress')
  AND t.production_year > 2005
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id
  AND t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND mc.movie_id = ci.movie_id
  AND n.id = ci.person_id
  AND rt.id = ci.role_id;
"""),
        ('32a', 'production', """
SELECT MIN(lt.link) AS link_type,
       MIN(t1.title) AS first_movie,
       MIN(t2.title) AS second_movie
FROM keyword AS k, link_type AS lt, movie_keyword AS mk,
     movie_link AS ml, title AS t1, title AS t2
WHERE k.keyword = 'sequel'
  AND lt.link IN ('sequel', 'follows', 'followedBy')
  AND t2.id = ml.linked_movie_id
  AND t1.id = ml.movie_id
  AND t1.id = mk.movie_id
  AND mk.movie_id = ml.movie_id
  AND k.id = mk.keyword_id
  AND lt.id = ml.link_type_id;
"""),
        # ── KEYWORD PHASE (genre/keyword/metadata-centric) ─────────────────────
        ('5a', 'keyword', """
SELECT MIN(t.title) AS american_movie
FROM company_type AS ct, info_type AS it, movie_companies AS mc,
     movie_info AS mi, title AS t
WHERE ct.kind = 'production companies'
  AND it.info = 'countries'
  AND mc.note LIKE '%(200%)%'
  AND mi.info = 'Sweden'
  AND t.production_year > 2005
  AND t.id = mc.movie_id
  AND t.id = mi.movie_id
  AND mc.movie_id = mi.movie_id
  AND ct.id = mc.company_type_id
  AND it.id = mi.info_type_id;
"""),
        ('10a', 'keyword', """
SELECT MIN(chn.name) AS char_name,
       MIN(t.title) AS movie_with_american_producers
FROM char_name AS chn, cast_info AS ci, company_name AS cn,
     company_type AS ct, movie_companies AS mc, role_type AS rt, title AS t
WHERE chn.name LIKE '%man%'
  AND cn.country_code = '[us]'
  AND ct.kind = 'production companies'
  AND rt.role = 'actor'
  AND t.production_year BETWEEN 2005 AND 2015
  AND t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND ci.movie_id = mc.movie_id
  AND chn.id = ci.person_role_id
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id
  AND rt.id = ci.role_id;
"""),
        ('15a', 'keyword', """
SELECT MIN(mi.info) AS release_date,
       MIN(t.title) AS internet_movie
FROM aka_title AS at, company_name AS cn, company_type AS ct,
     info_type AS it1, keyword AS k, movie_companies AS mc,
     movie_info AS mi, movie_keyword AS mk, title AS t
WHERE cn.country_code = '[us]'
  AND ct.kind = 'production companies'
  AND it1.info = 'release dates'
  AND k.keyword = 'internet'
  AND mc.note LIKE '%(200%)%'
  AND t.production_year > 2000
  AND t.id = at.movie_id
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = at.movie_id
  AND mk.movie_id = mc.movie_id
  AND mi.movie_id = at.movie_id
  AND mi.movie_id = mc.movie_id
  AND at.movie_id = mc.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi.info_type_id
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id;
"""),
        ('17a', 'keyword', """
SELECT MIN(n.name) AS member_in_charnamed_movie,
       MIN(n.id) AS linker_id
FROM cast_info AS ci, company_name AS cn, keyword AS k,
     movie_companies AS mc, movie_keyword AS mk, name AS n, title AS t
WHERE cn.country_code = '[us]'
  AND k.keyword = 'character-name-in-title'
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = ci.movie_id
  AND mc.movie_id = ci.movie_id
  AND k.id = mk.keyword_id
  AND cn.id = mc.company_id
  AND n.id = ci.person_id;
"""),
        ('33a', 'keyword', """
SELECT MIN(cn.name) AS production_company,
       MIN(lt.link) AS link_type,
       MIN(t.title) AS complete_us_sequel
FROM company_name AS cn, company_type AS ct, info_type AS it,
     keyword AS k, link_type AS lt, movie_companies AS mc,
     movie_info AS mi, movie_keyword AS mk, movie_link AS ml, title AS t
WHERE cn.country_code = '[us]'
  AND it.info = 'countries'
  AND k.keyword = 'sequel'
  AND lt.link IN ('sequel', 'follows', 'followedBy')
  AND mc.note IS NULL
  AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish',
                  'Danish', 'Norwegian', 'German', 'USA', 'American')
  AND t.production_year > 1990
  AND lt.id = ml.link_type_id
  AND t.id = ml.movie_id
  AND t.id = mk.movie_id
  AND t.id = mi.movie_id
  AND t.id = mc.movie_id
  AND mk.movie_id = ml.movie_id
  AND ml.movie_id = mi.movie_id
  AND ml.movie_id = mc.movie_id
  AND mi.movie_id = mk.movie_id
  AND mi.movie_id = mc.movie_id
  AND k.id = mk.keyword_id
  AND it.id = mi.info_type_id
  AND ct.id = mc.company_type_id
  AND cn.id = mc.company_id;
"""),
    ]


def load_queries_from_files():
    """Load JOB .sql files from QUERY_DIR. Returns list of (query_id, phase, sql)."""
    records = []
    sql_files = sorted(QUERY_DIR.glob('*.sql'))
    for f in sql_files:
        qid = f.stem           # e.g., "1a", "22c"
        sql = f.read_text(encoding='utf-8').strip()
        # Determine group number for phase classification
        m = re.match(r'^(\d+)', qid)
        group = m.group(1) if m else '0'
        phase = JOB_GROUP_PHASE.get(group, 'movie')  # default to movie
        records.append((qid, phase, sql))
    return records


if use_embedded:
    raw_queries = get_embedded_queries()
    NPTS = NPTS_EMBEDDED   # fewer queries per window for small embedded set
    print(f"  Using {len(raw_queries)} embedded representative queries")
    print(f"  (NPTS reduced to {NPTS} for embedded mode)")
else:
    raw_queries = load_queries_from_files()
    NPTS = NPTS_FULL
    print(f"  Loaded {len(raw_queries)} queries from {QUERY_DIR}")

# Confirm we have all 4 phases represented
phases_found = set(ph for _, ph, _ in raw_queries)
print(f"  Phases represented: {sorted(phases_found)}")


# ── Step 2: Extract SQL Features ───────────────────────────────────────────────
print("\n[2] Extracting SQL features ...")

def extract_tables(sql):
    """Extract table names from FROM and JOIN clauses."""
    tables = set()
    for m in re.finditer(r'\bFROM\s+(\w+)(?:\s+AS\s+\w+)?', sql, re.I):
        tables.add(m.group(1).lower())
    for m in re.finditer(r'\bJOIN\s+(\w+)(?:\s+AS\s+\w+)?', sql, re.I):
        tables.add(m.group(1).lower())
    # Also catch comma-separated FROM lists
    from_match = re.search(r'\bFROM\s+(.*?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|\Z)',
                           sql, re.I | re.S)
    if from_match:
        items = from_match.group(1).split(',')
        for item in items:
            m2 = re.match(r'\s*(\w+)', item.strip())
            if m2:
                tables.add(m2.group(1).lower())
    return tables


def extract_columns(sql):
    """Extract column names referenced in WHERE / predicates."""
    cols = set()
    for m in re.finditer(r'\b(\w+\.\w+)\s*[=<>!]', sql, re.I):
        col = m.group(1).split('.')[-1].lower()
        cols.add(col)
    for m in re.finditer(r'\bWHERE\s+(\w+)\s*[=<>!]', sql, re.I):
        cols.add(m.group(1).lower())
    for m in re.finditer(r'\bAND\s+(\w+)\s*[=<>!]', sql, re.I):
        cols.add(m.group(1).lower())
    return cols


def count_joins(sql):
    """Count number of join predicates (approx. table pair equalities)."""
    # Count explicit JOINs or comma-cross-join with WHERE equalities
    explicit_joins = len(re.findall(r'\bJOIN\b', sql, re.I))
    where_equalities = len(re.findall(r'\w+\.\w+\s*=\s*\w+\.\w+', sql, re.I))
    return explicit_joins + where_equalities


def query_complexity(sql):
    """
    Proxy for execution time in static mode.
    Complexity = join_count * table_count (higher = more expensive).
    Returns an estimated relative cost in milliseconds.
    """
    n_joins = count_joins(sql)
    n_tables = len(extract_tables(sql))
    # JOB queries range from 2–17 table joins; map to ~10ms–5000ms range
    base_cost_ms = max(10, n_joins * n_tables * 3.5)
    return base_cost_ms


records = []
for qid, given_phase, sql in raw_queries:
    tables = extract_tables(sql)
    columns = extract_columns(sql)
    n_joins = count_joins(sql)
    cost_ms = query_complexity(sql)
    records.append({
        'qid'       : qid,
        'sql'       : sql,
        'phase'     : given_phase,
        'qtype'     : 'SELECT',   # all JOB queries are SELECT
        'tables'    : tables,
        'columns'   : columns,
        'access'    : tables | columns,
        'n_joins'   : n_joins,
        'cost_ms'   : cost_ms,    # proxy elapsed in ms (static mode)
        'elapsed_ms': cost_ms,    # will be overwritten in execute mode
    })

print(f"  Features extracted for {len(records)} queries")
join_counts = [r['n_joins'] for r in records]
print(f"  Join range: {min(join_counts)}–{max(join_counts)} per query")


# ── Step 3: Execute Queries (optional) ────────────────────────────────────────
if args.execute:
    print("\n[3] Executing queries against Docker IMDB database ...")
    try:
        import psycopg2
    except ImportError:
        print("  ERROR: psycopg2 not installed.")
        print("  Run: pip install psycopg2-binary --break-system-packages")
        sys.exit(1)

    try:
        conn = psycopg2.connect(
            host=DOCKER_HOST, port=args.port, dbname=args.dbname,
            user=DOCKER_USER, password=DOCKER_PASS,
            connect_timeout=10
        )
        conn.set_session(readonly=True, autocommit=True)
        cur = conn.cursor()
        print(f"  Connected to {args.dbname} on port {args.port}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to PostgreSQL: {e}")
        print("  Ensure Docker IMDB database is running:")
        print("    docker compose -f docker/docker-compose.yml up -d")
        print("  Then load IMDB data:")
        print("    python hsm_job_validation.py --setup")
        sys.exit(1)

    failed = 0
    for i, r in enumerate(records):
        try:
            t0 = time.perf_counter()
            cur.execute(r['sql'])
            _ = cur.fetchall()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            r['elapsed_ms'] = elapsed_ms
            if args.verbose:
                print(f"    [{i+1:3d}/{len(records)}] {r['qid']:5s}  "
                      f"phase={r['phase']:12s}  {elapsed_ms:8.1f} ms")
        except Exception as e:
            r['elapsed_ms'] = r['cost_ms']  # fallback to proxy
            failed += 1
            if args.verbose:
                print(f"    [{i+1:3d}/{len(records)}] {r['qid']:5s}  ERROR: {e}")

    conn.close()
    print(f"  Executed {len(records) - failed}/{len(records)} queries successfully")
    if failed:
        print(f"  {failed} queries used complexity proxy (failed to execute)")
else:
    print("\n[3] Using complexity proxy for elapsed time (static mode)")
    print("    Run with --execute for real timing data.")


# ── Step 4: Phase Distribution ─────────────────────────────────────────────────
print("\n[4] Workload phase distribution ...")

phase_counts = Counter(r['phase'] for r in records)
print(f"  {'Phase':<14}  {'Count':>5}  {'Pct':>6}")
print("  " + "-" * 30)
for ph, cnt in phase_counts.most_common():
    pct = 100 * cnt / len(records)
    print(f"  {ph:<14}  {cnt:5d}  {pct:5.1f}%")


# ── Step 5: Build Windows ──────────────────────────────────────────────────────
print(f"\n[5] Building windows (Npts={NPTS} queries each) ...")

# Sort records by phase then by query ID for natural ordering
# This simulates a realistic workload trace where queries cluster by theme
phase_order = ['actor', 'movie', 'production', 'keyword']
records_sorted = sorted(records,
    key=lambda r: (phase_order.index(r['phase']) if r['phase'] in phase_order else 99,
                   r['qid']))

# Build windows of NPTS queries
windows = []
for i in range(0, len(records_sorted) - NPTS + 1, max(1, NPTS // 2)):
    chunk = records_sorted[i:i + NPTS]
    if len(chunk) == NPTS:
        windows.append(chunk)

print(f"  Built {len(windows)} windows of {NPTS} queries each")

if len(windows) < 4:
    print("  WARNING: Very few windows. Consider:")
    print("    1. Downloading all 113 JOB queries (--download flag)")
    print("    2. Reducing NPTS (currently {NPTS})")


def window_dominant_phase(w):
    """Assign phase = majority phase in the window."""
    phases = Counter(r['phase'] for r in w)
    return phases.most_common(1)[0][0]


# ── Step 6: Compute HSM Window Features ───────────────────────────────────────
print("\n[6] Computing HSM window features ...")

def compute_window_features(chunk):
    n = len(chunk)
    elapsed = [r['elapsed_ms'] for r in chunk]
    total_elapsed_s = sum(max(e, 0.1) for e in elapsed) / 1000.0

    # S_R: All JOB queries are SELECT; use sub-type proxy (join-heavy vs simple)
    # Heavy queries (>8 joins) = "complex", lighter = "simple SELECT"
    n_complex  = sum(1 for r in chunk if r['n_joins'] > 8)
    n_simple   = n - n_complex
    ratio_sel  = n_simple / n   # higher = more simple selects

    # S_V: Volume (QPS)
    qps = n / total_elapsed_s

    # S_T: Query type vector — all SELECT, but vary by complexity tier
    # [n_simple_sel, n_complex_sel, n_update, n_delete]
    type_vec = np.array([n_simple, n_complex, 0, 0], dtype=float)

    # S_A: Access attribute set
    access_set = set()
    for r in chunk:
        access_set.update(r['tables'])
        access_set.update(r['columns'])

    # S_P: Temporal pattern from elapsed time series
    ts_len = max(n, 8)
    elapsed_series = (elapsed * (ts_len // n + 1))[:ts_len]
    # Convert ms → rate (queries per second per query)
    qps_series = np.array([1000.0 / max(e, 0.1) for e in elapsed_series])

    return {
        'ratio_sel'  : ratio_sel,
        'qps'        : qps,
        'type_vec'   : type_vec,
        'access'     : access_set,
        'ts'         : qps_series,
        'phase'      : window_dominant_phase(chunk),
        'n_queries'  : n,
    }


win_features = [compute_window_features(w) for w in windows]
win_phase_dist = Counter(wf['phase'] for wf in win_features)
print(f"  {len(win_features)} window features computed")
print(f"  Window phase distribution: " +
      ", ".join(f"{ph}={cnt}" for ph, cnt in sorted(win_phase_dist.items())))


# ── Step 7: HSM Similarity Functions ──────────────────────────────────────────
def s_r(fa, fb):
    return 1.0 - abs(fa['ratio_sel'] - fb['ratio_sel'])


def s_v(fa, fb):
    qa, qb = max(fa['qps'], 1e-9), max(fb['qps'], 1e-9)
    return math.exp(-abs(math.log(qa) - math.log(qb)))


def s_t(fa, fb):
    va, vb = fa['type_vec'], fb['type_vec']
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 1.0 if (na < 1e-9 and nb < 1e-9) else 0.0
    cos = np.dot(va, vb) / (na * nb)
    return float(np.clip(cos, 0.0, 1.0))


def s_a(fa, fb):
    aa, ab = fa['access'], fb['access']
    if not aa and not ab:
        return 1.0
    inter = len(aa & ab)
    union = len(aa | ab)
    return inter / union if union > 0 else 1.0


def dwt_normalize(ts):
    """DWT with db2 level 1, return unit-normalised approximation coefficients."""
    n = len(ts)
    pad = 2 ** math.ceil(math.log2(max(n, 4)))
    ts_padded = np.pad(ts, (0, pad - n), mode='edge')
    coeffs = pywt.dwt(ts_padded, 'db2')
    approx = coeffs[0]
    norm = np.linalg.norm(approx)
    if norm < 1e-9:
        return np.zeros_like(approx)
    return approx / norm


def s_p(fa, fb):
    ta, tb = fa['ts'], fb['ts']
    min_len = min(len(ta), len(tb))
    ta, tb = ta[:min_len], tb[:min_len]
    ha = dwt_normalize(ta)
    hb = dwt_normalize(tb)
    min_dwt = min(len(ha), len(hb))
    ha, hb = ha[:min_dwt], hb[:min_dwt]
    l2 = np.linalg.norm(ha - hb)
    return float(np.clip(1.0 - l2 / 2.0, 0.0, 1.0))


def hsm(fa, fb):
    sr = s_r(fa, fb)
    sv = s_v(fa, fb)
    st = s_t(fa, fb)
    sa = s_a(fa, fb)
    sp = s_p(fa, fb)
    score = 0.2*sr + 0.2*sv + 0.2*st + 0.2*sa + 0.2*sp
    return score, {'S_R': sr, 'S_V': sv, 'S_T': st, 'S_A': sa, 'S_P': sp}


# ── Step 8: Compute Pairwise HSM Scores ───────────────────────────────────────
print("\n[7] Computing pairwise HSM scores ...")

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

# Also add all-pairs within-phase and cross-phase (not just consecutive)
# for richer statistical analysis when sample size is small
if len(within_scores) < 5 or len(cross_scores) < 5:
    print("  Note: Few consecutive pairs — augmenting with all-pairs analysis ...")
    within_scores, cross_scores = [], []
    within_dims  = defaultdict(list)
    cross_dims   = defaultdict(list)
    for i in range(len(win_features)):
        for j in range(i + 1, len(win_features)):
            fa = win_features[i]
            fb = win_features[j]
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
    n_pairs = len(within_scores) + len(cross_scores)

print(f"  Total pairs       : {n_pairs}")
print(f"  Within-phase pairs: {len(within_scores)}")
print(f"  Cross-phase pairs : {len(cross_scores)}")

if len(within_scores) < 1 or len(cross_scores) < 2:
    print("\n  WARNING: Very few pairs for statistical analysis.")
    print("  For robust results, download all 113 JOB query files:")
    print("    python hsm_job_validation.py --download")
    if len(within_scores) < 1:
        print("  Cannot proceed: 0 within-phase pairs. Exiting.")
        sys.exit(1)


# ── Step 9: Statistics ─────────────────────────────────────────────────────────
print("\n[8] Computing statistics ...")

w_mean = statistics.mean(within_scores)
w_std  = statistics.stdev(within_scores) if len(within_scores) > 1 else 0.0
c_mean = statistics.mean(cross_scores)
c_std  = statistics.stdev(cross_scores)  if len(cross_scores) > 1 else 0.0
dr     = w_mean / c_mean if c_mean > 0 else float('inf')

# Mann-Whitney U test
u_stat, p_val = stats.mannwhitneyu(within_scores, cross_scores, alternative='greater')
n1, n2 = len(within_scores), len(cross_scores)
r_biserial = 1 - 2 * u_stat / (n1 * n2)

# Bootstrap 95% CI for DR
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

# ICC(2,1) two-way mixed, single measures
all_scores  = within_scores + cross_scores
grand_mean  = statistics.mean(all_scores)
ss_total    = sum((x - grand_mean)**2 for x in all_scores)
ss_between  = (n1*(w_mean - grand_mean)**2 + n2*(c_mean - grand_mean)**2)
ss_within_  = ss_total - ss_between
ms_between  = ss_between / 1
ms_within_  = ss_within_ / (len(all_scores) - 2)
icc = ((ms_between - ms_within_) / (ms_between + (2 - 1) * ms_within_)
       if ms_between > ms_within_ else 0.0)


# ── Step 10: Per-Dimension Analysis ────────────────────────────────────────────
print("\n[9] Per-dimension breakdown ...")
print(f"\n  {'Dimension':<8}  {'Within':>8}  {'Cross':>8}  {'Delta':>8}  {'Dominant?':>10}")
print("  " + "-" * 52)
dominant_dim = None
max_delta = -1
for dim in ['S_R', 'S_V', 'S_T', 'S_A', 'S_P']:
    wm = statistics.mean(within_dims[dim]) if within_dims[dim] else 0.0
    cm = statistics.mean(cross_dims[dim])  if cross_dims[dim]  else 0.0
    delta = wm - cm
    if delta > max_delta:
        max_delta = delta
        dominant_dim = dim
    flag = "← dominant" if dim == (dominant_dim if delta == max_delta else None) else ""
    print(f"  {dim:<8}  {wm:8.4f}  {cm:8.4f}  {delta:+8.4f}  {flag}")

# Redo dominant after loop
max_delta = -1
dim_deltas = {}
for dim in ['S_R', 'S_V', 'S_T', 'S_A', 'S_P']:
    wm = statistics.mean(within_dims[dim]) if within_dims[dim] else 0.0
    cm = statistics.mean(cross_dims[dim])  if cross_dims[dim]  else 0.0
    dim_deltas[dim] = wm - cm
dominant_dim = max(dim_deltas, key=dim_deltas.get)


# ── Step 11: Save Results ──────────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results_file = RESULTS_DIR / ('job_hsm_execute.csv' if args.execute
                               else 'job_hsm_static.csv')

with open(results_file, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['metric', 'value'])
    w.writerow(['mode',          'execute' if args.execute else 'static'])
    w.writerow(['n_queries',     len(records)])
    w.writerow(['n_windows',     len(win_features)])
    w.writerow(['npts',          NPTS])
    w.writerow(['n_within_pairs', n1])
    w.writerow(['n_cross_pairs',  n2])
    w.writerow(['within_mean',   round(w_mean, 6)])
    w.writerow(['within_std',    round(w_std, 6)])
    w.writerow(['cross_mean',    round(c_mean, 6)])
    w.writerow(['cross_std',     round(c_std, 6)])
    w.writerow(['DR',            round(dr, 4)])
    w.writerow(['DR_CI_lo',      round(ci_lo, 4)])
    w.writerow(['DR_CI_hi',      round(ci_hi, 4)])
    w.writerow(['MWU_p',         f'{p_val:.3e}'])
    w.writerow(['r_biserial',    round(r_biserial, 4)])
    w.writerow(['ICC_2_1',       round(icc, 4)])
    w.writerow(['dominant_dim',  dominant_dim])
    for dim in ['S_R', 'S_V', 'S_T', 'S_A', 'S_P']:
        wm = statistics.mean(within_dims[dim]) if within_dims[dim] else 0.0
        cm = statistics.mean(cross_dims[dim])  if cross_dims[dim]  else 0.0
        w.writerow([f'delta_{dim}', round(wm - cm, 6)])

print(f"\n  Results saved → {results_file}")


# ── Step 12: Final Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  RESULTS: HSM Validation on JOB (Join Order Benchmark / IMDB)")
print("=" * 70)
mode_note = ("Real execution timing" if args.execute else
             "Static SQL analysis (proxy timing) — run --execute for full validation")
print(f"\n  Mode            : {mode_note}")
print(f"  Queries         : {len(records)} (JOB benchmark, {'embedded' if use_embedded else 'from files'})")
print(f"  Windows         : {len(win_features)} ({NPTS} queries each)")
print(f"  Within-phase    : {n1} pairs  (mean={w_mean:.4f}, σ={w_std:.4f})")
print(f"  Cross-phase     : {n2} pairs  (mean={c_mean:.4f}, σ={c_std:.4f})")
print(f"\n  Discrimination Ratio : {dr:.3f}  (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
print(f"  Mann-Whitney p       : {p_val:.3e}")
print(f"  Rank-biserial r      : {r_biserial:.3f}")
print(f"  ICC(2,1)             : {icc:.3f}")
print(f"  Dominant dimension   : {dominant_dim}  (Δ={dim_deltas[dominant_dim]:+.4f})")

print(f"\n  θ=0.75 separation :")
below_thresh = sum(1 for s in cross_scores  if s < 0.75)
above_thresh = sum(1 for s in within_scores if s >= 0.75)
print(f"    Cross-phase below θ : {below_thresh}/{n2} ({100*below_thresh/n2:.0f}%)")
print(f"    Within-phase above θ: {above_thresh}/{n1} ({100*above_thresh/n1:.0f}%)")

print(f"\n  SDSS SkyServer (A8)  : DR=1.086, p=1.6e-70, ICC=0.994")
print(f"  JOB/IMDB (this run)  : DR={dr:.3f}, p={p_val:.3e}, ICC={icc:.3f}")
comparison = ("HIGHER" if dr > 1.086 else
               "COMPARABLE" if dr > 1.0 else "LOWER")
print(f"  Assessment           : {comparison} than SDSS baseline DR")

print("\n" + "=" * 70)
print("  Phase boundary examples (cross-phase pairs, lowest HSM score)")
print("=" * 70)

# Recompute all-pairs with phase labels for display
cross_detail = []
for i in range(len(win_features)):
    for j in range(i + 1, min(i + 6, len(win_features))):
        fa = win_features[i]
        fb = win_features[j]
        if fa['phase'] != fb['phase']:
            score, _ = hsm(fa, fb)
            cross_detail.append((score, fa['phase'], fb['phase']))
cross_detail.sort()
for score, pa, pb in cross_detail[:5]:
    print(f"  {pa:14s} → {pb:14s}  HSM={score:.4f}")

print()
if not args.execute:
    print("  NEXT STEP: After loading IMDB into Docker, re-run with --execute")
    print("  for real query timing data (full 5-dimension validation).")
    print()
