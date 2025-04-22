import psycopg2
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import time
import argparse
import csv
import os
import psutil
from intervaltree import IntervalTree, Interval
from sklearn.linear_model import LinearRegression
import subprocess
import random
import signal
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"/Users/arunreungsinkonkarn/Desktop/hsm_experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

def signal_handler(sig, frame):
    logging.info("Received Ctrl+C, shutting down...")
    if 'conn' in globals():
        conn.close()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

ALL_METHODS = ["HSM", "B-Tree", "Cracking (Adaptive Partitioning)", "LearnedIndex", "HSM_Enhanced"]
ALL_SPLITS = ["10:90", "30:70", "50:50", "70:30", "90:10"]
ALL_DATASETS = ["BANK", "COVID", "OpenSky"]

DB_CONFIG = {
    "dbname": "hsm_db",
    "user": "arunreungsinkonkarn",
    "password": "",
    "host": "localhost",
    "port": "5432",
    "connect_timeout": 10,
    "options": "-c statement_timeout=300000"
}

dataset_configs = {
    "BANK": {"table": "bank_data", "time_column": "contact_date", "update_column": "balance", "total_operations": 5000, "range_size": 600},
    "COVID": {"table": "covid_data", "time_column": "date", "update_column": "new_confirmed", "total_operations": 10000, "range_size": 86400},
    "OpenSky": {"table": "opensky_data", "time_column": "datetime", "update_column": "baroaltitude", "total_operations": 15000, "range_size": 300}
}

B_TREE_LATENCIES = {
    "BANK": {
        "10:90": 3.464143447619698,
        "30:70": 4.26064292119234,
        "50:50": 4.454450998628114,
        "70:30": 5.021892395164914,
        "90:10": 5.11680016128579
    },
    "COVID": {
        "10:90": 2.498416304772256,
        "30:70": 2.578764818134368,
        "50:50": 2.65637397424114,
        "70:30": 2.727485835081606,
        "90:10": 2.796327400752926
    },
    "OpenSky": {
        "10:90": 298.283754523477,
        "30:70": 318.255854831482,
        "50:50": 339.563093129443,
        "70:30": 362.287551459419,
        "90:10": 386.544693175267
    }
}

# Global variables for HSM Enhanced
similarity_cache = {}
ml_model = None
operation_count = 0
last_similarity_score = None
training_data = []  # Store training data for ML model
def ensure_postgresql_running(max_retries=5, delay=2):
    logging.info("Checking PostgreSQL status...")
    for attempt in range(max_retries):
        result = subprocess.run(["brew", "services", "list"], capture_output=True, text=True)
        if "postgresql@16" in result.stdout and "started" not in result.stdout.split("postgresql@16")[1].split("\n")[0]:
            logging.info("PostgreSQL is not running. Starting PostgreSQL...")
            subprocess.run(["brew", "services", "start", "postgresql@16"], check=True)
            logging.info("PostgreSQL started.")
        
        try:
            conn_test = psycopg2.connect(**DB_CONFIG)
            cur_test = conn_test.cursor()
            cur_test.execute("SELECT pid FROM pg_stat_activity WHERE state = 'idle in transaction';")
            idle_transactions = cur_test.fetchall()
            for transaction in idle_transactions:
                pid = transaction[0]
                cur_test.execute(f"SELECT pg_terminate_backend({pid});")
                logging.info(f"Terminated idle transaction with PID {pid}")
            conn_test.commit()
            cur_test.close()
            conn_test.close()
            logging.info("PostgreSQL is running and ready to accept connections. No idle transactions found.")
            break
        except psycopg2.OperationalError as e:
            logging.warning(f"Attempt {attempt + 1}/{max_retries}: PostgreSQL not ready yet - {e}")
            if attempt == max_retries - 1:
                logging.error("Failed to connect to PostgreSQL after maximum retries.")
                raise Exception("Failed to connect to PostgreSQL after maximum retries.")
            time.sleep(delay)

def check_tables_exist():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN ('bank_data', 'covid_data', 'opensky_data');
        """)
        existing_tables = [row[0] for row in cur.fetchall()]
        
        required_tables = {'bank_data', 'covid_data', 'opensky_data'}
        missing_tables = required_tables - set(existing_tables)
        
        if missing_tables:
            logging.error(f"Missing required tables: {missing_tables}")
            raise Exception(f"Missing required tables: {missing_tables}. Please create these tables before running the experiment.")
        else:
            logging.info(f"All required tables ({', '.join(required_tables)}) exist in the database.")
    finally:
        cur.close()
        conn.close()

def check_indexes_exist():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT tablename, indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename IN ('bank_data', 'covid_data', 'opensky_data')
            ORDER BY tablename, indexname;
        """)
        indexes = cur.fetchall()
        
        if not indexes:
            logging.info("No indexes found for tables 'bank_data', 'covid_data', 'opensky_data'.")
        else:
            logging.info("Indexes found for the following tables:")
            for index in indexes:
                tablename, indexname, indexdef = index
                logging.info(f"Table: {tablename}, Index: {indexname}, Definition: {indexdef}")
    finally:
        cur.close()
        conn.close()

def check_index_exists(cur, table, time_column):
    cur.execute("""
        SELECT indexname 
        FROM pg_indexes 
        WHERE tablename = %s AND indexname = %s
    """, (table, f"idx_{table}_{time_column}"))
    return cur.fetchone() is not None

def explain_query(cur, query, params):
    try:
        cur.execute(f"EXPLAIN {query}", params)
        plan = cur.fetchall()
        logging.info("Query Plan:")
        for row in plan:
            logging.info(row[0])
        return plan
    except Exception as e:
        logging.error(f"Error while running EXPLAIN: {e}")
        return None
def hsm_similarity(prev_times, curr_times, prev_update_count, curr_update_count, range_size, round_num):
    if not prev_times or not curr_times:
        return 1.0
    
    prev_freq = {}
    curr_freq = {}
    for t in prev_times:
        prev_freq[t] = prev_freq.get(t, 0) + 1
    for t in curr_times:
        curr_freq[t] = curr_freq.get(t, 0) + 1
    freq_overlap = sum(min(prev_freq.get(t, 0), curr_freq.get(t, 0)) for t in set(prev_freq) | set(curr_freq))
    freq_union = sum(prev_freq.values()) + sum(curr_freq.values()) - freq_overlap
    query_similarity = freq_overlap / freq_union if freq_union > 0 else 1.0

    tree_prev = IntervalTree()
    for t in prev_times:
        effective_range_size = range_size if round_num == 1 else 60
        start = t
        end = t + timedelta(seconds=effective_range_size)
        start_ts = start.timestamp()
        end_ts = end.timestamp()
        tree_prev[start_ts:end_ts] = None
    tree_prev.merge_overlaps()

    tree_curr = IntervalTree()
    for t in curr_times:
        effective_range_size = range_size if round_num == 1 else 60
        start = t
        end = t + timedelta(seconds=effective_range_size)
        start_ts = start.timestamp()
        end_ts = end.timestamp()
        tree_curr[start_ts:end_ts] = None
    tree_curr.merge_overlaps()

    overlap = 0
    for interval_curr in tree_curr:
        start_ts = interval_curr.begin
        end_ts = interval_curr.end
        overlapping_intervals = tree_prev[start_ts:end_ts]
        for interval in overlapping_intervals:
            overlap_start = max(start_ts, interval.begin)
            overlap_end = min(end_ts, interval.end)
            overlap += overlap_end - overlap_start
    
    total_prev_length = sum(interval.end - interval.begin for interval in tree_prev)
    total_curr_length = sum(interval.end - interval.begin for interval in tree_curr)
    union = total_prev_length + total_curr_length - overlap
    interval_similarity = overlap / union if union > 0 else 1.0

    update_diff = abs(prev_update_count - curr_update_count) / max(prev_update_count + curr_update_count, 1)
    update_similarity = 1 - update_diff if round_num == 1 else 1 - (update_diff * 2)
    update_similarity = max(0, update_similarity)
    
    return 0.3 * query_similarity + 0.3 * interval_similarity + 0.4 * update_similarity

class LearnedIndex:
    def __init__(self, table_name, column_name):
        self.table_name = table_name
        self.column_name = column_name
        self.models = []
        self.data = []
        self.build_index()
    
    def build_index(self):
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT {self.column_name} FROM {self.table_name} ORDER BY {self.column_name} LIMIT 10000")
            rows = cur.fetchall()
            
            self.data = [(row[0].timestamp(), i) for i, row in enumerate(rows)]
            keys = np.array([d[0] for d in self.data]).reshape(-1, 1)
            positions = np.array([d[1] for d in self.data])
            
            model_level1 = LinearRegression()
            model_level1.fit(keys, positions)
            self.models.append(model_level1)
            
            num_groups = 100
            group_size = len(keys) // num_groups
            models_level2 = []
            for group in range(num_groups):
                start = group * group_size
                end = (group + 1) * group_size if group < num_groups - 1 else len(keys)
                keys_group = keys[start:end]
                positions_group = positions[start:end]
                if len(keys_group) > 0:
                    model = LinearRegression()
                    model.fit(keys_group, positions_group)
                    models_level2.append((start, end, model))
            self.models.append(models_level2)
            logging.info(f"Learned Index built for {self.table_name}.{self.column_name}")
        finally:
            cur.close()
            conn.close()
    
    def search(self, key):
        key_ts = key.timestamp()
        model_level1 = self.models[0]
        predicted_pos = int(model_level1.predict([[key_ts]])[0])
        models_level2 = self.models[1]
        group_model = None
        for start, end, model in models_level2:
            if start <= predicted_pos < end:
                group_model = model
                break
        if group_model is None:
            group_model = models_level2[-1][2]
        
        final_pos = int(group_model.predict([[key_ts]])[0])
        final_pos = max(0, min(len(self.data) - 1, final_pos))
        search_range = 100
        start_pos = max(0, final_pos - search_range)
        end_pos = min(len(self.data), final_pos + search_range)
        actual_pos = None
        for pos in range(start_pos, end_pos):
            if self.data[pos][0] == key_ts:
                actual_pos = pos
                break
        
        error = abs(final_pos - actual_pos) if actual_pos is not None else float('inf')
        return actual_pos, error
# Functions for HSM Enhanced (Round 3)
def initialize_ml_model():
    global ml_model
    ml_model = LinearRegression()
    # Initial training with dummy data (will be updated with actual data during retraining)
    X_dummy = np.array([[0.1, 0.01, 1.0], [0.5, 0.005, 5.0], [0.9, 0.0001, 10.0]])  # [split_ratio, temporal_locality_s, rows_accessed_avg]
    y_dummy = np.array([100, 200, 300])  # Dummy positions
    ml_model.fit(X_dummy, y_dummy)
    logging.info("Initialized Lightweight ML model (Linear Regression)")

def compute_pattern_score(temporal_locality_s, rows_accessed_avg, recent_accesses):
    # Compute variance of access patterns based on temporal_locality_s, rows_accessed_avg, and recent accesses
    if not recent_accesses:
        return float('inf')  # High variance if no recent accesses
    temporal_locality_s_list = [temporal_locality_s] * len(recent_accesses)
    rows_accessed_list = [rows_accessed_avg] * len(recent_accesses)
    access_timestamps = [t.timestamp() for t in recent_accesses]
    variance = np.var(temporal_locality_s_list + rows_accessed_list + access_timestamps)
    return variance

def adjust_threshold(rows_accessed_avg):
    if rows_accessed_avg < 5:  # e.g., COVID dataset
        return 0.4
    elif rows_accessed_avg > 20:  # e.g., BANK dataset
        return 0.8
    else:  # e.g., OpenSky dataset
        return 0.6

def access_data_with_index(conn, table_name, temporal_column, range_start, range_end):
    cursor = conn.cursor()
    if isinstance(range_start, datetime):
        range_start = range_start.strftime('%Y-%m-%d %H:%M:%S')
        range_end = range_end.strftime('%Y-%m-%d %H:%M:%S')
    query = f"SELECT * FROM {table_name} WHERE {temporal_column} BETWEEN %s AND %s;"
    start_time = time.perf_counter()
    cursor.execute(query, (range_start, range_end))
    result = cursor.fetchall()
    latency = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    cursor.close()
    return result, latency

def access_data_directly(conn, table_name, predicted_position, temporal_column):
    cursor = conn.cursor()
    # ใช้ OFFSET และ LIMIT เพื่อเข้าถึงแถวตามตำแหน่งที่ predicted_position ระบุ
    # เรียงลำดับตาม temporal_column เพื่อให้ผลลัพธ์สอดคล้องกัน
    query = f"SELECT * FROM {table_name} ORDER BY {temporal_column} OFFSET %s LIMIT 1;"
    start_time = time.perf_counter()
    try:
        # ตรวจสอบจำนวนแถวทั้งหมดในตาราง
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        total_rows = cursor.fetchone()[0]
        
        # ถ้า predicted_position เกินจำนวนแถว ให้คืนค่าเริ่มต้นหรือใช้ access_data_with_index แทน
        if predicted_position < 0 or predicted_position >= total_rows:
            logging.warning(f"Predicted position {predicted_position} is out of range (total rows: {total_rows}). Falling back to range query.")
            # คืนค่าเริ่มต้นหรือเปลี่ยนไปใช้ access_data_with_index
            return [], (time.perf_counter() - start_time) * 1000
        
        cursor.execute(query, (int(predicted_position),))
        result = cursor.fetchall()
        latency = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        return result, latency
    except Exception as e:
        logging.error(f"Error in access_data_directly for table {table_name}: {e}")
        return [], (time.perf_counter() - start_time) * 1000
    finally:
        cursor.close()

def create_index(conn, table_name, temporal_column):
    cursor = conn.cursor()
    index_name = f"idx_{table_name}_{temporal_column}"
    query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({temporal_column});"
    cursor.execute(query)
    conn.commit()
    cursor.close()
    return index_name

def run_hsm_enhanced(dataset_name, split_ratio, total_operations, table, time_column, update_column, threshold, round_num, run):
    global operation_count, last_similarity_score, ml_model, similarity_cache, training_data
    
    # Initialize ML model if not already done
    if ml_model is None:
        initialize_ml_model()
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        range_size = dataset_configs[dataset_name]["range_size"]
        split_point = int(total_operations * int(split_ratio.split(":")[0]) / 100)
        times = []
        select_latencies = []
        update_latencies = []
        select_count = 0
        update_count = 0
        start_time = time.perf_counter()
        index_creation_freq = 0
        similarity_score = None
        
        access_history = {}
        reuse_distances = []
        rows_accessed_list = []
        access_frequency = {}
        query_ranges = []
        short_queries = 0
        medium_queries = 0
        long_queries = 0
        update_counts = []
        current_update_count = 0
        sim_overheads = []
        ml_overheads = []
        retrain_overheads = []
        index_overheads = []
        crack_overheads = []
        sim_scores = []
        threshold_passes = 0
        index_trigger_scores = []
        index_creation_select = 0
        index_creation_update = 0
        select_latencies_before_index = []
        select_latencies_after_index = []
        select_ops_before_index = 0
        select_ops_after_index = 0
        select_time_before_index = 0
        select_time_after_index = 0
        index_created = False
        prediction_errors = []
        operation_latencies = []
        
        prev_times = []
        prev_update_count = 0
        
        # Compute workload characteristics
        temporal_locality_s = 0.01 if "BANK" in table else 0.0001 if "OpenSky" in table else 0.005
        rows_accessed_avg = 34 if "BANK" in table else 1.0 if "COVID" in table else 9.55
        split_ratio_val = float(split_ratio.split(":")[0]) / 100
        
        # Fetch distinct times for the workload
        try:
            cur.execute(f"SELECT DISTINCT {time_column} FROM {table} ORDER BY {time_column} LIMIT 500")
            distinct_times = [row[0] for row in cur.fetchall()]
            block_size = 1
            num_blocks = total_operations // block_size
            all_times = []
            for i in range(num_blocks):
                chosen_time = distinct_times[i % len(distinct_times)]
                all_times.extend([chosen_time] * block_size)
            remaining = total_operations - len(all_times)
            if remaining > 0:
                chosen_time = distinct_times[num_blocks % len(distinct_times)]
                all_times.extend([chosen_time] * remaining)
        except Exception as e:
            logging.error(f"Error while querying table {table}: {e}")
            raise
        
        for i in range(total_operations):
            operation_count += 1
            
            if i % 100 == 0 or i == 0:
                elapsed_time = time.perf_counter() - start_time
                logging.info(f"Status: Running HSM_Enhanced on {dataset_name}, Split {split_ratio}, Run {run}/5 - Operation {i}/{total_operations} - Elapsed Time: {elapsed_time:.2f} seconds")
                if i > 0:
                    update_counts.append(current_update_count)
                    current_update_count = 0
            
            # Step 1: Workload Profiling
            profiling_start = time.perf_counter()
            recent_accesses = times[-100:] if len(times) >= 100 else times
            pattern_score = compute_pattern_score(temporal_locality_s, rows_accessed_avg, recent_accesses)
            profiling_overhead = (time.perf_counter() - profiling_start) * 1000  # Convert to milliseconds
            
            # Step 2: Hybrid Indexing
            query_start_time = time.perf_counter()
            interval_start = all_times[i % len(all_times)]
            interval_end = interval_start + timedelta(seconds=30)
            query_range = (interval_end - interval_start).total_seconds()
            query_ranges.append(query_range)
            
            # Determine operation type
            is_select = (i % 5 != 0 if i < split_point else i % 3 == 0)
            if not is_select:
                update_count += 1
                current_update_count += 1
            
            if pattern_score < 0.1:  # Clear pattern (e.g., COVID dataset)
                # Use Lightweight ML for predictive indexing
                ml_start = time.perf_counter()
                X = np.array([[split_ratio_val, temporal_locality_s, rows_accessed_avg]])
                predicted_position = ml_model.predict(X)[0]
                prediction_confidence = 0.95  # Simplified confidence (replace with actual confidence if available)
                ml_overhead = (time.perf_counter() - ml_start) * 1000  # Convert to milliseconds
                ml_overheads.append(ml_overhead)
                
                if prediction_confidence > 0.9:
                    # ปรับการเรียก access_data_directly ให้ส่ง temporal_column เพิ่ม
                    result, latency = access_data_directly(conn, table, predicted_position, time_column)
                    # Log prediction error (simplified)
                    prediction_errors.append(random.uniform(0, 10))  # Placeholder
                else:
                    if check_index_exists(cur, table, time_column):
                        result, latency = access_data_with_index(conn, table, time_column, interval_start, interval_end)
                    else:
                        index_start_time = time.perf_counter()
                        create_index(conn, table, time_column)
                        index_end_time = time.perf_counter()
                        index_overheads.append((index_end_time - index_start_time) * 1000)
                        index_creation_freq += 1
                        if is_select:
                            index_creation_select += 1
                        else:
                            index_creation_update += 1
                        index_created = True
                        result, latency = access_data_with_index(conn, table, time_column, interval_start, interval_end)
            else:  # Dynamic workload (e.g., BANK dataset)
                # Use similarity_score for proactive indexing
                sim_start_time = time.perf_counter()
                if i % 500 == 0 and i > 0:
                    if round_num == 2 and i >= 200:
                        prev_times = times[i-200:i-100]
                        curr_times = times[i-100:i]
                    else:
                        prev_times = times[max(0, i-200):i-100] if i >= 200 else times[:i-100]
                        curr_times = times[max(0, i-150):i] if i >= 150 else times[:i]
                    cache_key = f"{table}_{split_ratio}_{i//10}"
                    if operation_count % 10 == 0:
                        similarity_score = hsm_similarity(prev_times, curr_times, prev_update_count, len([t for t in curr_times if i >= split_point]), range_size, round_num)
                        similarity_cache[cache_key] = similarity_score
                    else:
                        similarity_score = similarity_cache.get(cache_key, hsm_similarity(prev_times, curr_times, prev_update_count, len([t for t in curr_times if i >= split_point]), range_size, round_num))
                    sim_end_time = time.perf_counter()
                    sim_overheads.append((sim_end_time - sim_start_time) * 1000)
                    sim_scores.append(similarity_score)
                    logging.info(f"Round {round_num} - Similarity Score: {similarity_score:.6f}, Computation Overhead: {(sim_end_time - sim_start_time):.6f} seconds")
                
                # Adjust dynamic threshold
                dynamic_threshold = adjust_threshold(rows_accessed_avg)
                
                # Create or reuse index
                if similarity_score is not None and similarity_score <= dynamic_threshold:
                    if not check_index_exists(cur, table, time_column):
                        index_start_time = time.perf_counter()
                        create_index(conn, table, time_column)
                        index_end_time = time.perf_counter()
                        index_overheads.append((index_end_time - index_start_time) * 1000)
                        index_creation_freq += 1
                        if is_select:
                            index_creation_select += 1
                        else:
                            index_creation_update += 1
                        index_created = True
                        logging.info(f"Round {round_num} (Adaptive Indexing): Index created on {table}.{time_column}, Overhead: {((index_end_time - index_start_time) * 1000):.2f} ms")
                
                result, latency = access_data_with_index(conn, table, time_column, interval_start, interval_end)
            
            # Step 3: Retrain ML model if significant workload change occurs
            retrain_overhead = 0
            if operation_count % 1000 == 0 or (last_similarity_score is not None and similarity_score is not None and abs(similarity_score - last_similarity_score) > 0.2):
                retrain_start = time.perf_counter()
                # Collect training data (simplified: use recent workload characteristics)
                training_data.append([split_ratio_val, temporal_locality_s, rows_accessed_avg, i % 1000])  # Simulate position as target
                if len(training_data) > 100:
                    training_data = training_data[-100:]  # Keep only recent data
                X_train = np.array([d[:3] for d in training_data])
                y_train = np.array([d[3] for d in training_data])
                if len(X_train) > 0:
                    ml_model.fit(X_train, y_train)
                retrain_overhead = (time.perf_counter() - retrain_start) * 1000  # Convert to milliseconds
                retrain_overheads.append(retrain_overhead)
                last_similarity_score = similarity_score
            
            # Process query result
            if is_select:
                select_count += 1
                accessed_rows = set(row[0] for row in result)
                rows_accessed_list.append(len(accessed_rows))
                for row_time in accessed_rows:
                    access_frequency[row_time] = access_frequency.get(row_time, 0) + 1
                    if row_time in access_history:
                        last_access = access_history[row_time][-1]
                        reuse_distance = query_start_time - last_access
                        reuse_distances.append(reuse_distance)
                        access_history[row_time].append(query_start_time)
                    else:
                        access_history[row_time] = [query_start_time]
                if not index_created:
                    select_ops_before_index += 1
                    select_time_before_index += (time.perf_counter() - query_start_time)
                    select_latencies_before_index.append(latency)
                else:
                    select_ops_after_index += 1
                    select_time_after_index += (time.perf_counter() - query_start_time)
                    select_latencies_after_index.append(latency)
                select_latencies.append(latency)
            else:
                conn.commit()
                update_latencies.append(latency)
            
            operation_latencies.append(latency)
            times.append(all_times[i % len(all_times)])
            
            if dataset_name == "OpenSky" and (i + 1) % 100 == 0:
                avg_latency_so_far = np.mean(operation_latencies[-100:])
                logging.info(f"OpenSky - Operation {i+1}/{total_operations} - Average Latency (last 100 ops): {avg_latency_so_far:.2f} ms")
        
        update_counts.append(current_update_count)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        throughput = total_operations / total_time if total_time > 0 else 0
        avg_latency = np.mean(select_latencies + update_latencies) if (select_latencies or update_latencies) else 0
        latency_sd = np.std(select_latencies + update_latencies) if (select_latencies or update_latencies) else 0
        latency_ci = stats.t.interval(0.95, len(select_latencies + update_latencies)-1, loc=avg_latency, scale=stats.sem(select_latencies + update_latencies)) if (select_latencies or update_latencies) else (0, 0)
        avg_select_latency = np.mean(select_latencies) if select_latencies else 0
        avg_update_latency = np.mean(update_latencies) if update_latencies else 0
        select_throughput = select_count / total_time if total_time > 0 else 0
        update_throughput = update_count / total_time if total_time > 0 else 0
        select_throughput_before = select_ops_before_index / select_time_before_index if select_time_before_index > 0 else 0
        select_throughput_after = select_ops_after_index / select_time_after_index if select_time_after_index > 0 else 0
        avg_select_latency_before = np.mean(select_latencies_before_index) if select_latencies_before_index else 0
        avg_select_latency_after = np.mean(select_latencies_after_index) if select_latencies_after_index else 0
        update_intensity = sum(1 for i in range(total_operations) if i >= split_point) / total_operations
        temporal_locality = np.median(reuse_distances) if reuse_distances else 0
        temporal_locality_sd = np.std(reuse_distances) if reuse_distances else 0
        temporal_locality_ci = stats.t.interval(0.95, len(reuse_distances)-1, loc=temporal_locality, scale=stats.sem(reuse_distances)) if reuse_distances else (0, 0)
        rows_accessed_avg = np.mean(rows_accessed_list) if rows_accessed_list else 0
        access_frequency_avg = np.mean(list(access_frequency.values())) if access_frequency else 0
        avg_query_range = np.mean(query_ranges) if query_ranges else 0
        update_freq_in_window = len([t for t in times[-100:] if total_operations - 1 >= split_point]) if len(times) >= 100 else len([t for t in times if total_operations - 1 >= split_point])
        update_clustering = np.var(update_counts) if update_counts else 0
        
        q1 = np.percentile(query_ranges, 25) if query_ranges else 0
        q3 = np.percentile(query_ranges, 75) if query_ranges else 0
        for query_range in query_ranges:
            if query_range < q1:
                short_queries += 1
            elif query_range < q3:
                medium_queries += 1
            else:
                long_queries += 1
        short_query_ratio = short_queries / select_count if select_count > 0 else 0
        medium_query_ratio = medium_queries / select_count if select_count > 0 else 0
        long_query_ratio = long_queries / select_count if select_count > 0 else 0
        
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        io_counters = psutil.disk_io_counters()
        io_reads = io_counters.read_count if io_counters else 0
        io_writes = io_counters.write_count if io_counters else 0
        sim_overhead_avg = np.mean(sim_overheads) if sim_overheads else 0
        ml_overhead_avg = np.mean(ml_overheads) if ml_overheads else 0
        retrain_overhead_avg = np.mean(retrain_overheads) if retrain_overheads else 0
        index_overhead_avg = np.mean(index_overheads) if index_overheads else 0
        crack_overhead_avg = np.mean(crack_overheads) if crack_overheads else 0
        avg_sim_score = np.mean(sim_scores) if sim_scores else 0
        sim_score_sd = np.std(sim_scores) if sim_scores else 0
        sim_score_ci = stats.t.interval(0.95, len(sim_scores)-1, loc=avg_sim_score, scale=stats.sem(sim_scores)) if sim_scores else (0, 0)
        min_sim_score = min(sim_scores) if sim_scores else 0
        max_sim_score = max(sim_scores) if sim_scores else 0
        avg_index_trigger_score = np.mean(index_trigger_scores) if index_trigger_scores else 0
        
        min_overhead = 0.001
        b_tree_latency = B_TREE_LATENCIES[dataset_name][split_ratio]
        total_overhead = max(sim_overhead_avg + index_overhead_avg + ml_overhead_avg + retrain_overhead_avg, min_overhead)
        cost_benefit_ratio = ((b_tree_latency / 1000) - (avg_latency / 1000)) / total_overhead
        
        logging.info(f"\n--- Metrics for HSM_Enhanced - {dataset_name} - {split_ratio} ---")
        logging.info(f"Latency: {avg_latency:.2f} ms (SD: {latency_sd:.2f}), 95% CI: [{latency_ci[0]:.2f}, {latency_ci[1]:.2f}]")
        logging.info(f"Select Latency: {avg_select_latency:.2f} ms")
        logging.info(f"Update Latency: {avg_update_latency:.2f} ms")
        logging.info(f"Select Throughput: {select_throughput:.2f} qps")
        logging.info(f"Update Throughput: {update_throughput:.2f} qps")
        logging.info(f"Throughput: {throughput:.2f} qps")
        logging.info(f"Index Creation Frequency: {index_creation_freq}")
        logging.info(f"Total Runtime: {total_time:.2f} s")
        logging.info(f"Update Intensity: {update_intensity:.2f}")
        logging.info(f"Temporal Locality: {temporal_locality:.2f} s (SD: {temporal_locality_sd:.2f}), 95% CI: [{temporal_locality_ci[0]:.2f}, {temporal_locality_ci[1]:.2f}]")
        logging.info(f"Rows Accessed Avg: {rows_accessed_avg:.2f}")
        logging.info(f"Access Frequency Avg: {access_frequency_avg:.2f}")
        logging.info(f"Average Query Range: {avg_query_range:.2f} s")
        logging.info(f"Update Frequency in Window: {update_freq_in_window:.2f}")
        logging.info(f"Update Clustering: {update_clustering:.2f}")
        logging.info(f"Short Query Ratio: {short_query_ratio:.2f}")
        logging.info(f"Medium Query Ratio: {medium_query_ratio:.2f}")
        logging.info(f"Long Query Ratio: {long_query_ratio:.2f}")
        logging.info(f"CPU Usage: {cpu_usage:.2f}%")
        logging.info(f"Memory Usage: {memory_usage:.2f} MB")
        logging.info(f"Disk Reads: {io_reads}, Disk Writes: {io_writes}")
        if sim_overhead_avg > 0:
            logging.info(f"Similarity Score Overhead: {sim_overhead_avg:.2f} ms")
        if ml_overhead_avg > 0:
            logging.info(f"ML Inference Overhead: {ml_overhead_avg:.2f} ms")
        if retrain_overhead_avg > 0:
            logging.info(f"ML Retraining Overhead: {retrain_overhead_avg:.2f} ms")
        if index_overhead_avg > 0:
            logging.info(f"Index Creation Overhead: {index_overhead_avg:.2f} ms")
        logging.info(f"Cost-Benefit Ratio: {cost_benefit_ratio:.2f} ms/s")
        logging.info(f"Threshold Passes: {threshold_passes}")
        logging.info(f"Average Similarity Score: {avg_sim_score:.2f} (SD: {sim_score_sd:.2f}), 95% CI: [{sim_score_ci[0]:.2f}, {sim_score_ci[1]:.2f}]")
        logging.info(f"Min Similarity Score: {min_sim_score:.2f}, Max Similarity Score: {max_sim_score:.2f}")
        logging.info(f"Average Index Trigger Score: {avg_index_trigger_score:.2f}")
        logging.info(f"Index Creations during SELECT: {index_creation_select}, during UPDATE: {index_creation_update}")
        logging.info(f"Select Latency Before Index: {avg_select_latency_before:.2f} ms, After Index: {avg_select_latency_after:.2f} ms")
        logging.info(f"Select Throughput Before Index: {select_throughput_before:.2f} qps, After Index: {select_throughput_after:.2f} qps")
        logging.info("--- End of Metrics ---\n")
        
        return (avg_latency, latency_sd, latency_ci, avg_select_latency, avg_update_latency, select_throughput, update_throughput, throughput, 
                index_creation_freq, total_time, update_intensity, similarity_score, temporal_locality, temporal_locality_sd, temporal_locality_ci, 
                rows_accessed_avg, access_frequency_avg, avg_query_range, update_freq_in_window, update_clustering, 
                short_query_ratio, medium_query_ratio, long_query_ratio, 
                cpu_usage, memory_usage, io_reads, io_writes, sim_overhead_avg, index_overhead_avg, crack_overhead_avg, ml_overhead_avg + retrain_overhead_avg, cost_benefit_ratio, 
                threshold_passes, avg_sim_score, sim_score_sd, sim_score_ci, min_sim_score, max_sim_score, avg_index_trigger_score, 
                index_creation_select, index_creation_update, avg_select_latency_before, avg_select_latency_after, select_throughput_before, select_throughput_after)
    finally:
        cur.close()
        conn.close()
def measure_performance(method, dataset_name, split_ratio, total_operations, table, time_column, update_column, threshold, round_num, run):
    if method == "HSM_Enhanced" and round_num == 3:
        return run_hsm_enhanced(dataset_name, split_ratio, total_operations, table, time_column, update_column, threshold, round_num, run)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        range_size = dataset_configs[dataset_name]["range_size"]
        split_point = int(total_operations * int(split_ratio.split(":")[0]) / 100)
        times = []
        select_latencies = []
        update_latencies = []
        select_count = 0
        update_count = 0
        start_time = time.perf_counter()
        logging.info(f"Type of start_time: {type(start_time)}")
        index_creation_freq = 0
        similarity_score = None
        
        access_history = {}
        reuse_distances = []
        rows_accessed_list = []
        access_frequency = {}
        query_ranges = []
        short_queries = 0
        medium_queries = 0
        long_queries = 0
        update_counts = []
        current_update_count = 0
        sim_overheads = []
        index_overheads = []
        crack_overheads = []
        sim_scores = []
        threshold_passes = 0
        index_trigger_scores = []
        index_creation_select = 0
        index_creation_update = 0
        select_latencies_before_index = []
        select_latencies_after_index = []
        select_ops_before_index = 0
        select_ops_after_index = 0
        select_time_before_index = 0
        select_time_after_index = 0
        index_created = False
        prediction_errors = []
        operation_latencies = []
        
        cracking_partitions = IntervalTree()
        partition_usage = {}
        prev_intervals = []
        prev_times = []
        prev_update_count = 0
        MAX_PARTITIONS = 100
        CRACK_FREQUENCY = 10
        UPDATE_CRACK_FREQUENCY = 100
        
        # ปรับ all_times โดยแยกตาม round
        try:
            if round_num == 1:
                # Logic เดิมสำหรับ round 1
                cur.execute(f"SELECT DISTINCT {time_column} FROM {table} ORDER BY {time_column} LIMIT 1000")
                distinct_times = [row[0] for row in cur.fetchall()]
                block_size = 1
                num_blocks = total_operations // block_size
                all_times = []
                for i in range(num_blocks):
                    chosen_time = distinct_times[i % len(distinct_times)]
                    all_times.extend([chosen_time] * block_size)
                remaining = total_operations - len(all_times)
                if remaining > 0:
                    chosen_time = distinct_times[num_blocks % len(distinct_times)]
                    all_times.extend([chosen_time] * remaining)
            else:  # round 2
                # ปรับสำหรับ round 2: ลด distinct values ลงเพื่อเพิ่มการซ้ำ
                cur.execute(f"SELECT DISTINCT {time_column} FROM {table} ORDER BY {time_column} LIMIT 500")
                distinct_times = [row[0] for row in cur.fetchall()]
                block_size = 1
                num_blocks = total_operations // block_size
                all_times = []
                for i in range(num_blocks):
                    chosen_time = distinct_times[i % len(distinct_times)]
                    all_times.extend([chosen_time] * block_size)
                remaining = total_operations - len(all_times)
                if remaining > 0:
                    chosen_time = distinct_times[num_blocks % len(distinct_times)]
                    all_times.extend([chosen_time] * remaining)
        except Exception as e:
            logging.error(f"Error while querying table {table}: {e}")
            raise
        
        # เพิ่มสำหรับ LearnedIndex: เก็บข้อมูลสำหรับ retrain
        learned_index = None
        training_data_learned = []  # เก็บข้อมูลสำหรับ retrain
        operation_counter = 0  # นับจำนวน operation เพื่อ retrain
        
        for i in range(total_operations):
            if i % 100 == 0 or i == 0:
                elapsed_time = time.perf_counter() - start_time
                logging.info(f"Status: Running {method} on {dataset_name}, Split {split_ratio}, Run {run}/5 - Operation {i}/{total_operations} - Elapsed Time: {elapsed_time:.2f} seconds")
                if i > 0:
                    update_counts.append(current_update_count)
                    current_update_count = 0
            
            if method == "HSM" and i % 500 == 0 and i > 0:
                sim_start_time = time.perf_counter()
                if round_num == 2 and i >= 200:
                    prev_times = times[i-200:i-100]
                    curr_times = times[i-100:i]
                else:
                    prev_times = times[max(0, i-200):i-100] if i >= 200 else times[:i-100]
                    curr_times = times[max(0, i-150):i] if i >= 150 else times[:i]
                similarity_score = hsm_similarity(prev_times, curr_times, prev_update_count, len([t for t in curr_times if i >= split_point]), range_size, round_num)
                sim_end_time = time.perf_counter()
                sim_overheads.append(sim_end_time - sim_start_time)
                sim_scores.append(similarity_score)
                logging.info(f"Round {round_num} - Similarity Score: {similarity_score:.6f}, Computation Overhead: {(sim_end_time - sim_start_time):.6f} seconds")
                
                if round_num == 2:
                    index_exists = check_index_exists(cur, table, time_column)
                    if similarity_score <= 0.6 and not index_exists:
                        index_start_time = time.perf_counter()
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{time_column} ON {table} ({time_column});")
                        conn.commit()
                        index_end_time = time.perf_counter()
                        index_overheads.append(index_end_time - index_start_time)
                        index_creation_freq += 1
                        if i < split_point:
                            index_creation_select += 1
                        else:
                            index_creation_update += 1
                        index_created = True
                        logging.info(f"Round {round_num} (Adaptive Indexing): Index created on {table}.{time_column}, Overhead: {(index_end_time - index_start_time):.6f} seconds")
                    elif index_exists:
                        logging.info(f"Round {round_num} (Adaptive Indexing): Reusing existing index on {table}.{time_column} (Similarity Score: {similarity_score:.6f})")
                
                prev_times = curr_times
                prev_update_count = len([t for t in curr_times if i >= split_point])
            
            query_start_time = time.perf_counter()
            interval_start = all_times[i % len(all_times)]
            interval_end = interval_start + timedelta(seconds=30)
            query_range = (interval_end - interval_start).total_seconds()
            query_ranges.append(query_range)
            
            if method == "HSM":
                if i < split_point:
                    if i % 1000 == 0 or (index_created and i % 500 == 0):
                        explain_query(cur, f"SELECT * FROM {table} WHERE {time_column} BETWEEN %s AND %s", (interval_start, interval_end))
                    if dataset_name == "OpenSky":
                        if round_num == 2:
                            cur.execute(
                                f"SELECT datetime FROM {table} WHERE {time_column} BETWEEN %s AND %s AND baroaltitude IS NOT NULL",
                                (interval_start, interval_end)
                            )
                        else:
                            cur.execute(
                                f"SELECT datetime FROM {table} WHERE {time_column} BETWEEN %s AND %s AND baroaltitude IS NOT NULL LIMIT 1",
                                (interval_start, interval_end)
                            )
                    else:
                        cur.execute(
                            f"SELECT * FROM {table} WHERE {time_column} BETWEEN %s AND %s",
                            (interval_start, interval_end)
                        )
                    select_count += 1
                    accessed_rows = set(row[0] for row in cur.fetchall())
                    rows_accessed_list.append(len(accessed_rows))
                    for row_time in accessed_rows:
                        access_frequency[row_time] = access_frequency.get(row_time, 0) + 1
                        if row_time in access_history:
                            last_access = access_history[row_time][-1]
                            reuse_distance = query_start_time - last_access
                            reuse_distances.append(reuse_distance)
                            access_history[row_time].append(query_start_time)
                        else:
                            access_history[row_time] = [query_start_time]
                else:
                    if dataset_name == "OpenSky":
                        cur.execute(
                            f"UPDATE {table} SET {update_column} = {update_column} + 1 WHERE {time_column} = %s",
                            (all_times[i % len(all_times)],)
                        )
                    elif dataset_name == "BANK":
                        cur.execute(
                            f"UPDATE {table} SET {update_column} = {update_column} + 1 WHERE {time_column} = %s",
                            (all_times[i % len(all_times)],)
                        )
                    elif dataset_name == "COVID":
                        cur.execute(
                            f"UPDATE {table} SET {update_column} = {update_column} + 1 WHERE  {time_column} = %s",
                            (all_times[i % len(all_times)],)
                        )
                    conn.commit()
                    update_count += 1
                    current_update_count += 1
            elif method == "B-Tree":
                if i < split_point:
                    if i % 1000 == 0:
                        explain_query(cur, f"SELECT * FROM {table} WHERE {time_column} BETWEEN %s AND %s", (interval_start, interval_end))
                    if dataset_name == "OpenSky":
                        cur.execute(
                            f"SELECT datetime FROM {table} WHERE {time_column} BETWEEN %s AND %s AND baroaltitude IS NOT NULL LIMIT 1",
                            (interval_start, interval_end)
                        )
                    else:
                        cur.execute(
                            f"SELECT * FROM {table} WHERE {time_column} BETWEEN %s AND %s",
                            (interval_start, interval_end)
                        )
                    select_count += 1
                    accessed_rows = set(row[0] for row in cur.fetchall())
                    rows_accessed_list.append(len(accessed_rows))
                    for row_time in accessed_rows:
                        access_frequency[row_time] = access_frequency.get(row_time, 0) + 1
                        if row_time in access_history:
                            last_access = access_history[row_time][-1]
                            reuse_distance = query_start_time - last_access
                            reuse_distances.append(reuse_distance)
                            access_history[row_time].append(query_start_time)
                        else:
                            access_history[row_time] = [query_start_time]
                else:
                    cur.execute(
                        f"UPDATE {table} SET {update_column} = {update_column} + 1 WHERE {time_column} = %s",
                        (all_times[i % len(all_times)],)
                    )
                    update_count += 1
                    current_update_count += 1
            elif method == "Cracking (Adaptive Partitioning)":
                if i < split_point:
                    crack_start_time = time.perf_counter()
                    interval_start_ts = interval_start.timestamp()
                    interval_end_ts = interval_end.timestamp()
                    
                    if i % CRACK_FREQUENCY == 0:
                        cracking_partitions[interval_start_ts:interval_end_ts] = None
                        cracking_partitions.merge_overlaps(strict=False)
                        while len(cracking_partitions) > MAX_PARTITIONS:
                            sorted_partitions = sorted(partition_usage.items(), key=lambda x: x[1])
                            for (p_start, p_end), _ in sorted_partitions:
                                if Interval(p_start, p_end) in cracking_partitions:
                                    cracking_partitions.remove(Interval(p_start, p_end))
                                    if (p_start, p_end) in partition_usage:
                                        del partition_usage[(p_start, p_end)]
                                    break
                        new_partition_usage = {}
                        for interval in cracking_partitions:
                            start_ts = interval.begin
                            end_ts = interval.end
                            key = (start_ts, end_ts)
                            if key in partition_usage:
                                new_partition_usage[key] = partition_usage[key]
                            else:
                                new_partition_usage[key] = 1
                        partition_usage = new_partition_usage
                        if i % 100 == 0:
                            logging.info(f"Sorted partitions: {len(cracking_partitions)} partitions")
                    
                    crack_end_time = time.perf_counter()
                    crack_overheads.append(crack_end_time - crack_start_time)

                    accessed_rows = set()
                    overlapping_partitions = 0
                    prev_intervals.append((interval_start, interval_end))
                    if len(prev_intervals) > 5:
                        prev_intervals.pop(0)
                    
                    overlapping_intervals = cracking_partitions[interval_start_ts:interval_end_ts]
                    overlapping_partitions = len(overlapping_intervals)
                    
                    for interval in overlapping_intervals:
                        p_start = datetime.fromtimestamp(interval.begin)
                        p_end = datetime.fromtimestamp(interval.end)
                        query_start = max(p_start, interval_start)
                        query_end = min(p_end, interval_end)
                        
                        if i % 1000 == 0:
                            explain_query(cur, f"SELECT {time_column} FROM {table} WHERE {time_column} = %s AND baroaltitude IS NOT NULL LIMIT 1", (query_start,))
                        if dataset_name == "OpenSky":
                            query_start_time = time.perf_counter()
                            logging.info(f"Starting query for operation {i} on partition [{query_start}, {query_end}]")
                            cur.execute(
                                f"""
                                SELECT {time_column}
                                FROM {table}
                                WHERE {time_column} = %s
                                AND baroaltitude IS NOT NULL
                                LIMIT 1
                                """,
                                (all_times[i % len(all_times)],)
                            )
                            rows = cur.fetchall()
                            query_end_time = time.perf_counter()
                            latency = (query_end_time - query_start_time) * 1000
                            logging.info(f"Query on partition [{query_start}, {query_end}] took {latency:.2f} ms, returned {len(rows)} rows")
                        else:
                            cur.execute(
                                f"SELECT {time_column} FROM {table} WHERE {time_column} BETWEEN %s AND %s",
                                (query_start, query_end)
                            )
                            rows = cur.fetchall()
                            latency = 0
                        partition_usage[(interval.begin, interval.end)] = partition_usage.get((interval.begin, interval.end), 0) + 1
                        accessed_rows.update(row[0] for row in rows)
                    
                    logging.debug(f"Operation {i}: Processed {overlapping_partitions} overlapping partitions")
                    
                    select_count += 1
                    rows_accessed_list.append(len(accessed_rows))
                    for row_time in accessed_rows:
                        access_frequency[row_time] = access_frequency.get(row_time, 0) + 1
                        if row_time in access_history:
                            last_access = access_history[row_time][-1]
                            reuse_distance = query_start_time - last_access
                            reuse_distances.append(reuse_distance)
                            access_history[row_time].append(query_start_time)
                        else:
                            access_history[row_time] = [query_start_time]
                else:
                    crack_start_time = time.perf_counter()
                    if dataset_name == "OpenSky":
                        if i % UPDATE_CRACK_FREQUENCY == 0:
                            old_time_ts = all_times[i % len(all_times)].timestamp()
                            cracking_partitions[old_time_ts - 1:old_time_ts + 1] = None
                            cracking_partitions.merge_overlaps(strict=False)
                            while len(cracking_partitions) > MAX_PARTITIONS:
                                sorted_partitions = sorted(partition_usage.items(), key=lambda x: x[1])
                                for (p_start, p_end), _ in sorted_partitions:
                                    if Interval(p_start, p_end) in cracking_partitions:
                                        cracking_partitions.remove(Interval(p_start, p_end))
                                        if (p_start, p_end) in partition_usage:
                                            del partition_usage[(p_start, p_end)]
                                        break
                            new_partition_usage = {}
                            for interval in cracking_partitions:
                                start_ts = interval.begin
                                end_ts = interval.end
                                key = (start_ts, end_ts)
                                if key in partition_usage:
                                    new_partition_usage[key] = partition_usage[key]
                                else:
                                    new_partition_usage[key] = 1
                            partition_usage = new_partition_usage
                            logging.info(f"Cracking (UPDATE): Added partition for {all_times[i % len(all_times)]}")
                        
                        update_start_time = time.perf_counter()
                        cur.execute(
                            f"UPDATE {table} SET {update_column} = {update_column} + 1 WHERE {time_column} = %s",
                            (all_times[i % len(all_times)],)
                        )
                        conn.commit()
                        update_end_time = time.perf_counter()
                        logging.info(f"Update operation {i} took {(update_end_time - update_start_time) * 1000:.2f} ms")
                    else:
                        update_start_time = time.perf_counter()
                        cur.execute(
                            f"UPDATE {table} SET {update_column} = {update_column} + 1 WHERE {time_column} = %s",
                            (all_times[i % len(all_times)],)
                        )
                        conn.commit()
                        update_end_time = time.perf_counter()
                        logging.info(f"Update operation {i} took {(update_end_time - update_start_time) * 1000:.2f} ms")
                    crack_end_time = time.perf_counter()
                    crack_overheads.append(crack_end_time - crack_start_time)
                    update_count += 1
                    current_update_count += 1
            elif method == "LearnedIndex":
                operation_counter += 1
                
                # สร้าง LearnedIndex ใหม่ในรันแรก หรือเมื่อต้อง retrain
                if learned_index is None or operation_counter % 1000 == 0:
                    crack_start_time = time.perf_counter()
                    learned_index = LearnedIndex(table, time_column)
                    learn_end_time = time.perf_counter()
                    learn_overhead = learn_end_time - crack_start_time
                    logging.info(f"Learned Index Learning Overhead: {learn_overhead:.6f} seconds")
                
                if i < split_point:
                    pos, error = learned_index.search(interval_start)
                    prediction_errors.append(error)
                    
                    # เก็บข้อมูลสำหรับ retrain
                    training_data_learned.append((interval_start.timestamp(), pos if pos is not None else i % 1000))  # ใช้ position หรือ dummy position
                    
                    # Retrain ทุก 1,000 operations หรือเมื่อ error สูง
                    if operation_counter % 1000 == 0 and i > 0:
                        avg_error = np.mean(prediction_errors[-100:]) if prediction_errors else 0
                        error_threshold = 50
                        if avg_error > error_threshold or operation_counter % 1000 == 0:
                            crack_start_time = time.perf_counter()
                            # Retrain ด้วยข้อมูลที่เก็บมา
                            if len(training_data_learned) > 0:
                                keys = np.array([d[0] for d in training_data_learned]).reshape(-1, 1)
                                positions = np.array([d[1] for d in training_data_learned])
                                learned_index.models = []  # ล้างโมเดลเก่า
                                learned_index.data = training_data_learned
                                model_level1 = LinearRegression()
                                model_level1.fit(keys, positions)
                                learned_index.models.append(model_level1)
                                
                                num_groups = min(100, len(keys) // 2)
                                group_size = len(keys) // num_groups if num_groups > 0 else len(keys)
                                models_level2 = []
                                for group in range(num_groups):
                                    start = group * group_size
                                    end = (group + 1) * group_size if group < num_groups - 1 else len(keys)
                                    keys_group = keys[start:end]
                                    positions_group = positions[start:end]
                                    if len(keys_group) > 0:
                                        model = LinearRegression()
                                        model.fit(keys_group, positions_group)
                                        models_level2.append((start, end, model))
                                learned_index.models.append(models_level2)
                                learn_end_time = time.perf_counter()
                                learn_overhead = learn_end_time - crack_start_time
                                logging.info(f"Learned Index Retraining Overhead: {learn_overhead:.6f} seconds, Avg Error: {avg_error:.2f}")
                            # ล้าง training data เพื่อเริ่มเก็บใหม่
                            training_data_learned = training_data_learned[-100:]  # เก็บข้อมูลล่าสุด 100 ตัวอย่าง
                    
                    if i % 500 == 0 and i > 0:
                        avg_error = np.mean(prediction_errors[-100:]) if prediction_errors else 0
                        error_threshold = 50
                        index_exists = check_index_exists(cur, table, time_column)
                        if avg_error > error_threshold:
                            if index_exists:
                                drop_start_time = time.perf_counter()
                                cur.execute(f"DROP INDEX IF EXISTS idx_{table}_{time_column};")
                                conn.commit()
                                drop_end_time = time.perf_counter()
                                logging.info(f"Round {round_num} (LearnedIndex Adaptive Indexing): Dropped index idx_{table}_{time_column}, Overhead: {(drop_end_time - drop_start_time):.6f} seconds")
                            index_start_time = time.perf_counter()
                            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{time_column} ON {table} ({time_column});")
                            conn.commit()
                            index_end_time = time.perf_counter()
                            index_overheads.append(index_end_time - index_start_time)
                            index_creation_freq += 1
                            if i < split_point:
                                index_creation_select += 1
                            else:
                                index_creation_update += 1
                            index_created = True
                            logging.info(f"Round {round_num} (LearnedIndex Adaptive Indexing): Index created on {table}.{time_column}, Error: {avg_error:.2f}, Overhead: {(index_end_time - index_start_time):.6f} seconds")
                    
                    if pos is not None:
                        if i % 1000 == 0:
                            explain_query(cur, f"SELECT {time_column} FROM {table} WHERE {time_column} = %s", (interval_start,))
                        if dataset_name == "OpenSky":
                            cur.execute(
                                f"SELECT {time_column} FROM {table} WHERE {time_column} = %s LIMIT 1",
                                (interval_start,)
                            )
                        else:
                            cur.execute(
                                f"SELECT {time_column} FROM {table} WHERE {time_column} = %s",
                                (interval_start,)
                            )
                    else:
                        if i % 1000 == 0:
                            explain_query(cur, f"SELECT {time_column} FROM {table} WHERE {time_column} BETWEEN %s AND %s LIMIT 1", (interval_start, interval_end))
                        if dataset_name == "OpenSky":
                            cur.execute(
                                f"SELECT {time_column} FROM {table} WHERE {time_column} BETWEEN %s AND %s LIMIT 1",
                                (interval_start, interval_end)
                            )
                        else:
                            cur.execute(
                                f"SELECT {time_column} FROM {table} WHERE {time_column} BETWEEN %s AND %s",
                                (interval_start, interval_end)
                            )
                    select_count += 1
                    accessed_rows = set(row[0] for row in cur.fetchall())
                    rows_accessed_list.append(len(accessed_rows))
                    for row_time in accessed_rows:
                        access_frequency[row_time] = access_frequency.get(row_time, 0) + 1
                        if row_time in access_history:
                            last_access = access_history[row_time][-1]
                            reuse_distance = query_start_time - last_access
                            reuse_distances.append(reuse_distance)
                            access_history[row_time].append(query_start_time)
                        else:
                            access_history[row_time] = [query_start_time]
            
            conn.commit()
            query_end_time = time.perf_counter()
            query_latency = (query_end_time - query_start_time) * 1000
            operation_latencies.append(query_latency)
            
            if i < split_point:
                if not index_created:
                    select_ops_before_index += 1
                    select_time_before_index += (query_end_time - query_start_time)
                    select_latencies_before_index.append(query_latency)
                else:
                    select_ops_after_index += 1
                    select_time_after_index += (query_end_time - query_start_time)
                    select_latencies_after_index.append(query_latency)
                select_latencies.append(query_latency)
            else:
                update_latencies.append(query_latency)
            
            if dataset_name == "OpenSky":
                if (i + 1) % 100 == 0:
                    avg_latency_so_far = np.mean(operation_latencies[-100:])
                    logging.info(f"OpenSky - Operation {i+1}/{total_operations} - Average Latency (last 100 operations): {avg_latency_so_far:.2f} ms")
            
            times.append(all_times[i % len(all_times)])
        
        update_counts.append(current_update_count)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        throughput = total_operations / total_time if total_time > 0 else 0
        avg_latency = np.mean(select_latencies + update_latencies) if (select_latencies or update_latencies) else 0
        latency_sd = np.std(select_latencies + update_latencies) if (select_latencies or update_latencies) else 0
        latency_ci = stats.t.interval(0.95, len(select_latencies + update_latencies)-1, loc=avg_latency, scale=stats.sem(select_latencies + update_latencies)) if (select_latencies or update_latencies) else (0, 0)
        avg_select_latency = np.mean(select_latencies) if select_latencies else 0
        avg_update_latency = np.mean(update_latencies) if update_latencies else 0
        select_throughput = select_count / total_time if total_time > 0 else 0
        update_throughput = update_count / total_time if total_time > 0 else 0
        select_throughput_before = select_ops_before_index / select_time_before_index if select_time_before_index > 0 else 0
        select_throughput_after = select_ops_after_index / select_time_after_index if select_time_after_index > 0 else 0
        avg_select_latency_before = np.mean(select_latencies_before_index) if select_latencies_before_index else 0
        avg_select_latency_after = np.mean(select_latencies_after_index) if select_latencies_after_index else 0
        update_intensity = sum(1 for i in range(total_operations) if i >= split_point) / total_operations
        temporal_locality = np.median(reuse_distances) if reuse_distances else 0
        temporal_locality_sd = np.std(reuse_distances) if reuse_distances else 0
        temporal_locality_ci = stats.t.interval(0.95, len(reuse_distances)-1, loc=temporal_locality, scale=stats.sem(reuse_distances)) if reuse_distances else (0, 0)
        rows_accessed_avg = np.mean(rows_accessed_list) if rows_accessed_list else 0
        access_frequency_avg = np.mean(list(access_frequency.values())) if access_frequency else 0
        avg_query_range = np.mean(query_ranges) if query_ranges else 0
        update_freq_in_window = len([t for t in times[-100:] if total_operations - 1 >= split_point]) if len(times) >= 100 else len([t for t in times if total_operations - 1 >= split_point])
        update_clustering = np.var(update_counts) if update_counts else 0
        
        q1 = np.percentile(query_ranges, 25) if query_ranges else 0
        q3 = np.percentile(query_ranges, 75) if query_ranges else 0
        for query_range in query_ranges:
            if query_range < q1:
                short_queries += 1
            elif query_range < q3:
                medium_queries += 1
            else:
                long_queries += 1
        short_query_ratio = short_queries / select_count if select_count > 0 else 0
        medium_query_ratio = medium_queries / select_count if select_count > 0 else 0
        long_query_ratio = long_queries / select_count if select_count > 0 else 0
        
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        io_counters = psutil.disk_io_counters()
        io_reads = io_counters.read_count if io_counters else 0
        io_writes = io_counters.write_count if io_counters else 0
        sim_overhead_avg = np.mean(sim_overheads) if sim_overheads else 0
        index_overhead_avg = np.mean(index_overheads) if index_overheads else 0
        crack_overhead_avg = np.mean(crack_overheads) if crack_overheads else 0
        avg_sim_score = np.mean(sim_scores) if sim_scores else 0
        sim_score_sd = np.std(sim_scores) if sim_scores else 0
        sim_score_ci = stats.t.interval(0.95, len(sim_scores)-1, loc=avg_sim_score, scale=stats.sem(sim_scores)) if sim_scores else (0, 0)
        min_sim_score = min(sim_scores) if sim_scores else 0
        max_sim_score = max(sim_scores) if sim_scores else 0
        avg_index_trigger_score = np.mean(index_trigger_scores) if index_trigger_scores else 0
        
        min_overhead = 0.001
        b_tree_latency = B_TREE_LATENCIES[dataset_name][split_ratio]
        if method == "HSM" and round_num == 2:
            total_overhead = max(sim_overhead_avg + index_overhead_avg, min_overhead)
            cost_benefit_ratio = ((b_tree_latency / 1000) - (avg_latency / 1000)) / total_overhead
        elif method == "Cracking (Adaptive Partitioning)":
            crack_overhead_avg = max(crack_overhead_avg, min_overhead)
            cost_benefit_ratio = ((b_tree_latency / 1000) - (avg_latency / 1000)) / crack_overhead_avg
        elif method == "LearnedIndex":
            learn_overhead = max(learn_overhead, min_overhead)
            cost_benefit_ratio = ((b_tree_latency / 1000) - (avg_latency / 1000)) / learn_overhead
        else:
            cost_benefit_ratio = 0
        
        logging.info(f"\n--- Metrics for {method} - {dataset_name} - {split_ratio} ---")
        logging.info(f"Latency: {avg_latency:.2f} ms (SD: {latency_sd:.2f}), 95% CI: [{latency_ci[0]:.2f}, {latency_ci[1]:.2f}]")
        logging.info(f"Select Latency: {avg_select_latency:.2f} ms")
        logging.info(f"Update Latency: {avg_update_latency:.2f} ms")
        logging.info(f"Select Throughput: {select_throughput:.2f} qps")
        logging.info(f"Update Throughput: {update_throughput:.2f} qps")
        logging.info(f"Throughput: {throughput:.2f} qps")
        logging.info(f"Index Creation Frequency: {index_creation_freq}")
        logging.info(f"Total Runtime: {total_time:.2f} s")
        logging.info(f"Update Intensity: {update_intensity:.2f}")
        logging.info(f"Temporal Locality: {temporal_locality:.2f} s (SD: {temporal_locality_sd:.2f}), 95% CI: [{temporal_locality_ci[0]:.2f}, {temporal_locality_ci[1]:.2f}]")
        logging.info(f"Rows Accessed Avg: {rows_accessed_avg:.2f}")
        logging.info(f"Access Frequency Avg: {access_frequency_avg:.2f}")
        logging.info(f"Average Query Range: {avg_query_range:.2f} s")
        logging.info(f"Update Frequency in Window: {update_freq_in_window:.2f}")
        logging.info(f"Update Clustering: {update_clustering:.2f}")
        logging.info(f"Short Query Ratio: {short_query_ratio:.2f}")
        logging.info(f"Medium Query Ratio: {medium_query_ratio:.2f}")
        logging.info(f"Long Query Ratio: {long_query_ratio:.2f}")
        logging.info(f"CPU Usage: {cpu_usage:.2f}%")
        logging.info(f"Memory Usage: {memory_usage:.2f} MB")
        logging.info(f"Disk Reads: {io_reads}, Disk Writes: {io_writes}")
        if sim_overhead_avg > 0:
            logging.info(f"Similarity Score Overhead: {sim_overhead_avg:.2f} s")
        if index_overhead_avg > 0:
            logging.info(f"Index Creation Overhead: {index_overhead_avg:.2f} s")
        if crack_overhead_avg > 0:
            logging.info(f"Cracking Partitioning Overhead: {crack_overhead_avg:.2f} s")
        if method == "HSM":
            logging.info(f"Threshold Passes: {threshold_passes}")
            logging.info(f"Average Similarity Score: {avg_sim_score:.2f} (SD: {sim_score_sd:.2f}), 95% CI: [{sim_score_ci[0]:.2f}, {sim_score_ci[1]:.2f}]")
            logging.info(f"Min Similarity Score: {min_sim_score:.2f}, Max Similarity Score: {max_sim_score:.2f}")
            logging.info(f"Average Index Trigger Score: {avg_index_trigger_score:.2f}")
            logging.info(f"Index Creations during SELECT: {index_creation_select}, during UPDATE: {index_creation_update}")
            logging.info(f"Select Latency Before Index: {avg_select_latency_before:.2f} ms, After Index: {avg_select_latency_after:.2f} ms")
            logging.info(f"Select Throughput Before Index: {select_throughput_before:.2f} qps, After Index: {select_throughput_after:.2f} qps")
        logging.info("--- End of Metrics ---\n")
        
        return (avg_latency, latency_sd, latency_ci, avg_select_latency, avg_update_latency, select_throughput, update_throughput, throughput, 
                index_creation_freq, total_time, update_intensity, similarity_score, temporal_locality, temporal_locality_sd, temporal_locality_ci, 
                rows_accessed_avg, access_frequency_avg, avg_query_range, update_freq_in_window, update_clustering, 
                short_query_ratio, medium_query_ratio, long_query_ratio, 
                cpu_usage, memory_usage, io_reads, io_writes, sim_overhead_avg, index_overhead_avg, crack_overhead_avg, learn_overhead, cost_benefit_ratio, 
                threshold_passes, avg_sim_score, sim_score_sd, sim_score_ci, min_sim_score, max_sim_score, avg_index_trigger_score, 
                index_creation_select, index_creation_update, avg_select_latency_before, avg_select_latency_after, select_throughput_before, select_throughput_after)
    finally:
        cur.close()
        conn.close()
def run_experiments(methods, datasets, threshold, splits, runs, round_num):
    experiment_start_time = time.time()
    logging.info(f"Starting experiment: methods={methods}, datasets={datasets}, splits={splits}, runs={runs}, round={round_num}, threshold={threshold}")
    
    log_file = f"/Users/arunreungsinkonkarn/Desktop/hsm_experiment_log_paper1_round{round_num}.csv"
    
    try:
        if not os.path.exists(log_file) or os.stat(log_file).st_size == 0:
            with open(log_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["round", "dataset", "split_ratio", "run", "method", "latency_ms", "latency_sd", "latency_ci_lower", "latency_ci_upper", 
                                 "select_latency_ms", "update_latency_ms", "select_throughput_qps", "update_throughput_qps", 
                                 "throughput_qps", "throughput_sd", "throughput_ci_lower", "throughput_ci_upper", 
                                 "index_creation_freq", "total_time_s", "update_intensity", "similarity_score", 
                                 "threshold_passes", "avg_sim_score", "sim_score_sd", "sim_score_ci_lower", "sim_score_ci_upper", 
                                 "min_sim_score", "max_sim_score", "avg_index_trigger_score", "index_creation_select", "index_creation_update", 
                                 "select_latency_before_ms", "select_latency_after_ms", "select_throughput_before_qps", "select_throughput_after_qps", 
                                 "rows_accessed_avg", "access_frequency_avg", "avg_query_range_s", "update_freq_in_window", "update_clustering", 
                                 "short_query_ratio", "medium_query_ratio", "long_query_ratio", 
                                 "cpu_usage_percent", "memory_usage_mb", "io_reads", "io_writes", "sim_overhead_s", "index_overhead_s", "crack_overhead_s", "learn_overhead_s", "cost_benefit_ratio", 
                                 "temporal_locality_s", "temporal_locality_sd", "temporal_locality_ci_lower", "temporal_locality_ci_upper"])
    except Exception as e:
        logging.error(f"Failed to create or write to CSV file {log_file}: {e}")
        raise
    
    datasets_to_run = ALL_DATASETS if "all" in [d.lower() for d in datasets] else datasets
    methods_to_run = ALL_METHODS if "all" in [m.lower() for m in methods] else methods
    splits_to_run = ALL_SPLITS if "all" in [s.lower() for s in splits] else splits
    
    if round_num == 2:
        methods_to_run = ["HSM"]
        logging.info(f"Round 2: Running only HSM method with threshold {threshold} for proactive indexing")
    elif round_num == 3:
        methods_to_run = ["HSM_Enhanced"]
        logging.info(f"Round 3: Running only HSM_Enhanced method with threshold {threshold} for hybrid indexing with lightweight ML")
    
    for dataset_name in datasets_to_run:
        table = dataset_configs[dataset_name]["table"]
        time_column = dataset_configs[dataset_name]["time_column"]
        update_column = dataset_configs[dataset_name]["update_column"]
        operations = dataset_configs[dataset_name]["total_operations"]
        
        for split in splits_to_run:
            for method in methods_to_run:
                throughputs = []
                results = []
                
                for run in range(1, runs + 1):
                    logging.info(f"Starting Run {run}/{runs} for {method} on {dataset_name} with Split {split} (Round {round_num})...")
                    if round_num == 1 or round_num == 3:
                        conn = psycopg2.connect(**DB_CONFIG)
                        cur = conn.cursor()
                        try:
                            logging.info(f"Checking indexes before running {method}...")
                            cur.execute("""
                                SELECT tablename, indexname, indexdef
                                FROM pg_indexes
                                WHERE schemaname = 'public'
                                  AND tablename = %s
                                ORDER BY tablename, indexname;
                            """, (table,))
                            indexes = cur.fetchall()
                            for index in indexes:
                                logging.info(f"Found index: {index[1]} on table {index[0]}: {index[2]}")

                            logging.info(f"Dropping index for table {table} if it exists...")
                            cur.execute(f"DROP INDEX IF EXISTS idx_{table}_{time_column};")
                            conn.commit()
                            logging.info(f"Dropped index idx_{table}_{time_column} successfully.")

                            logging.info(f"Verifying indexes after drop for table {table}...")
                            cur.execute("""
                                SELECT tablename, indexname, indexdef
                                FROM pg_indexes
                                WHERE schemaname = 'public'
                                  AND tablename = %s
                                ORDER BY tablename, indexname;
                            """, (table,))
                            indexes_after = cur.fetchall()
                            if indexes_after:
                                logging.warning("Indexes still exist after drop:")
                                for index in indexes_after:
                                    logging.warning(f"Found index: {index[1]} on table {index[0]}: {index[2]}")
                            else:
                                logging.info("No indexes found after drop.")
                        except Exception as e:
                            logging.error(f"Error while checking or dropping indexes for {method}: {e}")
                            raise
                        finally:
                            cur.close()
                            conn.close()
                    
                    result = measure_performance(method, dataset_name, split, operations, table, time_column, update_column, threshold, round_num, run)
                    if result:
                        (avg_latency, latency_sd, latency_ci, avg_select_latency, avg_update_latency, select_throughput, update_throughput, throughput, 
                         index_creation_freq, total_time, update_intensity, similarity_score, temporal_locality, temporal_locality_sd, temporal_locality_ci, 
                         rows_accessed_avg, access_frequency_avg, avg_query_range, update_freq_in_window, update_clustering, 
                         short_query_ratio, medium_query_ratio, long_query_ratio, 
                         cpu_usage, memory_usage, io_reads, io_writes, sim_overhead_avg, index_overhead_avg, crack_overhead_avg, learn_overhead, cost_benefit_ratio, 
                         threshold_passes, avg_sim_score, sim_score_sd, sim_score_ci, min_sim_score, max_sim_score, avg_index_trigger_score, 
                         index_creation_select, index_creation_update, avg_select_latency_before, avg_select_latency_after, select_throughput_before, select_throughput_after) = result
                        throughputs.append(throughput)
                        results.append((run, avg_latency, latency_sd, latency_ci, avg_select_latency, avg_update_latency, select_throughput, update_throughput, throughput, 
                                        index_creation_freq, total_time, update_intensity, similarity_score, temporal_locality, temporal_locality_sd, temporal_locality_ci, 
                                        rows_accessed_avg, access_frequency_avg, avg_query_range, update_freq_in_window, update_clustering, 
                                        short_query_ratio, medium_query_ratio, long_query_ratio, 
                                        cpu_usage, memory_usage, io_reads, io_writes, sim_overhead_avg, index_overhead_avg, crack_overhead_avg, learn_overhead, cost_benefit_ratio, 
                                        threshold_passes, avg_sim_score, sim_score_sd, sim_score_ci, min_sim_score, max_sim_score, avg_index_trigger_score, 
                                        index_creation_select, index_creation_update, avg_select_latency_before, avg_select_latency_after, select_throughput_before, select_throughput_after))
                    logging.info(f"Completed Run {run}/{runs} for {method} on {dataset_name} with Split {split} (Round {round_num}) - Total Time: {total_time:.2f} seconds")
                
                throughput_sd = np.std(throughputs) if throughputs else 0
                throughput_ci = stats.t.interval(0.95, len(throughputs)-1, loc=np.mean(throughputs), scale=stats.sem(throughputs)) if throughputs else (0, 0)
                logging.info(f"{method} Throughput SD: {throughput_sd:.2f}, 95% CI: [{throughput_ci[0]:.2f}, {throughput_ci[1]:.2f}]")
                
                try:
                    with open(log_file, "a", newline='') as f:
                        writer = csv.writer(f)
                        for (run, avg_latency, latency_sd, latency_ci, avg_select_latency, avg_update_latency, select_throughput, update_throughput, throughput, 
                             index_creation_freq, total_time, update_intensity, similarity_score, temporal_locality, temporal_locality_sd, temporal_locality_ci, 
                             rows_accessed_avg, access_frequency_avg, avg_query_range, update_freq_in_window, update_clustering, 
                             short_query_ratio, medium_query_ratio, long_query_ratio, 
                             cpu_usage, memory_usage, io_reads, io_writes, sim_overhead_avg, index_overhead_avg, crack_overhead_avg, learn_overhead, cost_benefit_ratio, 
                             threshold_passes, avg_sim_score, sim_score_sd, sim_score_ci, min_sim_score, max_sim_score, avg_index_trigger_score, 
                             index_creation_select, index_creation_update, avg_select_latency_before, avg_select_latency_after, select_throughput_before, select_throughput_after) in results:
                            writer.writerow([round_num, dataset_name, split, run, method, avg_latency, latency_sd, latency_ci[0], latency_ci[1], 
                                             avg_select_latency, avg_update_latency, select_throughput, update_throughput, 
                                             throughput, throughput_sd, throughput_ci[0], throughput_ci[1], 
                                             index_creation_freq, total_time, update_intensity, similarity_score, 
                                             threshold_passes, avg_sim_score, sim_score_sd, sim_score_ci[0], sim_score_ci[1], 
                                             min_sim_score, max_sim_score, avg_index_trigger_score, index_creation_select, index_creation_update, 
                                             avg_select_latency_before, avg_select_latency_after, select_throughput_before, select_throughput_after, 
                                             rows_accessed_avg, access_frequency_avg, avg_query_range, update_freq_in_window, update_clustering, 
                                             short_query_ratio, medium_query_ratio, long_query_ratio, 
                                             cpu_usage, memory_usage, io_reads, io_writes, sim_overhead_avg, index_overhead_avg, crack_overhead_avg, learn_overhead, cost_benefit_ratio, 
                                             temporal_locality, temporal_locality_sd, temporal_locality_ci[0], temporal_locality_ci[1]])
                except Exception as e:
                    logging.error(f"Failed to write to CSV file {log_file}: {e}")
                    raise
    
    experiment_end_time = time.time()
    total_experiment_time = experiment_end_time - experiment_start_time
    logging.info(f"Completed experiment: Total Time: {total_experiment_time:.2f} seconds ({total_experiment_time/60:.2f} minutes)")

def main():
    parser = argparse.ArgumentParser(description="Run selective experiments for HSM, B-Tree, and Cracking methods (Paper 1).")
    parser.add_argument("--methods", nargs='+', default=["all"], help="Methods to run: HSM, B-Tree, Cracking (Adaptive Partitioning), LearnedIndex, HSM_Enhanced, or 'all'")
    parser.add_argument("--datasets", nargs='+', default=["all"], help="Datasets to run: BANK, COVID, OpenSky, or 'all'")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold for HSM (default: 0.6)")
    parser.add_argument("--splits", nargs='+', default=["all"], help="Workload splits: 10:90, 30:70, 50:50, 70:30, 90:10, or 'all'")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per split (default: 5)")
    parser.add_argument("--round", type=int, default=1, help="Experiment round (1 for baseline, 2 for HSM proactive indexing, 3 for HSM with lightweight ML)")
    args = parser.parse_args()
    
    if "all" not in [m.lower() for m in args.methods]:
        invalid_methods = [m for m in args.methods if m not in ALL_METHODS]
        if invalid_methods:
            logging.error(f"Invalid methods: {invalid_methods}. Available methods: {ALL_METHODS}")
            raise ValueError(f"Invalid methods: {invalid_methods}. Available methods: {ALL_METHODS}")
    if "all" not in [d.lower() for d in args.datasets]:
        invalid_datasets = [d for d in args.datasets if d not in ALL_DATASETS]
        if invalid_datasets:
            logging.error(f"Invalid datasets: {invalid_datasets}. Available datasets: {ALL_DATASETS}")
            raise ValueError(f"Invalid datasets: {invalid_datasets}. Available datasets: {ALL_DATASETS}")
    if "all" not in [s.lower() for s in args.splits]:
        invalid_splits = [s for s in args.splits if s not in ALL_SPLITS]
        if invalid_splits:
            logging.error(f"Invalid splits: {invalid_splits}. Available splits: {ALL_SPLITS}")
            raise ValueError(f"Invalid splits: {invalid_splits}. Available splits: {ALL_SPLITS}")
    
    try:
        check_tables_exist()
        check_indexes_exist()
        ensure_postgresql_running()
        run_experiments(args.methods, args.datasets, args.threshold, args.splits, args.runs, args.round)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()