# HSM-ML: Efficient Temporal Indexing for Dynamic Workloads without PKs

## Overview
This repository contains the source code and dataset for the research study titled **"HSM-ML: Efficient Temporal Indexing for Dynamic Workloads without PKs"**. The study proposes HSM-ML, a Hierarchical Similarity Measurement model enhanced with Machine Learning, to enable efficient temporal indexing in dynamic workloads that lack a Primary Key (PK). HSM-ML leverages workload similarity to proactively create indices, optimizing query performance in temporal databases. The research evaluates HSM-ML against traditional indexing methods (B-Tree, Cracking, and LearnedIndex) using three real-world datasets: BANK (20,000 records), COVID (100,000 records), and OpenSky (2,183,879 records).

The key findings include:
- HSM-ML achieves low select latency (L_s = 5.454–5.467 ms in OpenSky, split 90:10) and high throughput (Q_ps = 165–185 qps in OpenSky) in large-scale, dynamic workloads without a PK.
- In smaller datasets like COVID, HSM-ML maintains competitive performance (L_s = 0.559–0.578 ms, Q_ps = 1697–1743 qps, split 10:90).
- In select-heavy workloads like BANK (split 90:10), HSM-ML remains adaptable, though LearnedIndex outperforms with Q_ps = 2149–2168 qps.

This work highlights the potential of HSM-ML for managing unstructured Big Data applications, such as JSON, Graphs, Spatial, Temporal, Tweets, and Reviews, paving the way for future research in scalable data management.

## Repository Structure
- **sourcecode/**: Contains the source code for implementing HSM-ML
  - `run_experiment_selective_paper1.py`: Script for implementing HSM-ML and conducting experiments
  - `stat-hsm-ml.py`: Script for calculating statistical metrics (e.g., select latency, throughput).
  - `graph-hsm-ml.py`: Script for generate graphs (e.g., Graphs latency, throughput).
- **dataset/**: Includes the experimental datasets:
  - `BANK.csv`: Financial transaction dataset (20,000 records).
  - `COVID.csv`: Health statistics dataset (100,000 records).
  - `OpenSky.csv`: Real-time tracking dataset (2,183,879 records).
  - `B77W_test_processed.csv`: Additional large-scale dataset used in experiments (Note: Managed with Git LFS due to file size).
  - `experiment_data.xlsx`: Dataset used for statistical analysis and graph generation.
- **docs/**: Contains the paper draft and related documentation (if applicable).

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/stevezhai139/Research-Source-Code-and-Dataset.git
