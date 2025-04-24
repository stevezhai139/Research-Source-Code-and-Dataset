import pandas as pd
import numpy as np
from scipy.stats import norm, f_oneway, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# read file .xlsx
xlsx_file_path = "experiment_data.xlsx"
df = pd.read_excel(xlsx_file_path, engine='openpyxl')

# structure check
print("โครงสร้างข้อมูลในไฟล์ .xlsx:")
print(df.head())
print("\nคอลัมน์ที่มีในไฟล์:", df.columns.tolist())

# Filter method 
methods_of_interest = ['HSM Proactive Indexing with ML', 'B-Tree', 'Cracking (Adaptive Partitioning)', 'LearnedIndex']
df = df[df['method'].isin(methods_of_interest)]

# Define metrics
performance_metrics = [
    'throughput_qps', 'latency_ms', 'select_latency_ms',
    'update_latency_ms', 'total_time_s', 'select_throughput_before_qps', 'select_throughput_after_qps'
]

resource_metrics = [
    'cpu_usage_percent', 'memory_usage_mb', 'io_reads', 'io_writes', 'overhead_s'
]

additional_metrics = [
    'index_creation_freq', 'update_intensity', 'similarity_score', 'threshold_passes',
    'avg_sim_score', 'avg_index_trigger_score', 'index_creation_select', 'index_creation_update',
    'rows_accessed_avg', 'access_frequency_avg', 'avg_query_range_s', 'update_freq_in_window',
    'update_clustering', 'short_query_ratio', 'medium_query_ratio', 'long_query_ratio', 'temporal_locality_s'
]

all_metrics = performance_metrics + resource_metrics + additional_metrics

# Confidence Interval (95%)
def calculate_confidence_interval(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    if n <= 1:
        return mean, mean
    z = norm.ppf(0.975)  # z-score สำหรับ 95% CI (1.96)
    margin_error = z * (std / np.sqrt(n))
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    return ci_lower, ci_upper

# (mean, std, min, max, CI)
def calculate_statistics(df, groupby_columns, metrics):
    stats_dict = {}
    
    for metric in metrics:
        stats = df.groupby(groupby_columns)[metric].agg(['mean', 'std', 'min', 'max']).reset_index()
        stats.columns = groupby_columns + [f'{metric}_mean', f'{metric}_std', f'{metric}_min', f'{metric}_max']
        
        ci_stats = df.groupby(groupby_columns)[metric].apply(lambda x: calculate_confidence_interval(x)).reset_index()
        ci_stats[[f'{metric}_ci_lower', f'{metric}_ci_upper']] = pd.DataFrame(ci_stats[metric].tolist(), index=ci_stats.index)
        ci_stats = ci_stats.drop(columns=[metric])
        
        stats = stats.merge(ci_stats, on=groupby_columns)
        
        stats_dict[metric] = stats
    
    return stats_dict

# ANOVA
def calculate_anova(df, groupby_columns, metric):
    grouped = df.groupby(groupby_columns)
    anova_results = []
    
    for name, group in grouped:
        dataset, split_ratio = name
        methods = group['method'].unique()
        if len(methods) > 1:
            method_groups = [group[group['method'] == method][metric].values for method in methods]
            valid_groups = [g for g in method_groups if len(g) > 1 and np.std(g, ddof=1) > 0]
            if len(valid_groups) > 1:
                f_stat, p_value = f_oneway(*valid_groups)
                anova_results.append({
                    'dataset': dataset,
                    'split_ratio': split_ratio,
                    'metric': metric,
                    'f_statistic': f_stat,
                    'p_value': p_value
                })
            else:
                anova_results.append({
                    'dataset': dataset,
                    'split_ratio': split_ratio,
                    'metric': metric,
                    'f_statistic': np.nan,
                    'p_value': np.nan,
                    'note': 'Insufficient data or zero variance in some groups'
                })
    
    return pd.DataFrame(anova_results)

# Post-hoc Test (Tukey's HSD)
def calculate_posthoc_tukey(df, groupby_columns, metric):
    grouped = df.groupby(groupby_columns)
    tukey_results = []
    
    for name, group in grouped:
        dataset, split_ratio = name
        methods = group['method'].unique()
        if len(methods) > 1:
            method_groups = [group[group['method'] == method][metric].values for method in methods]
            valid_groups = [g for g in method_groups if len(g) > 1 and np.std(g, ddof=1) > 0]
            if len(valid_groups) > 1:
                temp_df = group[['method', metric]].copy()
                tukey = pairwise_tukeyhsd(endog=temp_df[metric], groups=temp_df['method'], alpha=0.05)
                tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                tukey_df['dataset'] = dataset
                tukey_df['split_ratio'] = split_ratio
                tukey_df['metric'] = metric
                tukey_results.append(tukey_df)
            else:
                tukey_results.append(pd.DataFrame({
                    'dataset': [dataset],
                    'split_ratio': [split_ratio],
                    'metric': [metric],
                    'group1': [np.nan],
                    'group2': [np.nan],
                    'meandiff': [np.nan],
                    'p-adj': [np.nan],
                    'lower': [np.nan],
                    'upper': [np.nan],
                    'reject': [np.nan],
                    'note': ['Insufficient data or zero variance in some groups']
                }))
    
    return pd.concat(tukey_results, ignore_index=True)

#  Wilcoxon Rank-Sum Test (Mann-Whitney U Test)
def calculate_wilcoxon(df, groupby_columns, metric, method1, method2):
    grouped = df.groupby(groupby_columns)
    wilcoxon_results = []
    
    for name, group in grouped:
        dataset, split_ratio = name
        group1 = group[group['method'] == method1][metric].values
        group2 = group[group['method'] == method2][metric].values
        if len(group1) > 0 and len(group2) > 0:
            if len(group1) <= 1 or len(group2) <= 1:
                wilcoxon_results.append({
                    'dataset': dataset,
                    'split_ratio': split_ratio,
                    'metric': metric,
                    'method1': method1,
                    'method2': method2,
                    'u_statistic': np.nan,
                    'p_value': np.nan,
                    'note': 'Insufficient data (need at least 2 data points per group)'
                })
            elif np.all(group1 == group1[0]) and np.all(group2 == group2[0]):
                wilcoxon_results.append({
                    'dataset': dataset,
                    'split_ratio': split_ratio,
                    'metric': metric,
                    'method1': method1,
                    'method2': method2,
                    'u_statistic': np.nan,
                    'p_value': np.nan,
                    'note': 'All values are identical within groups'
                })
            else:
                u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                wilcoxon_results.append({
                    'dataset': dataset,
                    'split_ratio': split_ratio,
                    'metric': metric,
                    'method1': method1,
                    'method2': method2,
                    'u_statistic': u_stat,
                    'p_value': p_value
                })
    
    return pd.DataFrame(wilcoxon_results)

# function Correlation ระหว่าง dataset, split_ratio, method with selected metrics
def calculate_correlation(df, target_metric):
    categorical_columns = ['dataset', 'split_ratio', 'method']
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    
    correlations = df_encoded.corr()[target_metric].drop(target_metric)
    correlation_df = pd.DataFrame(correlations).reset_index()
    correlation_df.columns = ['variable', f'correlation_with_{target_metric}']
    
    return correlation_df

# statistics  dataset, split_ratio, และ method
groupby_columns = ['dataset', 'split_ratio', 'method']
stats = calculate_statistics(df, groupby_columns, all_metrics)

# statistics display
print("=== ค่าสถิติสำหรับการเปรียบเทียบประสิทธิภาพ ความเสถียร การใช้ทรัพยากร และเมตริกซ์เพิ่มเติม ===")
print("\n--- เมตริกซ์สำหรับประสิทธิภาพ ---")
for metric in performance_metrics:
    print(f"\nค่าสถิติสำหรับ {metric}:")
    print(stats[metric].to_string(index=False))
    print("\n")

print("\n--- เมตริกซ์สำหรับการใช้ทรัพยากร ---")
for metric in resource_metrics:
    print(f"\nค่าสถิติสำหรับ {metric}:")
    print(stats[metric].to_string(index=False))
    print("\n")

print("\n--- เมตริกซ์เพิ่มเติม ---")
for metric in additional_metrics:
    print(f"\nค่าสถิติสำหรับ {metric}:")
    print(stats[metric].to_string(index=False))
    print("\n")

# ANOVA and Post-hoc Test (Tukey's HSD)
print("=== ผลการวิเคราะห์ ANOVA และ Post-hoc Test (Tukey's HSD) ===")
anova_groupby_columns = ['dataset', 'split_ratio']
anova_results = {}
tukey_results = {}

for metric in all_metrics:
    # ANOVA
    anova_result = calculate_anova(df, anova_groupby_columns, metric)
    if not anova_result.empty:
        anova_results[metric] = anova_result
        print(f"\nANOVA สำหรับ {metric}:")
        print(anova_result.to_string(index=False))
        print("\n")
    
    # Post-hoc Test (Tukey's HSD)
    tukey_result = calculate_posthoc_tukey(df, anova_groupby_columns, metric)
    if not tukey_result.empty:
        tukey_results[metric] = tukey_result
        print(f"\nPost-hoc Test (Tukey's HSD) สำหรับ {metric}:")
        print(tukey_result.to_string(index=False))
        print("\n")

# Wilcoxon Rank-Sum Test compare to HSM Proactive Indexing with ML vs all methods 
print("=== ผลการวิเคราะห์ Wilcoxon Rank-Sum Test ===")
method1 = 'HSM Proactive Indexing with ML'
other_methods = ['B-Tree', 'Cracking (Adaptive Partitioning)', 'LearnedIndex']
wilcoxon_results = {}

for metric in all_metrics:
    wilcoxon_results[metric] = {}
    for method2 in other_methods:
        wilcoxon_result = calculate_wilcoxon(df, ['dataset', 'split_ratio'], metric, method1, method2)
        if not wilcoxon_result.empty:
            wilcoxon_results[metric][method2] = wilcoxon_result
            print(f"\nWilcoxon Rank-Sum Test สำหรับ {metric} (เปรียบเทียบ {method1} vs {method2}):")
            print(wilcoxon_result.to_string(index=False))
            print("\n")

# Correlation / throughput_qps / latency_ms
print("=== ความสัมพันธ์ระหว่าง dataset, split_ratio, method กับเมตริกซ์ที่เลือก ===")
correlation_metrics = ['throughput_qps', 'latency_ms']
correlation_results = {}
for metric in correlation_metrics:
    correlation_df = calculate_correlation(df, metric)
    correlation_results[metric] = correlation_df
    print(f"\nความสัมพันธ์กับ {metric}:")
    print(correlation_df.to_string(index=False))
    print("\n")

# save result to file.xlsx
with pd.ExcelWriter('complete_analysis_output.xlsx', engine='openpyxl') as writer:
    for metric in performance_metrics:
        stats[metric].to_excel(writer, sheet_name=f'perf_{metric}', index=False)
    for metric in resource_metrics:
        stats[metric].to_excel(writer, sheet_name=f'resource_{metric}', index=False)
    for metric in additional_metrics:
        stats[metric].to_excel(writer, sheet_name=f'additional_{metric}', index=False)
    for metric in all_metrics:
        if metric in anova_results:
            anova_results[metric].to_excel(writer, sheet_name=f'anova_{metric}', index=False)
        if metric in tukey_results:
            tukey_results[metric].to_excel(writer, sheet_name=f'tukey_{metric}', index=False)
    for metric in all_metrics:
        for method2 in other_methods:
            if metric in wilcoxon_results and method2 in wilcoxon_results[metric]:
                wilcoxon_results[metric][method2].to_excel(writer, sheet_name=f'wilcoxon_{metric}_{method2}', index=False)
    for metric in correlation_metrics:
        correlation_results[metric].to_excel(writer, sheet_name=f'corr_{metric}', index=False)

print("บันทึกผลลัพธ์ทั้งหมดลงในไฟล์ complete_analysis_output.xlsx แล้ว")