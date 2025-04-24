import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Load and preprocess dataset
df = pd.read_excel("experiment_data.xlsx")

# Rename methods for better display in graphs
df["method"] = df["method"].replace({
    "Cracking (Adaptive Partitioning)": "Cracking",
    "HSM Proactive Indexing with ML": "HSM-ML",
    "HSM-ProactiveIndex-ML": "HSM-ML"
})

methods_to_include = ["B-Tree", "Cracking", "LearnedIndex", "HSM-ML"]
filtered_df = df[(df["method"].isin(methods_to_include)) & (df["dataset"] == "OpenSky")]

# STEP 2: Prepare summarized data
summary = filtered_df.groupby(["dataset", "method"]).agg({
    "latency_ms": "mean",
    "throughput_qps": "mean"
}).reset_index()

opensky_data = summary[summary["dataset"] == "OpenSky"]
opensky_split = filtered_df
hsm_ml_df = opensky_split[opensky_split["method"] == "HSM-ML"]
overhead_avg = hsm_ml_df.groupby("split_ratio").agg({"overhead_s": "mean"}).reset_index()

# STEP 3: Utility to save figures
def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close()

# STEP 4: Plot and Save

# Graph 1: Latency & Throughput (All Datasets)
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for dataset in summary["dataset"].unique():
    data = summary[summary["dataset"] == dataset]
    axs[0].plot(data["method"], data["latency_ms"], marker='o', label=dataset)
    axs[1].plot(data["method"], data["throughput_qps"], marker='o', label=dataset)
axs[0].set_title("Average Latency (All Datasets)")
axs[0].set_ylabel("Latency (ms)")
axs[0].legend()
axs[0].grid(True)
axs[1].set_title("Average Throughput (All Datasets)")
axs[1].set_ylabel("Throughput (Qps)")
axs[1].legend()
axs[1].grid(True)
save_plot(fig, "Graph_1.png")

# Graph 2: Latency & Throughput (OpenSky)
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=opensky_data, x="method", y="latency_ms", ax=axs[0], color="skyblue")
axs[0].set_title("Latency (OpenSky)")
axs[0].set_ylabel("Latency (ms)")
sns.barplot(data=opensky_data, x="method", y="throughput_qps", ax=axs[1], color="lightgreen")
axs[1].set_title("Throughput (OpenSky)")
axs[1].set_ylabel("Throughput (Qps)")
save_plot(fig, "Graph_2.png")

# Graph 3: Boxplots of Latency & Throughput by Split Ratio
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=opensky_split, x="split_ratio", y="latency_ms", hue="method", ax=axs[0])
axs[0].set_title("Latency Distribution by Split Ratio (OpenSky)")
axs[0].set_ylabel("Latency (ms)")
axs[0].legend_.remove()
sns.boxplot(data=opensky_split, x="split_ratio", y="throughput_qps", hue="method", ax=axs[1])
axs[1].set_title("Throughput Distribution by Split Ratio (OpenSky)")
axs[1].set_ylabel("Throughput (Qps)")
axs[1].legend_.remove()
save_plot(fig, "Graph_3.png")

# Graph 4: Overhead Bar Chart and Scatter Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=overhead_avg, x="split_ratio", y="overhead_s", color="#FFA07A", ax=axs[0])
axs[0].set_title("Overhead by Split Ratio (HSM-ML)")
axs[0].set_ylabel("Overhead (s)")
sns.scatterplot(data=hsm_ml_df, x="overhead_s", y="throughput_qps", hue="split_ratio",
                palette="viridis", s=100, ax=axs[1])
axs[1].set_title("Overhead vs Throughput (HSM-ML)")
axs[1].set_xlabel("Overhead (s)")
axs[1].set_ylabel("Throughput (Qps)")
axs[1].legend(title="Split Ratio")
save_plot(fig, "Graph_4.png")
