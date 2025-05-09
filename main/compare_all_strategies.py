import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Загрузка результатов ===

baseline_df = pd.read_csv("/data/arxiv_summarization_results.csv")
abstract_df = pd.read_csv("/data/arxiv_summarization_results_abstract.csv")
extractive_df = pd.read_csv("/data/arxiv_summarization_results_extractive.csv")

baseline_df["Strategy"] = "Baseline (full_text)"
abstract_df["Strategy"] = "Abstract only"
extractive_df["Strategy"] = "Extractive+Abstractive"

combined = pd.concat([baseline_df, abstract_df, extractive_df], ignore_index=True)

# === 2. Визуализация ROUGE-1 ===

plt.figure(figsize=(14,6))
sns.barplot(x="Model", y="ROUGE-1", hue="Strategy", data=combined)
plt.title("ROUGE-1 Across Strategies")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/pics/compare_rouge1_all_strategies.png")
plt.show()

# === 3. Визуализация ROUGE-2 ===

plt.figure(figsize=(14,6))
sns.barplot(x="Model", y="ROUGE-2", hue="Strategy", data=combined)
plt.title("ROUGE-2 Across Strategies")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/pics/compare_rouge2_all_strategies.png")
plt.show()

# === 4. Визуализация BLEU ===

plt.figure(figsize=(14,6))
sns.barplot(x="Model", y="BLEU", hue="Strategy", data=combined)
plt.title("BLEU Score Across Strategies")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/pics/compare_bleu_all_strategies.png")
plt.show()

# === 5. Визуализация времени выполнения ===

plt.figure(figsize=(14,6))
sns.barplot(x="Model", y="Avg_time_per_sample", hue="Strategy", data=combined)
plt.title("Average Generation Time per Sample")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Time (sec)")
plt.tight_layout()
plt.savefig("/pics/compare_time_all_strategies.png")
plt.show()

# === 6. Визуализация: скорость vs качество (ROUGE-1) ===

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=combined,
    x="Avg_time_per_sample",
    y="ROUGE-1",
    hue="Strategy",
    style="Model",
    size="Model_params_millions",
    sizes=(100, 800),
)
plt.title("Speed vs Quality Tradeoff (ROUGE-1)")
plt.xlabel("Average Time per Sample (sec)")
plt.ylabel("ROUGE-1")
plt.tight_layout()
plt.savefig("/pics/compare_speed_vs_quality_all_strategies.png")
plt.show()
