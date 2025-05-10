import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Загрузка результатов ===

baseline_df = pd.read_csv("/data/arxiv_summarization_results.csv")
abstract_df = pd.read_csv("/data/arxiv_summarization_results_abstract.csv")
extractive_df = pd.read_csv("/data/arxiv_summarization_results_extractive.csv")
finetuned_df = pd.read_csv("/data/finetuned_extract_abstractive_results.csv")

baseline_df["Strategy"] = "Baseline (full_text)"
abstract_df["Strategy"] = "Abstract only"
extractive_df["Strategy"] = "Extractive+Abstractive"
finetuned_df["Strategy"] = "Finetuned Extract+Abstractive"

combined = pd.concat([baseline_df, abstract_df, extractive_df, finetuned_df], ignore_index=True)

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

# === 7. Итоговая таблица по среднему ROUGE-1 для каждой стратегии ===

avg_rouge_by_strategy = (
    combined.groupby("Strategy")["ROUGE-1"]
    .mean()
    .reset_index()
    .sort_values(by="ROUGE-1", ascending=False)
)

print("\n=== Среднее значение ROUGE-1 по стратегиям ===")
print(avg_rouge_by_strategy)

# === 8. Лучшая модель в каждой стратегии по ROUGE-1 ===

best_models = (
    combined.loc[combined.groupby("Strategy")["ROUGE-1"].idxmax()]
    .sort_values(by="ROUGE-1", ascending=False)
    .reset_index(drop=True)
)

print("\n=== Лучшие модели в каждой стратегии (по ROUGE-1) ===")
print(best_models[["Strategy", "Model", "ROUGE-1", "BLEU", "Avg_time_per_sample"]])

# === 9. Сохраняем итоговые таблицы ===

avg_rouge_by_strategy.to_csv("/data/summary_avg_rouge1_by_strategy.csv", index=False)
best_models.to_csv("/data/summary_best_models_by_strategy.csv", index=False)

# === 10. Рейтинг моделей: ROUGE-1 и скорость ===

ranked = combined.copy()
ranked["ROUGE-1_norm"] = ranked["ROUGE-1"] / ranked["ROUGE-1"].max()
ranked["Time_norm"] = 1 - (ranked["Avg_time_per_sample"] / ranked["Avg_time_per_sample"].max())

alpha = 0.7
beta = 0.3
ranked["Composite_Score"] = alpha * ranked["ROUGE-1_norm"] + beta * ranked["Time_norm"]

ranked_sorted = ranked.sort_values(by="Composite_Score", ascending=False).reset_index(drop=True)

print("\n=== Общий рейтинг моделей (качество + скорость) ===")
print(ranked_sorted[["Strategy", "Model", "ROUGE-1", "Avg_time_per_sample", "Composite_Score"]].head(10))

ranked_sorted.to_csv("/data/summary_ranked_models.csv", index=False)

# === 11. Визуализация итогового рейтинга моделей ===

top_n = 10

plt.figure(figsize=(14, 6))
sns.barplot(
    data=ranked_sorted.head(top_n),
    x="Model",
    y="Composite_Score",
    hue="Strategy",
    dodge=False,
    palette="viridis"
)
plt.title(f"Top {top_n} Models by Composite Score (Quality + Speed)")
plt.ylabel("Composite Score (0–1)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/pics/ranked_models_composite_score.png")
plt.show()
