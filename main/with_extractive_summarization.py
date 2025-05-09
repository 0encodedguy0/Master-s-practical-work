import pandas as pd
import torch
import csv
import time
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from tqdm import tqdm
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# === 1. Чтение данных ===

df = pd.read_json("/data/output.json", orient='records', lines=True)
df = df[['title', 'abstract', 'full_text']]

# === 2. Подготовка моделей ===

device = 0 if torch.cuda.is_available() else -1

model_infos = {
    'facebook/bart-large-cnn': 406,
    'google/pegasus-xsum': 568,
    'philschmid/bart-large-cnn-samsum': 406,
    't5-small': 60,
    't5-base': 220,
    'google/flan-t5-base': 250,
    'google/flan-t5-large': 780
}

summarizers = {}
for model_name in model_infos.keys():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizers[model_name] = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

# === 3. Метрики ===

rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')

# === 4. Экстрактивная суммаризация абзаца ===

def extract_key_sentences(text, max_sentences=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, max_sentences)
    return " ".join([str(sentence) for sentence in summary])

# === 5. Генерация заголовков с экстрактивной суммаризацией ===

def generate_summaries_with_extractive(summarizer, texts, max_length=30):
    generated = []
    times = []
    for text in tqdm(texts, desc="Summarizing"):
        start = time.time()

        # Разбиваем текст на абзацы и применяем TextRank к каждому
        paragraphs = text.split('\n\n')
        key_sentences = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                summary = extract_key_sentences(paragraph, max_sentences=2)
                if summary:
                    key_sentences.append(summary)

        combined_summary = " ".join(key_sentences)[:1024]

        summary = summarizer(combined_summary, max_length=max_length, min_length=5, do_sample=False)[0]['summary_text']

        end = time.time()
        generated.append(summary)
        times.append(end - start)
    return generated, times

# === 6. Основной эксперимент ===

texts = df['full_text'].tolist()[:1000]
true_titles = df['title'].tolist()[:1000]

results = {}

for model_name, summarizer in summarizers.items():
    print(f"\nГенерация с помощью {model_name} с экстрактивной предварительной обработкой...")
    generated_titles, times = generate_summaries_with_extractive(summarizer, texts)

    rouge_output = rouge.compute(predictions=generated_titles, references=true_titles)
    references_bleu = [[ref] for ref in true_titles]
    predictions_bleu = [pred for pred in generated_titles]
    bleu_output = bleu.compute(predictions=predictions_bleu, references=references_bleu)

    results[model_name] = {
        'ROUGE-1': rouge_output['rouge1'],
        'ROUGE-2': rouge_output['rouge2'],
        'ROUGE-L': rouge_output['rougeL'],
        'BLEU': bleu_output['bleu'],
        'Avg_time_per_sample': sum(times) / len(times),
        'Model_params_millions': model_infos[model_name]
    }

results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
print(results_df)

# Сохраняем результаты
results_df.to_csv("/data/arxiv_summarization_results_extractive.csv", index=False)

# === 7. Визуализация ===

sns.set(style="whitegrid")

# ROUGE-1
plt.figure(figsize=(12,6))
sns.barplot(x="Model", y="ROUGE-1", data=results_df, palette="Blues_d")
plt.title("ROUGE-1 Comparison Across Models (with Extractive)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/pics/rouge1_comparison_extractive.png")
plt.show()

# ROUGE-2
plt.figure(figsize=(12,6))
sns.barplot(x="Model", y="ROUGE-2", data=results_df, palette="Greens_d")
plt.title("ROUGE-2 Comparison Across Models (with Extractive)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/pics/rouge2_comparison_extractive.png")
plt.show()

# BLEU
plt.figure(figsize=(12,6))
sns.barplot(x="Model", y="BLEU", data=results_df, palette="Reds_d")
plt.title("BLEU Comparison Across Models (with Extractive)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/pics/bleu_comparison_extractive.png")
plt.show()

# Скорость vs Качество
plt.figure(figsize=(10,6))
sns.scatterplot(x="Avg_time_per_sample", y="ROUGE-1", size="Model_params_millions", hue="Model", data=results_df, legend='brief', sizes=(100, 1000))
plt.title("Speed vs Quality Tradeoff (with Extractive Preprocessing)")
plt.xlabel("Average Time per Sample (sec)")
plt.ylabel("ROUGE-1")
plt.tight_layout()
plt.savefig("/pics/speed_vs_quality_extractive.png")
plt.show()
