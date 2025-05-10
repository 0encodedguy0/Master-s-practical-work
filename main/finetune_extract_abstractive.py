# 1. Импорт
import pandas as pd
import torch
import re
import time
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, TrainingArguments
)
from datasets import Dataset
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 2. Экстрактивная суммаризация
def extract_summary_per_paragraph(text, n_sentences=3):
    paragraphs = text.strip().split('\n\n')
    summarizer = TextRankSummarizer()
    summaries = []
    for para in paragraphs:
        if not para.strip():
            continue
        parser = PlaintextParser.from_string(para, Tokenizer("english"))
        summary = summarizer(parser.document, n_sentences)
        sentences = [str(s) for s in summary]
        summaries.extend(sentences)
    return ' '.join(summaries)

# 3. Загрузка данных
df = pd.read_json("/data/output.json", lines=True)
df = df[['title', 'abstract', 'full_text']].dropna()
df['extracted'] = df['full_text'].apply(lambda x: extract_summary_per_paragraph(x, n_sentences=2))

# 4. Делим на train/test
train_df = df.iloc[:800]
test_df = df.iloc[800:1000]

# 5. Fine-tuning модель на (abstract → title)
model_checkpoint = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

train_data = Dataset.from_pandas(train_df[['abstract', 'title']].rename(columns={'abstract': 'input_text', 'title': 'target_text'}))

def preprocess(example):
    inputs = tokenizer(example['input_text'], truncation=True, padding='max_length', max_length=512)
    targets = tokenizer(example['target_text'], truncation=True, padding='max_length', max_length=64)
    inputs['labels'] = targets['input_ids']
    return inputs

train_data = train_data.map(preprocess, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned_model",
    evaluation_strategy="no",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_strategy="no",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("🧠 Дообучение модели...")
trainer.train()

# 6. Генерация заголовков на test-части по экстрактивным суммаризациям
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

generated_titles = []
start_times = []
for text in tqdm(test_df['extracted'].tolist(), desc="Generating titles"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    start = time.time()
    output = model.generate(**inputs, max_length=30, num_beams=4)
    end = time.time()
    title = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_titles.append(title)
    start_times.append(end - start)

# 7. Оценка
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

rouge_result = rouge.compute(predictions=generated_titles, references=test_df['title'].tolist())
bleu_result = bleu.compute(predictions=generated_titles, references=[[ref] for ref in test_df['title'].tolist()])

print("\n=== Результаты после дообучения + экстрактивной суммаризации ===")
print(f"ROUGE-1: {rouge_result['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_result['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")
print(f"BLEU: {bleu_result['bleu']:.4f}")
print(f"Avg time per sample: {sum(start_times) / len(start_times):.2f} sec")

# 8. Сохраняем результаты
pd.DataFrame({
    'true_title': test_df['title'],
    'generated_title': generated_titles,
    'extracted_text': test_df['extracted']
}).to_csv("/data/finetuned_extract_abstractive_results.csv", index=False)

