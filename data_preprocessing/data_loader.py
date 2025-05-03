import os
import json
import time
import requests
import feedparser
import fitz  # PyMuPDF
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


MATH_CLASSES = [
    'math.' + cat for cat in [
        'AC', 'AG', 'AP', 'AT', 'CA', 'CO', 'CT', 'CV', 'DG', 'DS',
        'FA', 'GM', 'GN', 'GR', 'GT', 'HO', 'IT', 'KT', 'LO',
        'MG', 'MP', 'NA', 'NT', 'OA', 'OC', 'PR', 'QA', 'RA',
        'RT', 'SG', 'SP', 'ST', 'math-ph'
    ]
]

CS_CLASSES = [
    'cs.' + cat for cat in [
        'AI', 'AR', 'CC', 'CE', 'CG', 'CL', 'CR', 'CV', 'CY', 'DB',
        'DC', 'DL', 'DM', 'DS', 'ET', 'FL', 'GL', 'GR', 'GT', 'HC',
        'IR', 'IT', 'LG', 'LO', 'MA', 'MM', 'MS', 'NA', 'NE', 'NI',
        'OH', 'OS', 'PF', 'PL', 'RO', 'SC', 'SD', 'SE', 'SI', 'SY',
    ]
]

PHYSICS_CLASSES = [
    'physics.' + cat for cat in [
        'acc-ph', 'ao-ph', 'app-ph', 'atm-clus', 'atom-ph', 'bio-ph',
        'chem-ph', 'class-ph', 'comp-ph', 'data-an', 'ed-ph', 'flu-dyn',
        'gen-ph', 'geo-ph', 'hist-ph', 'ins-det', 'med-ph', 'optics',
        'plasm-ph', 'pop-ph', 'soc-ph', 'space-ph',
    ]
]

Q_BIO_CLASSES = [
    'q-bio.' + cat for cat in [
        'BM', 'CB', 'GN', 'MN', 'NC', 'OT', 'PE', 'QM', 'SC', 'TO',
    ]
]

STAT_CLASSES = [
    'stat.' + cat for cat in [
        'AP', 'CO', 'ME', 'ML', 'OT', 'TH',
    ]
]
CATEGORIES = MATH_CLASSES + CS_CLASSES + PHYSICS_CLASSES + Q_BIO_CLASSES + STAT_CLASSES
ARTICLES_TO_DOWNLOAD = 2000
MAX_PER_PAGE = 100
THREADS = 8
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def search_arxiv(categories, start_index, max_results):
    base_url = 'http://export.arxiv.org/api/query?'
    query = '+OR+'.join([f"cat:{c}*" for c in categories])
    url = f"{base_url}search_query={query}&start={start_index}&max_results={max_results}&sortOrder=descending"
    return feedparser.parse(url)


def download_and_process(entry):
    title = entry.title.strip()
    abstract = entry.summary.strip()
    pdf_url = entry.link.replace('abs', 'pdf') + ".pdf"
    pdf_filename = os.path.join(OUTPUT_DIR, pdf_url.split("/")[-1])

    try:
        r = requests.get(pdf_url, stream=True, timeout=30)
        if r.status_code != 200:
            return None

        with open(pdf_filename, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

        doc = fitz.open(pdf_filename)
        full_text = "\n".join(page.get_text() for page in doc)
        if len(full_text) < 500:
            return None

        return {
            "title": title,
            "abstract": abstract,
            "pdf_url": pdf_url,
            "full_text": full_text
        }

    except Exception:
        return None
    

def delete_pdf_files(folder_path):
    # Проходим по всем файлам в указанной папке
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):  # Проверяем расширение файла
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)  # Удаляем файл
                print(f"Файл {filename} был удалён.")
            except Exception as e:
                print(f"Ошибка при удалении файла {filename}: {e}")
    

def main():
    output_path = os.path.join(OUTPUT_DIR, "articles.jsonl")
    all_entries = []
    start = 0

    print("Шаг 1: Сбор метаданных статей с arXiv...")
    while len(all_entries) < ARTICLES_TO_DOWNLOAD:
        print(f" → Страница: start={start}")
        feed = search_arxiv(CATEGORIES, start_index=start, max_results=MAX_PER_PAGE)
        entries = feed.entries
        if not entries:
            print("Больше записей не найдено.")
            break

        all_entries.extend(entries)
        print(f"  Загружено статей: {len(all_entries)}")
        start += MAX_PER_PAGE
        time.sleep(3)

    # Ограничим до нужного количества
    all_entries = all_entries[:ARTICLES_TO_DOWNLOAD]

    print("Шаг 2: Многопоточная загрузка PDF и извлечение текста...")
    saved = 0
    with open(output_path, 'w', encoding='utf-8') as out_file:
        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            futures = [executor.submit(download_and_process, entry) for entry in all_entries]

            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    saved += 1

    pdfs_path = './data'
    delete_pdf_files(pdfs_path)
    print(f"\nГотово! Сохранено {saved} статей в {output_path}")


if __name__ == "__main__":
    main()