import os
import json
import pandas as pd
import re
import unicodedata


# Расширенный набор аббревиатур
abbreviations = {
    # латинские
    'e.g', 'E.G', 'E.g', 'i.e', 'I.e', 'I.E', 'cf', 'etc', 'et al', 'viz', 'ibid', 'vs',

    # титулы и степени
    'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'PhD', 'Ph.D', 'B.Sc', 'M.Sc',

    # военные, церковные, профессиональные
    'Gen', 'Col', 'Capt', 'Sgt', 'Maj', 'Rev', 'Hon', 'Eng',

    # юридические / организационные
    'Inc', 'Ltd', 'Co', 'Corp', 'Univ', 'Dept', 'Assoc',

    # технические
    'Fig', 'Eq', 'Ref', 'Vol', 'pp', 'No', 'Ch', 'Sec', 'Art',

    # даты
    'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun',

    # временные обозначения / страны
    'A.M', 'P.M', 'U.S', 'U.K', 'U.N'
}


def is_abbreviation(line):
    # Берём последнее слово перед точкой
    match = re.search(r'(\b[\w.]+)\.$', line.strip())
    if not match:
        return False
    word = match.group(1).replace('.', '')
    return word in abbreviations


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = text.split('\n')
    processed_lines = []

    for i in range(len(lines)):
        line = lines[i].strip()
        if not line:
            continue

        if line.endswith('.') and i + 1 < len(lines) and lines[i + 1].strip():
            if not is_abbreviation(line):
                line += '\n'  # предполагаемый конец абзаца

        processed_lines.append(line)

    full_text = ' '.join(processed_lines)

    # Разбиваем по предполагаемым абзацам
    full_text = re.sub(r'\n+', '\n\n', full_text)

    # Очищаем от мусора, не трогая нужную пунктуацию
    full_text = re.sub(r'[^\w\s.,!?;:()\[\]{}«»"-]', '', full_text)
    full_text = re.sub(r'[ \t]+', ' ', full_text)

    return full_text.strip()


def normalize_for_matching(text):
    if not isinstance(text, str):
        return ''

    # Unicode нормализация и замена лигатур вручную
    text = unicodedata.normalize('NFKD', text)
    text = text.replace('ﬀ', 'ff').replace('ﬁ', 'fi').replace('ﬂ', 'fl') \
               .replace('ﬃ', 'ffi').replace('ﬄ', 'ffl').replace('ﬅ', 'st').replace('ﬆ', 'st')

    # Удаляем все акценты и символы вне ASCII
    text = text.encode('ASCII', 'ignore').decode('ASCII')

    # Убираем переносы и дефисы на границах слов
    text = re.sub(r'[\n\r\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.lower().strip()


def remove_title_abstract_texts(title, abstract, full_text):

    full_text = full_text.replace('ﬀ', 'ff').replace('ﬁ', 'fi').replace('ﬂ', 'fl') \
               .replace('ﬃ', 'ffi').replace('ﬄ', 'ffl').replace('ﬅ', 'st').replace('ﬆ', 'st')

    if title in full_text:
        full_text = full_text.replace(title, '')
    
    if abstract in full_text:
        full_text = full_text.replace(abstract, '')
    
    return full_text


def remove_title_abstract_block(full_text, title, abstract):
    norm_full = normalize_for_matching(full_text)
    norm_title = normalize_for_matching(title)
    norm_abstract = normalize_for_matching(abstract)

    start_idx = norm_full.find(norm_title)
    end_idx = norm_full.find(norm_abstract)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        # Берём примерно первые 30 символов оригинального текста для поиска
        snippet_start = re.escape(title[:100])
        snippet_end = re.escape(abstract[-100:])

        raw_start = re.search(snippet_start, full_text, re.IGNORECASE)
        raw_end = re.search(snippet_end, full_text, re.IGNORECASE | re.DOTALL)

        if raw_start and raw_end:
            start_pos = raw_start.start()
            end_pos = raw_end.end()
            return full_text[:start_pos] + full_text[end_pos:]

    return full_text


def process_jsonl_files(folder_path):
    data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        title_clean = clean_text(entry.get('title', ''))
                        abstract_clean = clean_text(entry.get('abstract', ''))
                        full_text_raw = entry.get('full_text', '')
                        full_text_trimmed = remove_title_abstract_block(full_text_raw, title_clean, abstract_clean)
                        full_text_clean = clean_text(full_text_trimmed)
                        full_text_clean = remove_title_abstract_texts(title_clean, abstract_clean, full_text_clean)

                        cleaned_entry = {
                            'title': title_clean,
                            'abstract': abstract_clean,
                            'pdf_url': entry.get('pdf_url', ''),
                            'full_text': full_text_clean
                        }
                        data.append(cleaned_entry)
                    except json.JSONDecodeError:
                        print(f"Ошибка парсинга строки в файле: {filename}")
                        continue

    return pd.DataFrame(data)


# Задайте путь к папке с .jsonl
folder_path = '/data'
df = process_jsonl_files(folder_path)

# Сохраняем в CSV
df.to_json('/data/output.json', orient='records', lines=True)
