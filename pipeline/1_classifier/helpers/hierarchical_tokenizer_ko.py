import re
import math
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer

_PH = {".": "\x00", "!": "\x01", "?": "\x02"}
_QUOTE_SPAN = re.compile(r'"[^"]*"|“[^”]*”')

_PAREN_SPAN = re.compile(r'\([^()]*\)')

def _protect_sentence_punct(text):
    text = re.sub(r'(?<=\d)\.(?=\d)', _PH["."], text)
    text = re.sub(r'\bex\.', 'ex' + _PH["."], text, flags=re.IGNORECASE)

    def _q(m):
        s = m.group()
        for k, v in _PH.items():
            s = s.replace(k, v)
        return s
    text = _QUOTE_SPAN.sub(_q, text)
    return _PAREN_SPAN.sub(_q, text)

def _restore_sentence_punct(text):
    for k, v in _PH.items():
        text = text.replace(v, k)
    return text

def split_sentences_korean(input_string, min_length=30):

    input_string = _protect_sentence_punct(input_string)

    sentence_endings = r'[.!?]'

    splits = re.split(f'({sentence_endings})', input_string)

    sentences = []
    for i in range(0, len(splits)-1, 2):
        if i+1 < len(splits):
            sentences.append(splits[i] + splits[i+1])
        else:
            sentences.append(splits[i])

    if len(splits) % 2 == 1 and splits[-1].strip():
        sentences.append(splits[-1])

    sentences = [s.strip() for s in sentences if s.strip()]

    result = []
    for sent in sentences:
        if len(sent) > min_length or len(result) == 0:
            result.append(sent)
        else:
            if result:
                result[-1] += ' ' + sent
            else:
                result.append(sent)

    result = [_restore_sentence_punct(s) for s in result]

    return result

def clean_korean_text(text):

    text = re.sub(r'\n+', ' ', text)

    text = re.sub(r'\t+', ' ', text)

    text = re.sub(r' +', ' ', text)

    text = re.sub(r'([_\-.])\1+', r'\1', text)

    text = text.replace('*', '')

    text = text.strip()

    return text

def process_clean_data_korean(report_files, meta_frame, sentence_min_length=30):

    data_dict = {}

    for file in report_files:

        report_id = file.split('/')[-1].split('.')[0]

        if report_id in meta_frame.code.values:
            final_diag = meta_frame[meta_frame.code == report_id].final_diag.values[0]

            if not math.isnan(final_diag):
                final_diag = int(final_diag)

                with open(file, "r", encoding="utf-8") as f:
                    report = f.read()

                clean_report = clean_korean_text(report)

                sentences = split_sentences_korean(clean_report, min_length=sentence_min_length)

                pat_id = report_id[:12]

                for count, sentence in enumerate(sentences):
                    data_dict[f"{report_id}_{count}"] = [sentence, final_diag, report_id, pat_id]

    data_df = pd.DataFrame(data_dict).T
    data_df.columns = ["text", "labels", "report_id", "pat_id"]

    return data_df

def hierarchical_tokenizer_korean(
    report_files,
    meta_frame,
    tokenizer_name="klue/roberta-base",
    sentence_max_length=64,
    sentence_min_length=30,
    report_max_length=64
):

    data_df = process_clean_data_korean(report_files, meta_frame, sentence_min_length)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        model_max_length=sentence_max_length
    )

    data_df['tokenized_sentences'] = data_df['text'].apply(
        lambda x: tokenizer(x, truncation=True, padding="max_length", max_length=sentence_max_length)
    )

    all_input_ids = []
    all_attention_mask = []
    all_diagnosis = []
    all_report_ids = []

    for report_id in data_df.report_id.unique():
        report_data = data_df[data_df.report_id == report_id]
        report_tokenized = report_data.tokenized_sentences
        report_diagnosis = report_data.labels.iloc[0]
        num_sentences = len(report_tokenized)

        report_input_ids = np.stack(report_tokenized.map(lambda v: v["input_ids"]))
        report_attention_mask = np.stack(report_tokenized.map(lambda v: v["attention_mask"]))

        if num_sentences > report_max_length:

            padded_input_ids = report_input_ids[:report_max_length]
            padded_attention_mask = report_attention_mask[:report_max_length]
        else:

            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

            padding = np.full(
                (report_max_length - num_sentences, sentence_max_length),
                fill_value=pad_token_id
            )
            attention_padding = np.zeros(
                (report_max_length - num_sentences, sentence_max_length),
                dtype=np.int64
            )

            padded_input_ids = np.vstack((report_input_ids, padding))
            padded_attention_mask = np.vstack((report_attention_mask, attention_padding))

        all_input_ids.append(padded_input_ids)
        all_attention_mask.append(padded_attention_mask)
        all_diagnosis.append(report_diagnosis)
        all_report_ids.append(report_id)

    all_input_ids = torch.tensor(np.stack(all_input_ids))
    all_attention_mask = torch.tensor(np.stack(all_attention_mask))
    all_diagnosis = torch.tensor(all_diagnosis)
    all_report_ids = np.array(all_report_ids)

    return all_input_ids, all_attention_mask, all_diagnosis, all_report_ids
