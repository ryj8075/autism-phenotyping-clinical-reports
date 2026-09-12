from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

_PH = {".": "\x00", "!": "\x01", "?": "\x02"}
_QUOTE_SPAN = re.compile(r'"[^"]*"|“[^”]*”')

_PAREN_SPAN = re.compile(r'\([^()]*\)')

def _protect_sentence_punct(text: str) -> str:
    text = re.sub(r'(?<=\d)\.(?=\d)', _PH["."], text)
    text = re.sub(r'\bex\.', 'ex' + _PH["."], text, flags=re.IGNORECASE)  # (3) ex.

    def _q(m):
        s = m.group()
        for k, v in _PH.items():
            s = s.replace(k, v)
        return s
    text = _QUOTE_SPAN.sub(_q, text)
    return _PAREN_SPAN.sub(_q, text)

def _restore_sentence_punct(text: str) -> str:
    for k, v in _PH.items():
        text = text.replace(v, k)
    return text

def _clean_korean_text(text: str) -> str:
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'([_\-.])\1+', r'\1', text)
    text = text.replace('*', '')
    text = text.strip()
    return text

def segment_sentences(
    text: str,
    method: str = "tokenizer_aligned",
    min_length: int = 30,
    max_length: int = 500,
) -> List[str]:

    text = text.strip()
    if not text:
        return []

    if method != "tokenizer_aligned":
        raise ValueError(
            f"Unsupported segmentation method: {method!r}. Only 'tokenizer_aligned' is "
            "supported. It reproduces the encoder's segmentation exactly, so that each "
            "sentence index lines up with the corresponding row of the attention matrix."
        )

    return _segment_tokenizer_aligned(text, min_length, max_length)

def _segment_tokenizer_aligned(
    text: str, min_length: int, max_length: int
) -> List[str]:

    text = _clean_korean_text(text)

    text = _protect_sentence_punct(text)

    sentence_endings = r'[.!?]'
    splits = re.split(f'({sentence_endings})', text)

    sentences = []
    for i in range(0, len(splits) - 1, 2):
        if i + 1 < len(splits):
            sentences.append(splits[i] + splits[i + 1])
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

    for i, s in enumerate(result):
        if len(s) > max_length:
            logger.warning(
                "Sentence %d length %d > max_length %d: %.60s...",
                i, len(s), max_length, s,
            )

    return result

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python segment.py <file.txt>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        text = f.read()

    sents = segment_sentences(text)
    for i, s in enumerate(sents):
        print(f"[{i:03d}] {s}")
    print(f"\nTotal sentences: {len(sents)}")
