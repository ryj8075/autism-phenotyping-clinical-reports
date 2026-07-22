from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

_SENT_END = re.compile(
    r'(?<=[.!?])'
    r'(?:\s+|$)'
)

_MISSING_SPACE = re.compile(
    r'(?<=[.!?])'
    r'(?=[\uac00-\ud7a3A-Z])'
)

_MULTI_SPACE = re.compile(r'\s+')

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

    if method == "tokenizer_aligned":
        return _segment_tokenizer_aligned(text, min_length, max_length)
    elif method == "kss":
        return _segment_kss(text, min_length, max_length)

    return _segment_regex(text, min_length, max_length)

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

def _segment_regex(
    text: str, min_length: int, max_length: int
) -> List[str]:

    text = _MISSING_SPACE.sub(" ", text)

    raw_parts = _SENT_END.split(text)

    parts: List[str] = []
    for p in raw_parts:
        p = _MULTI_SPACE.sub(" ", p).strip()
        if p:
            parts.append(p)

    sentences = _merge_short(parts, min_length)

    for i, s in enumerate(sentences):
        if len(s) > max_length:
            logger.warning(
                "Sentence %d length %d > max_length %d: %.60s...",
                i, len(s), max_length, s,
            )

    return sentences

def _segment_kss(
    text: str, min_length: int, max_length: int
) -> List[str]:

    try:
        import kss  # type: ignore
    except ImportError:
        logger.warning("kss is not installed; falling back to regex segmentation.")
        return _segment_regex(text, min_length, max_length)

    raw = kss.split_sentences(text)
    parts = [_MULTI_SPACE.sub(" ", s).strip() for s in raw if s.strip()]
    sentences = _merge_short(parts, min_length)

    for i, s in enumerate(sentences):
        if len(s) > max_length:
            logger.warning(
                "Sentence %d length %d > max_length %d: %.60s...",
                i, len(s), max_length, s,
            )

    return sentences

def _merge_short(parts: List[str], min_length: int) -> List[str]:

    if not parts:
        return []

    merged: List[str] = [parts[0]]
    for p in parts[1:]:
        if len(p) < min_length and merged:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)

    if len(merged) > 1 and len(merged[0]) < min_length:
        merged[1] = merged[0] + " " + merged[1]
        merged.pop(0)

    return merged

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
