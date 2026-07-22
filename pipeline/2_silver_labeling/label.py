from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from schemas import (
    DomainDef,
    DomainRegistry,
    LabelItem,
    LLMResponse,
    SentenceLabel,
)

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")

def _extract_json(text: str) -> str:

    m = _JSON_BLOCK.search(text)
    if m:
        return m.group(1).strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return text[start:end]
    return text.strip()

def _repair_json(raw: str) -> Optional[dict]:

    raw = raw.strip()
    # trailing comma before } or ]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    stack = []
    for ch in raw:
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    raw += "".join(reversed(stack))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None

def parse_llm_response(
    text: str,
    valid_domain_ids: Set[str],
    registry: Optional[DomainRegistry] = None,
) -> LLMResponse:

    raw_json = _extract_json(text)
    data = _repair_json(raw_json)

    if data is None:
        logger.warning("Failed to parse JSON: %.200s", text)
        return LLMResponse(labels=[])

    labels_raw = data.get("labels")

    if labels_raw is None:
        label_single = data.get("label")
        if isinstance(label_single, dict):
            logger.warning(
                "LLM returned the legacy multi-class format ('label' as a single object); wrapping it as a list. "
                "Check that the prompt asks for a multi-label 'labels' list."
            )
            labels_raw = [label_single]
        else:
            labels_raw = []

    if not isinstance(labels_raw, list):
        logger.warning("labels is not a list: %s", type(labels_raw))
        return LLMResponse(labels=[])

    items: List[LabelItem] = []
    for label_raw in labels_raw:
        if not isinstance(label_raw, dict):
            logger.debug("labels item is not a dict: %s", type(label_raw))
            continue

        domain_id = str(label_raw.get("domain_id", "")).strip()
        if domain_id not in valid_domain_ids:
            logger.debug("Ignoring invalid domain: %s", domain_id)
            continue

        domain_code = None
        if registry:
            domain_code = registry.id_to_code.get(domain_id)

        try:
            items.append(
                LabelItem(
                    domain_id=domain_id,
                    domain_code=domain_code,
                    confidence=float(label_raw.get("confidence", 0)),
                    rationale=str(label_raw.get("rationale", "")),
                )
            )
        except (ValueError, TypeError) as e:
            logger.debug("Failed to parse label item: %s - %s", label_raw, e)
            continue

    return LLMResponse(labels=items)

class Labeler:

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.llm_cfg = config["llm"]
        self.prompt_cfg = config["prompt"]
        self.threshold = config["threshold"]
        self.domains: List[DomainDef] = [
            DomainDef(
                code=str(d.get("code", "")),
                id=d["id"],
                name_ko=d.get("name_ko", ""),
                name_en=d.get("name_en", ""),
                description=d.get("description", ""),
                pilot_active=d.get("pilot_active", True),
                category=d.get("category", ""),
                subcategory=d.get("subcategory", ""),
            )
            for d in config["domains"]
        ]
        self.registry = DomainRegistry(self.domains)
        self.valid_ids = {d.id for d in self.domains}
        self._client = None

    def _domain_list_str(self) -> str:
        lines = []
        for d in self.domains:
            lines.append(
                f"- {d.id} ({d.code}. {d.name_ko} / {d.name_en}): {d.description.strip()}"
            )
        return "\n".join(lines)

    def _get_client(self):
        if self._client is not None:
            return self._client

        provider = self.llm_cfg.get("provider", "openai")
        if provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Install the openai package: pip install openai")

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("Set the OPENAI_API_KEY environment variable.")
            self._client = OpenAI(api_key=api_key)

        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError("Install the anthropic package: pip install anthropic")

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("Set the ANTHROPIC_API_KEY environment variable.")
            timeout = self.llm_cfg.get("timeout_sec", 60)
            self._client = Anthropic(
                api_key=api_key,
                timeout=float(timeout),
            )

        elif provider == "local":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Install the openai package: pip install openai")

            base_url = self.llm_cfg.get("base_url", "http://localhost:11434/v1")
            self._client = OpenAI(api_key="local", base_url=base_url)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        return self._client

    def _call_llm(self, sentence: str, seed: Optional[int] = None) -> str:

        client = self._get_client()
        provider = self.llm_cfg.get("provider", "openai")

        system_msg = self.prompt_cfg["system"]
        user_msg = self.prompt_cfg["user_template"].format(
            domain_list=self._domain_list_str(),
            sentence=sentence,
        )

        max_retries = self.llm_cfg.get("max_retries", 3)
        retry_delay = self.llm_cfg.get("retry_delay_sec", 2)

        for attempt in range(1, max_retries + 1):
            try:
                if provider == "openai":
                    kwargs: Dict[str, Any] = dict(
                        model=self.llm_cfg["model"],
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=self.llm_cfg.get("temperature", 0.0),
                        max_tokens=self.llm_cfg.get("max_tokens", 1024),
                        timeout=self.llm_cfg.get("timeout_sec", 60),
                    )
                    if seed is not None:
                        kwargs["seed"] = seed
                    resp = client.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content or ""

                elif provider == "anthropic":
                    resp = client.messages.create(
                        model=self.llm_cfg["model"],
                        system=system_msg,
                        messages=[{"role": "user", "content": user_msg}],
                        temperature=self.llm_cfg.get("temperature", 0.0),
                        max_tokens=self.llm_cfg.get("max_tokens", 1024),
                    )
                    return resp.content[0].text if resp.content else ""

                elif provider == "local":
                    kwargs = dict(
                        model=self.llm_cfg["model"],
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=self.llm_cfg.get("temperature", 0.0),
                        max_tokens=self.llm_cfg.get("max_tokens", 1024),
                        timeout=self.llm_cfg.get("timeout_sec", 60),
                    )
                    resp = client.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content or ""

            except Exception as e:
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s", attempt, max_retries, e
                )
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)

        raise RuntimeError(f"All {max_retries} LLM calls failed")

    def label_sentence(
        self,
        sentence: str,
        report_id: str,
        sentence_idx: int,
    ) -> SentenceLabel:

        n = self.llm_cfg.get("self_consistency_n", 1)
        seed = self.llm_cfg.get("seed", 42)
        conf_min = self.threshold.get("confidence_min", 0.5)
        sc_ratio = self.threshold.get("self_consistency_ratio", 0.5)

        raw_responses: List[LLMResponse] = []

        for i in range(n):
            try:
                call_seed = seed + i if seed is not None else None
                raw_text = self._call_llm(sentence, seed=call_seed)
                parsed = parse_llm_response(raw_text, self.valid_ids, self.registry)
                raw_responses.append(parsed)
            except Exception as e:
                logger.error("Sentence labeling failed [%s:%d]: %s", report_id, sentence_idx, e)
                return SentenceLabel(
                    report_id=report_id,
                    sentence_idx=sentence_idx,
                    sentence=sentence,
                    model=self.llm_cfg["model"],
                    seed=seed,
                    error=str(e),
                )

        domain_votes: Counter = Counter()
        domain_confs: Dict[str, List[float]] = {}
        domain_rationales: Dict[str, List[str]] = {}

        for resp in raw_responses:
            seen_ids: Set[str] = set()
            for item in resp.labels:
                did = item.domain_id
                if did in seen_ids:
                    continue
                seen_ids.add(did)
                domain_votes[did] += 1
                domain_confs.setdefault(did, []).append(item.confidence)
                domain_rationales.setdefault(did, []).append(item.rationale)

        final_labels: List[LabelItem] = []
        for did, count in domain_votes.items():
            ratio = count / n
            confs = domain_confs[did]
            avg_conf = sum(confs) / len(confs)

            if ratio >= sc_ratio and avg_conf >= conf_min:
                best_rat = max(domain_rationales[did], key=len, default="")
                final_labels.append(
                    LabelItem(
                        domain_id=did,
                        domain_code=self.registry.id_to_code.get(did),
                        confidence=round(avg_conf, 3),
                        rationale=best_rat,
                    )
                )

        final_labels.sort(key=lambda it: it.confidence, reverse=True)

        max_labels = self.threshold.get("max_labels", 0)
        if max_labels and max_labels > 0:
            final_labels = final_labels[:max_labels]

        if not final_labels:
            fallback_id = self.registry.fallback_id

            all_confs = [c for confs in domain_confs.values() for c in confs]
            avg_all = round(sum(all_confs) / len(all_confs), 3) if all_confs else 0.0
            final_labels = [
                LabelItem(
                    domain_id=fallback_id,
                    domain_code=self.registry.id_to_code.get(fallback_id),
                    confidence=avg_all,
                    rationale="Below threshold; using fallback",
                )
            ]

        return SentenceLabel(
            report_id=report_id,
            sentence_idx=sentence_idx,
            sentence=sentence,
            labels=final_labels,
            raw_responses=raw_responses if n > 1 else None,
            model=self.llm_cfg["model"],
            seed=seed,
        )
