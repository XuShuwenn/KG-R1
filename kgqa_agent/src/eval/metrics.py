"""Evaluation metrics for KGQA tasks.

Includes:
- Token-based F1 (normalized)
- Relaxed EM (normalized exact + bidirectional substring)
"""
from __future__ import annotations
import re
import json
from typing import List, Dict, Any, Tuple, Optional
import string

def qa_normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an)\b", " ", s)
    s = " ".join(s.split())
    return s

def parse_prediction(pred: str) -> List[str]:
    if not pred:
        return []
    clean = pred.strip()
    
    # Try parsing as JSON list
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            return [str(p).strip() for p in parsed if p]
        # If it's a single string in JSON format (unlikely but possible if model outputs "Answer")
        return [str(parsed).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback for legacy formats or malformed JSON
    # If prediction uses pipe separators: "A | B | C"
    if "|" in clean:
        return [p.strip() for p in clean.split("|") if p.strip()]
    
    # Fallback: return the whole prediction as single candidate
    return [clean]

def _single_exact_match(pred: str, golds: List[str]) -> float:
    """Compute EM for a single prediction string against golds."""
    if not pred:
        return 0.0

    if isinstance(golds, str):
        gold_list = [golds]
    else:
        gold_list = golds or []

    npred = qa_normalize_answer(pred)
    for g in gold_list:
        if qa_normalize_answer(str(g)) == npred:
            return 1.0
    for g in gold_list:
        if qa_normalize_answer(str(g)) in npred:
            return 1.0
    for g in gold_list:
        if npred and (npred in qa_normalize_answer(str(g))):
            return 1.0
    return 0.0

def exact_match(pred: str, golds: List[str]) -> float:
    """Compute best EM across all parsed predictions."""
    preds = parse_prediction(pred)
    if not preds:
        return 0.0
    return max(_single_exact_match(p, golds) for p in preds)


def _single_token_f1_score(pred: str, golds: List[str]) -> float:
    """Compute F1 for a single prediction string against golds."""
    if not pred:
        return 0.0
    p_tokens = qa_normalize_answer(pred).split()
    p_set = set(p_tokens)
    if not p_set:
        return 0.0

    best = 0.0
    for g in golds or []:
        g_tokens = qa_normalize_answer(str(g)).split()
        g_set = set(g_tokens)
        if not g_set:
            continue
        common = len(p_set & g_set)
        if common == 0:
            f1 = 0.0
        else:
            prec = common / len(p_set)
            rec = common / len(g_set)
            f1 = 2 * prec * rec / (prec + rec)
        if f1 > best:
            best = f1
    return best

def token_f1_score(pred: str, golds: List[str]) -> float:
    """Compute best F1 across all parsed predictions."""
    preds = parse_prediction(pred)
    if not preds:
        return 0.0
    return max(_single_token_f1_score(p, golds) for p in preds)

f1_score = token_f1_score