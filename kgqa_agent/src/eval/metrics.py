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


def token_f1_score(pred: str, golds: List[str]) -> float:
    """Compute F1 score using set-based matching.
    
    This function:
    1. Parses the prediction string into multiple candidate answers
    2. Merges all prediction candidates into a single token set
    3. Merges all gold answers into a single token set
    4. Computes F1 between the two token sets
    
    Args:
        pred: Prediction string (may contain multiple candidates)
        golds: List of gold answer strings
        
    Returns:
        F1 score (0.0 to 1.0) computed between the merged token sets
    """
    # Parse prediction into multiple candidates
    preds = parse_prediction(pred)
    if not preds:
        return 0.0
    
    # Merge all prediction candidates into a single token set
    pred_tokens = set()
    for p in preds:
        normalized = qa_normalize_answer(p)
        tokens = normalized.split()
        pred_tokens.update(tokens)
    
    if not pred_tokens:
        return 0.0
    
    # Merge all gold answers into a single token set
    gold_tokens = set()
    for g in golds or []:
        normalized = qa_normalize_answer(str(g))
        tokens = normalized.split()
        gold_tokens.update(tokens)
    
    if not gold_tokens:
        return 0.0
    
    # Compute F1 between the two token sets
    common = len(pred_tokens & gold_tokens)
    if common == 0:
        return 0.0
    
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

f1_score = token_f1_score