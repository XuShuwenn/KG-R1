#!/usr/bin/env python3
"""
Test script to verify the F1 calculation fix.

This tests that KB IDs and text are treated as alternative representations
of the same entity, not as separate entities.
"""

import string

def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


def test_f1_calculation():
    """Test the F1 calculation with KB ID and text representations."""

    # Simulate the calculation from _calculate_entity_level_f1_precision_recall

    # Test Case 1: Single entity with text and KB ID (the bug case)
    print("=" * 80)
    print("Test Case 1: Single entity with text and KB ID")
    print("=" * 80)

    predicted_answer = "Sebastián Piñera"
    ground_truth_raw = {
        'target_kb_id': ['m.064095'],
        'target_text': ['Sebastián Piñera']
    }

    # Parse predicted
    predicted_entities = [e.strip() for e in predicted_answer.split(',')]
    predicted_normalized_set = set()
    for pred_entity in predicted_entities:
        if pred_entity:
            normalized = normalize_answer(pred_entity)
            if normalized:
                predicted_normalized_set.add(normalized)

    print(f"Predicted entities: {predicted_normalized_set}")

    # Parse ground truth
    target_texts = ground_truth_raw.get("target_text", [])
    target_kb_ids = ground_truth_raw.get("target_kb_id", [])

    if not isinstance(target_texts, list):
        target_texts = [target_texts] if target_texts else []
    if not isinstance(target_kb_ids, list):
        target_kb_ids = [target_kb_ids] if target_kb_ids else []

    num_entities = max(len(target_texts), len(target_kb_ids))

    # Build list of ground truth entities (FIXED VERSION)
    ground_truth_entities = []
    for i in range(num_entities):
        entity_representations = set()

        if i < len(target_texts) and target_texts[i]:
            normalized_text = normalize_answer(str(target_texts[i]))
            if normalized_text:
                entity_representations.add(normalized_text)

        if i < len(target_kb_ids) and target_kb_ids[i]:
            normalized_kb = normalize_answer(str(target_kb_ids[i]))
            if normalized_kb:
                entity_representations.add(normalized_kb)

        if entity_representations:
            ground_truth_entities.append(entity_representations)

    print(f"Ground truth entities: {ground_truth_entities}")
    print(f"Number of ground truth entities: {len(ground_truth_entities)}")

    # Match predicted entities against ground truth
    matched_gt_entities = set()
    matched_pred_entities = set()

    for pred_entity in predicted_normalized_set:
        for gt_idx, gt_entity_reps in enumerate(ground_truth_entities):
            if pred_entity in gt_entity_reps:
                matched_gt_entities.add(gt_idx)
                matched_pred_entities.add(pred_entity)
                break

    num_correct = len(matched_pred_entities)
    num_predicted = len(predicted_normalized_set)
    num_ground_truth = len(ground_truth_entities)

    precision = num_correct / num_predicted if num_predicted > 0 else 0.0
    recall = num_correct / num_ground_truth if num_ground_truth > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    print(f"\nMatched predicted entities: {matched_pred_entities}")
    print(f"Matched ground truth entities: {matched_gt_entities}")
    print(f"\nnum_correct: {num_correct}")
    print(f"num_predicted: {num_predicted}")
    print(f"num_ground_truth: {num_ground_truth}")
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")

    # Expected result
    expected_f1 = 1.0
    if abs(f1 - expected_f1) < 0.001:
        print(f"\n✅ TEST PASSED: F1 = {f1:.3f} (expected {expected_f1:.3f})")
    else:
        print(f"\n❌ TEST FAILED: F1 = {f1:.3f} (expected {expected_f1:.3f})")


    # Test Case 2: Multi-entity answer
    print("\n" + "=" * 80)
    print("Test Case 2: Multi-entity answer (comma-separated)")
    print("=" * 80)

    predicted_answer = "Entity A, Entity B"
    ground_truth_raw = {
        'target_kb_id': ['m.001', 'm.002'],
        'target_text': ['Entity A', 'Entity B']
    }

    # Parse predicted
    predicted_entities = [e.strip() for e in predicted_answer.split(',')]
    predicted_normalized_set = set()
    for pred_entity in predicted_entities:
        if pred_entity:
            normalized = normalize_answer(pred_entity)
            if normalized:
                predicted_normalized_set.add(normalized)

    print(f"Predicted entities: {predicted_normalized_set}")

    # Parse ground truth
    target_texts = ground_truth_raw.get("target_text", [])
    target_kb_ids = ground_truth_raw.get("target_kb_id", [])

    if not isinstance(target_texts, list):
        target_texts = [target_texts] if target_texts else []
    if not isinstance(target_kb_ids, list):
        target_kb_ids = [target_kb_ids] if target_kb_ids else []

    num_entities = max(len(target_texts), len(target_kb_ids))

    # Build list of ground truth entities
    ground_truth_entities = []
    for i in range(num_entities):
        entity_representations = set()

        if i < len(target_texts) and target_texts[i]:
            normalized_text = normalize_answer(str(target_texts[i]))
            if normalized_text:
                entity_representations.add(normalized_text)

        if i < len(target_kb_ids) and target_kb_ids[i]:
            normalized_kb = normalize_answer(str(target_kb_ids[i]))
            if normalized_kb:
                entity_representations.add(normalized_kb)

        if entity_representations:
            ground_truth_entities.append(entity_representations)

    print(f"Ground truth entities: {ground_truth_entities}")
    print(f"Number of ground truth entities: {len(ground_truth_entities)}")

    # Match
    matched_gt_entities = set()
    matched_pred_entities = set()

    for pred_entity in predicted_normalized_set:
        for gt_idx, gt_entity_reps in enumerate(ground_truth_entities):
            if pred_entity in gt_entity_reps:
                matched_gt_entities.add(gt_idx)
                matched_pred_entities.add(pred_entity)
                break

    num_correct = len(matched_pred_entities)
    num_predicted = len(predicted_normalized_set)
    num_ground_truth = len(ground_truth_entities)

    precision = num_correct / num_predicted if num_predicted > 0 else 0.0
    recall = num_correct / num_ground_truth if num_ground_truth > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    print(f"\nMatched predicted entities: {matched_pred_entities}")
    print(f"Matched ground truth entities: {matched_gt_entities}")
    print(f"\nnum_correct: {num_correct}")
    print(f"num_predicted: {num_predicted}")
    print(f"num_ground_truth: {num_ground_truth}")
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")

    expected_f1 = 1.0
    if abs(f1 - expected_f1) < 0.001:
        print(f"\n✅ TEST PASSED: F1 = {f1:.3f} (expected {expected_f1:.3f})")
    else:
        print(f"\n❌ TEST FAILED: F1 = {f1:.3f} (expected {expected_f1:.3f})")


    # Test Case 3: Partial match using KB ID
    print("\n" + "=" * 80)
    print("Test Case 3: Predicting KB ID instead of text")
    print("=" * 80)

    predicted_answer = "m.064095"
    ground_truth_raw = {
        'target_kb_id': ['m.064095'],
        'target_text': ['Sebastián Piñera']
    }

    # Parse predicted
    predicted_entities = [e.strip() for e in predicted_answer.split(',')]
    predicted_normalized_set = set()
    for pred_entity in predicted_entities:
        if pred_entity:
            normalized = normalize_answer(pred_entity)
            if normalized:
                predicted_normalized_set.add(normalized)

    print(f"Predicted entities: {predicted_normalized_set}")

    # Parse ground truth
    target_texts = ground_truth_raw.get("target_text", [])
    target_kb_ids = ground_truth_raw.get("target_kb_id", [])

    if not isinstance(target_texts, list):
        target_texts = [target_texts] if target_texts else []
    if not isinstance(target_kb_ids, list):
        target_kb_ids = [target_kb_ids] if target_kb_ids else []

    num_entities = max(len(target_texts), len(target_kb_ids))

    # Build list of ground truth entities
    ground_truth_entities = []
    for i in range(num_entities):
        entity_representations = set()

        if i < len(target_texts) and target_texts[i]:
            normalized_text = normalize_answer(str(target_texts[i]))
            if normalized_text:
                entity_representations.add(normalized_text)

        if i < len(target_kb_ids) and target_kb_ids[i]:
            normalized_kb = normalize_answer(str(target_kb_ids[i]))
            if normalized_kb:
                entity_representations.add(normalized_kb)

        if entity_representations:
            ground_truth_entities.append(entity_representations)

    print(f"Ground truth entities: {ground_truth_entities}")
    print(f"Number of ground truth entities: {len(ground_truth_entities)}")

    # Match
    matched_gt_entities = set()
    matched_pred_entities = set()

    for pred_entity in predicted_normalized_set:
        for gt_idx, gt_entity_reps in enumerate(ground_truth_entities):
            if pred_entity in gt_entity_reps:
                matched_gt_entities.add(gt_idx)
                matched_pred_entities.add(pred_entity)
                break

    num_correct = len(matched_pred_entities)
    num_predicted = len(predicted_normalized_set)
    num_ground_truth = len(ground_truth_entities)

    precision = num_correct / num_predicted if num_predicted > 0 else 0.0
    recall = num_correct / num_ground_truth if num_ground_truth > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    print(f"\nMatched predicted entities: {matched_pred_entities}")
    print(f"Matched ground truth entities: {matched_gt_entities}")
    print(f"\nnum_correct: {num_correct}")
    print(f"num_predicted: {num_predicted}")
    print(f"num_ground_truth: {num_ground_truth}")
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")

    expected_f1 = 1.0
    if abs(f1 - expected_f1) < 0.001:
        print(f"\n✅ TEST PASSED: F1 = {f1:.3f} (expected {expected_f1:.3f})")
    else:
        print(f"\n❌ TEST FAILED: F1 = {f1:.3f} (expected {expected_f1:.3f})")


if __name__ == "__main__":
    test_f1_calculation()
