from typing import Dict, List
import numpy as np


def find_peak_pick(p_prob: np.ndarray, threshold: float = 0.5) -> int:
    peak_idx = int(np.argmax(p_prob))
    peak_val = float(p_prob[peak_idx])
    if peak_val < threshold:
        return -1
    return peak_idx


def classification_from_residuals(
    pred_picks: List[int],
    true_picks: List[int],
    tolerance_samples: int = 40,
) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    residuals = []

    for pred, true in zip(pred_picks, true_picks):
        true_exists = true is not None and int(true) >= 0
        pred_exists = pred is not None and int(pred) >= 0

        if true_exists and pred_exists:
            dt = int(pred) - int(true)
            if abs(dt) <= tolerance_samples:
                tp += 1
                residuals.append(dt)
            else:
                fp += 1
                fn += 1
        elif true_exists and not pred_exists:
            fn += 1
        elif (not true_exists) and pred_exists:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    residuals = np.asarray(residuals, dtype=float)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "residual_mean_samples": float(np.mean(residuals)) if residuals.size > 0 else float("nan"),
        "residual_std_samples": float(np.std(residuals)) if residuals.size > 0 else float("nan"),
        "num_residuals": int(residuals.size),
    }
