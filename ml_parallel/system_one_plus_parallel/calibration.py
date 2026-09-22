"""Fit scalar calibration on development data; never use training or test labels."""

import torch
from torch.nn import functional as F


def fit_calibration(model, gates, differences):
    report = {"gate_records": len(gates), "preference_pairs": len(differences)}
    # Small data cannot establish calibrated probabilities; preserve identity.
    if len(gates) >= 4 and len({label for _, label in gates}) == 2:
        z = torch.tensor([z for z, _ in gates], dtype=torch.float64)
        labels = torch.tensor([y for _, y in gates], dtype=torch.float64)
        log_scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_scale, bias], max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            logits = log_scale.clamp(-4, 4).exp() * z + bias.clamp(-10, 10)
            loss = F.binary_cross_entropy_with_logits(logits, labels) + 0.001 * (
                log_scale.square() + bias.square()
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        model.omission_scale.fill_(float(log_scale.detach().clamp(-4, 4).exp()))
        model.omission_bias.fill_(float(bias.detach().clamp(-10, 10)))
        report["gate_fit"] = True
    else:
        report["gate_fit"] = False
    if differences:
        gaps = torch.tensor(differences, dtype=torch.float64)
        # Each preferred/rejected pair provides both directed comparisons.
        logits = torch.cat([gaps, -gaps])
        labels = torch.cat([torch.ones_like(gaps), torch.zeros_like(gaps)])
        log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temperature], max_iter=50, line_search_fn="strong_wolfe")

        def preference_closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(logits / log_temperature.clamp(-4, 4).exp(), labels)
            loss = loss + 0.001 * log_temperature.square()
            loss.backward()
            return loss

        optimizer.step(preference_closure)
        model.rank_temperature.fill_(float(log_temperature.detach().clamp(-4, 4).exp()))
    probabilities = [
        (float(model.missing_probability(torch.tensor(z, device=model.device))), y) for z, y in gates
    ]
    threshold, counts = calibrate_threshold(probabilities)
    report.update(counts)
    report["brier"] = (
        sum((p - y) ** 2 for p, y in probabilities) / len(probabilities) if probabilities else None
    )
    report["rank_temperature"] = float(model.rank_temperature)
    return threshold, report


def calibrate_threshold(rows):
    if not rows:
        return 0.5, {"false_negative_rate": None, "false_positive_rate": None}
    candidates = {0.0, 1.0, *(p for p, _ in rows)}

    def cost(t):
        fn = sum(y == 1 and p < t for p, y in rows)
        fp = sum(y == 0 and p >= t for p, y in rows)
        return fn + fp, fn, fp, t

    _, fn, fp, threshold = min(map(cost, candidates))
    positives = sum(y for _, y in rows)
    negatives = len(rows) - positives
    return threshold, {
        "false_negative_rate": fn / positives if positives else None,
        "false_positive_rate": fp / negatives if negatives else None,
    }
