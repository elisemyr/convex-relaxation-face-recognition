"""
Metrics for identification and for the door decision

Identification accuracy: share of employee faces given the right name

Door decision, for a threshold tau:
- false reject rate, FRR: share of employee faces that are NOT
  "correctly identified with score >= tau". A wrong name also counts as a failure, because the system would open the door for the wrong person
- false accept rate, FAR: share of visitor faces with score >= tau

Lowering tau lets more people in: FRR goes down and FAR goes up
The client chooses the balance. We pick tau on the validation visitors
and report FAR and FRR on the test visitors.
"""

import numpy as np

# np.trapezoid is the numpy 2 name, np.trapz the older one
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def accuracy(pred, true):
    return float(np.mean(np.asarray(pred) == np.asarray(true)))


def far_frr(tau, emp_pred, emp_true, emp_score, vis_score):
    """FAR and FRR at one threshold"""
    emp_ok = (np.asarray(emp_pred) == np.asarray(emp_true)) & (np.asarray(emp_score) >= tau)
    frr = 1.0 - emp_ok.mean()
    far = float(np.mean(np.asarray(vis_score) >= tau))
    return far, float(frr)


def far_frr_curve(emp_pred, emp_true, emp_score, vis_score, n_points=200):
    """FAR and FRR for many thresholds. Returns taus, far, frr as arrays"""
    all_scores = np.concatenate([emp_score, vis_score])
    taus = np.unique(np.quantile(all_scores, np.linspace(0, 1, n_points)))
    taus = np.concatenate([taus, [np.inf]])
    far, frr = zip(*(far_frr(t, emp_pred, emp_true, emp_score, vis_score) for t in taus))
    return taus, np.array(far), np.array(frr)


def roc_auc(far, frr):
    """Area under the curve of 1 - FRR against FAR. 1 is perfect"""
    order = np.argsort(far)
    return float(_trapezoid(1.0 - np.asarray(frr)[order], np.asarray(far)[order]))


def choose_tau(vis_score, target_far):
    """
    Smallest threshold whose FAR on these visitor scores is <= target_far

    With n visitor images, at most floor(target_far * n) of them may pass
    With few visitor images, a 1% target means that none may pass
    """
    s = np.sort(np.asarray(vis_score))[::-1]
    allowed = int(np.floor(target_far * len(s)))
    if allowed >= len(s):
        return -np.inf
    # Just above the score of the first visitor that must be refused
    return float(np.nextafter(s[allowed], np.inf))


def equal_error_rate(far, frr):
    """The point where FAR = FRR. A single number to compare methods"""
    i = int(np.argmin(np.abs(np.asarray(far) - np.asarray(frr))))
    return float((far[i] + frr[i]) / 2.0)