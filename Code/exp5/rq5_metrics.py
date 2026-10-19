"""Survey-based ranking metrics: Remove Precision, Retain Recall, and nDCG@k.

- human label per hunk = median of 3 reviewers, retain=2, not certain=1, reject=0
- per issue: n hunks, removal budget b = number of hunks with median reject, k = n - b
- a method gives each hunk a keep-score (higher = keep); the b lowest are "removed"
- Remove Precision: fraction of the b removed hunks whose label is reject (issues with b > 0)
- Retain Recall: fraction of retain hunks that stay in the top-k (issues with >= 1 retain hunk)
- macro-average over issues; issue-level bootstrap 95% CI
"""
import math, random


def _disc(i):  # i is 0-based rank
    return 1.0 / math.log2(i + 2)


def dcg(gains):
    return sum(g * _disc(i) for i, g in enumerate(gains))


def issue_metrics(labels, scores, budget):
    """labels, scores: parallel lists for one issue. Returns (ndcg, remove_prec|None, mr_recall|None)."""
    n = len(labels); k = n - budget
    order = sorted(range(n), key=lambda i: -scores[i])
    kept, removed = order[:k], order[k:]
    ideal = dcg(sorted(labels, reverse=True)[:k])
    nd = dcg([labels[i] for i in kept]) / ideal if ideal > 0 else 1.0
    rp = (sum(labels[i] == 0 for i in removed) / len(removed)) if budget > 0 else None
    mr = [i for i in range(n) if labels[i] == 2]
    rec = (sum(i in set(kept) for i in mr) / len(mr)) if mr else None
    return nd, rp, rec


def random_expected(labels, budget):
    """Exact expectation of the metrics under a uniformly random ranking."""
    n = len(labels); k = n - budget
    mean_g = sum(labels) / n
    ideal = dcg(sorted(labels, reverse=True)[:k])
    nd = (mean_g * sum(_disc(i) for i in range(k)) / ideal) if ideal > 0 else 1.0
    rp = (sum(l == 0 for l in labels) / n) if budget > 0 else None
    mr = sum(l == 2 for l in labels)
    rec = (k / n) if mr else None
    return nd, rp, rec


def _pct(vals, q):
    v = sorted(vals); pos = (len(v) - 1) * q / 100.0
    lo = int(math.floor(pos)); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def macro(pi, insts):
    res = []
    for j in range(3):
        v = [pi[i][j] for i in insts if pi[i][j] is not None]
        res.append(sum(v) / len(v) if v else float("nan"))
    return res


def bootstrap(pi, insts, B=5000, seed=0):
    rng = random.Random(seed); m = len(insts); boots = [[], [], []]
    for _ in range(B):
        samp = [insts[rng.randrange(m)] for _ in range(m)]
        r = macro(pi, samp)
        for j in range(3):
            if not math.isnan(r[j]): boots[j].append(r[j])
    return [(_pct(b, 2.5), _pct(b, 97.5)) for b in boots]
