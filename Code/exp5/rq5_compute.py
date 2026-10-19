"""Survey-based ranking metrics from labels and per-hunk scores.

usage: python exp5/rq5_compute.py --survey ../Data/Evaluation/Discussion [--scores exp5/scores_*.json ...]
Each score file: {"meta": {"name": ...}, "scores": {"<instance_id>|<hunk_id>": {"score": float}}}.
"""
import argparse, csv, json, random, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq5_metrics import issue_metrics, random_expected, macro, bootstrap

CODE = {"retain": 2, "not certain": 1, "reject": 0}

ap = argparse.ArgumentParser()
ap.add_argument("--survey", required=True)
ap.add_argument("--scores", nargs="+", default=sorted(str(p) for p in Path(__file__).resolve().parent.glob("scores_*.json")))
ap.add_argument("--draws", type=int, default=2000, help="random tie-break draws per issue")
args = ap.parse_args()
S = Path(args.survey)
for score_path in args.scores:
    pending = json.load(open(score_path)).get("meta", {}).get("pending_hunks", [])
    if pending:
        ap.error(f"{len(pending)} replaced hunks await reviewer confirmation and rescoring")

t2i = {r["task_id"]: r["swebench_instance_id"] for r in csv.DictReader(open(S / "task_id_mapping.csv"))}
votes = {}
for r in csv.DictReader(open(S / "labels.csv")):
    votes.setdefault(f"{t2i[r['task_id']]}|{r['hunk_id']}", []).append(CODE[r["label"]])
HS = {k: int(statistics.median(v)) for k, v in votes.items()}
BI = {}
for line in open(S / "survey_hunks.jsonl"):
    r = json.loads(line); BI[r["instance_id"]] = [h["id"] for h in r["hunks"]]
K = lambda i, h: f"{i}|{h}"
rb = {i: sum(HS[K(i, h)] == 0 for h in hh) for i, hh in BI.items()}
pos = [i for i in BI if 0 < rb[i] < len(BI[i])]


def pairwise(f):
    w = n = 0
    for i, hh in BI.items():
        for a in hh:
            for b in hh:
                if HS[K(i, a)] > HS[K(i, b)]:
                    n += 1; d = f(K(i, a)) - f(K(i, b)); w += 1 if d > 0 else .5 if d == 0 else 0
    return w / n, n


rows = []
pi = {i: random_expected([HS[K(i, h)] for h in BI[i]], rb[i]) for i in BI}
m, ci = macro(pi, pos), bootstrap(pi, pos)
rows.append(("Random", m, ci, 0.5))
for path in args.scores:
    d = json.load(open(path)); f = lambda k, s=d["scores"]: s[k]["score"]
    rng = random.Random(0); out = {}
    for i, hh in BI.items():
        L = [HS[K(i, h)] for h in hh]; sc = [f(K(i, h)) for h in hh]; acc = [[], [], []]
        for _ in range(args.draws):
            r = issue_metrics(L, [s + rng.random() * 1e-6 for s in sc], rb[i])
            for j in range(3):
                if r[j] is not None: acc[j].append(r[j])
        out[i] = tuple(sum(a) / len(a) if a else None for a in acc)
    rows.append((d["meta"]["name"], macro(out, pos), bootstrap(out, pos), pairwise(f)[0]))

print(f"{len(pos)} issues with at least one reject hunk, {sum(rb.values())} reject hunks, {pairwise(lambda k: 0)[1]} ordered pairs\n")
print("| Method | Remove Precision [95% CI] | Retain Recall [95% CI] | Pairwise agreement |")
print("|---|---|---|---|")
for name, m, ci, pw in rows:
    print(f"| {name} | {m[1]:.3f} [{ci[1][0]:.2f}, {ci[1][1]:.2f}] | {m[2]:.3f} [{ci[2][0]:.2f}, {ci[2][1]:.2f}] | {pw:.3f} |")
