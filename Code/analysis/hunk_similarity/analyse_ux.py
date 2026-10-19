"""Table: UniXcoder cosine similarity of changed lines (no unchanged context).
A = retained (T1/T2) hunk vs its matched reference hunk; B = T3 hunk vs closest reference hunk of the same issue;
control = retained hunk vs a random reference hunk of another issue in the same repo.  Inputs: pairs.json, emb_ux.npy."""
import argparse,json,numpy as np
from pathlib import Path
CODE = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--pairs', type=Path, default=CODE.parent / 'Data/Evaluation/hunk_similarity/pairs.json')
parser.add_argument('--embeddings', type=Path, default=CODE.parent / 'Data/Evaluation/hunk_similarity/emb_ux.npy')
parser.add_argument('--output', type=Path, default=CODE / 'logs/hunk_similarity/result_unixcoder.json')
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
d=json.load(open(args.pairs));P=d['pairs'];E=np.load(args.embeddings)
g=np.array([p['group'] for p in P]);tier=np.array([p['tier'] for p in P]);smp=np.array([p['sample'] for p in P])
def cos(A,B): return (A*B).sum(1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+1e-9)
def sims(view): return cos(E[[p[view][0] for p in P]],E[[p[view][1] for p in P]])
def desc(x): return dict(n=int(len(x)),mean=round(float(x.mean()),4),median=round(float(np.median(x)),4),p25=round(float(np.percentile(x,25)),4),p75=round(float(np.percentile(x,75)),4))
s=sims('v1');r=sims('c1');s2=sims('v2');r2=sims('c2')
A=s[g=='A'];B=s[g=='B']
res=dict(
 changed_lines=dict(A=desc(A),B=desc(B),random_control=desc(r[g=='A']),diff_A_minus_B=dict(mean=round(float(A.mean()-B.mean()),4),median=round(float(np.median(A)-np.median(B)),4)),
   A_T1=desc(s[(g=='A')&(tier==1)]),A_T2=desc(s[(g=='A')&(tier==2)]),A_sample0=desc(s[(g=='A')&(smp==0)]),A_sample1=desc(s[(g=='A')&(smp==1)])),
 added_lines_only=dict(A=desc(s2[g=='A']),B=desc(s2[g=='B']),random_control=desc(r2[g=='A'])),
 counts=d['counts'])
json.dump(res,open(args.output,'w'),indent=1)
c=res['changed_lines']
print('Pair\tn\tMean\tMedian\tIQR')
print(f"A: Retained LLM hunk vs. matched reference hunk\t{c['A']['n']}\t{c['A']['mean']:.3f}\t{c['A']['median']:.3f}\t{c['A']['p25']:.3f} to {c['A']['p75']:.3f}")
print(f"B: T3 hunk vs. same-file or nearest-path reference hunk\t{c['B']['n']}\t{c['B']['mean']:.3f}\t{c['B']['median']:.3f}\t{c['B']['p25']:.3f} to {c['B']['p75']:.3f}")
print(f"Difference (A-B)\t\t{c['diff_A_minus_B']['mean']:+.3f}\t{c['diff_A_minus_B']['median']:+.3f}")
print(f"Control: Random unrelated reference hunk\t{c['random_control']['n']}\t{c['random_control']['mean']:.3f}\t{c['random_control']['median']:.3f}")
