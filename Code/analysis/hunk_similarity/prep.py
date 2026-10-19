import json,gzip,re,sys,collections,random
import argparse
from pathlib import Path
CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from data_pipeline import merged_generations as mg
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=Path, default=CODE.parent / 'Data/eel_dataset')
parser.add_argument('--output', type=Path, default=CODE / 'logs/hunk_similarity/pairs.json')
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
random.seed(0)
def changed(text,added_only=False):
    out=[]
    for l in text.split('\n')[1:]:
        if l[:1]=='+' or (l[:1]=='-' and not added_only):
            if l.startswith(('+++','---')) and len(l)>3 and l[3]==' ' and False: continue
            out.append(l[1:])
    return '\n'.join(out)
refs={};repo={}
for l in gzip.open(args.data_root / 'haiku_generations.jsonl.gz','rt'):
    g=json.loads(l);iid=g['instance_id'];repo[iid]=g['repo']
    refs[iid]=[dict(file=h['file'],start=h['old_start'],text=h['text']) for h in mg.diff_hunks(g['meta']['gold_patch'])]
texts=[];index={}
def tid(t):
    if t not in index: index[t]=len(texts);texts.append(t)
    return index[t]
pairs=[]   # dict(group, iid, kind, a_view1, b_view1, a_view2, b_view2, meta)
cnt=collections.Counter()
allrefs=[(iid,r) for iid,rs in refs.items() for r in rs]
byrepo=collections.defaultdict(list)
for iid,r in allrefs: byrepo[repo[iid]].append((iid,r))
for l in open(args.data_root / 'llm_t12_hunks.jsonl'):
    row=json.loads(l);iid=row['instance_id'];hs=row['llm_t12_hunks'];R=refs[iid]
    # one-to-one nearest matching, same file within 20 lines
    cost=np.full((len(hs),len(R)),1e6)
    for a,h in enumerate(hs):
        for b,r in enumerate(R):
            if h['filepath']==r['file'] and abs(h['old_start_line']-r['start'])<20: cost[a,b]=abs(h['old_start_line']-r['start'])
    m={};usedb=set()
    for c_,a,b in sorted((cost[a,b],a,b) for a in range(len(hs)) for b in range(len(R)) if cost[a,b]<1e6):
        if a in m or b in usedb: continue
        m[a]=b;usedb.add(b)
    for a,h in enumerate(hs):
        if a in m: r=R[m[a]];cnt['A one-to-one']+=1
        else:
            cands=[r for r in R if r['file']==h['filepath']]
            if not cands: cnt['A no same-file ref']+=1;continue
            r=min(cands,key=lambda r:abs(r['start']-h['old_start_line']));cnt['A nearest (not unique)']+=1
        # random control: another issue in the same repo
        oi,orr=random.choice(byrepo[repo[iid]])
        while oi==iid: oi,orr=random.choice(byrepo[repo[iid]])
        pairs.append(dict(group='A',iid=iid,tier=h['tier_label'],sample=min(h['sample'],1),
            v1=(tid(changed(h['hunk_diff'])),tid(changed(r['text']))),v2=(tid(changed(h['hunk_diff'],True)),tid(changed(r['text'],True))),
            c1=(tid(changed(h['hunk_diff'])),tid(changed(orr['text']))),c2=(tid(changed(h['hunk_diff'],True)),tid(changed(orr['text'],True)))))
late=[]
for l in open(args.data_root / 'tier3_hunks.jsonl'):
    row=json.loads(l);iid=row['instance_id'];R=refs[iid]
    for h in row['tier3_hunks']:
        cands=[r for r in R if r['file']==h['filepath']]
        if not cands: late.append((iid,h));continue
        r=min(cands,key=lambda r:abs(r['start']-h['old_start_line']));cnt['B']+=1
        oi,orr=random.choice(byrepo[repo[iid]])
        while oi==iid: oi,orr=random.choice(byrepo[repo[iid]])
        pairs.append(dict(group='B',iid=iid,tier=3,sample=min(h['sample'],1),
            v1=(tid(changed(h['hunk_diff'])),tid(changed(r['text']))),v2=(tid(changed(h['hunk_diff'],True)),tid(changed(r['text'],True))),
            c1=(tid(changed(h['hunk_diff'])),tid(changed(orr['text']))),c2=(tid(changed(h['hunk_diff'],True)),tid(changed(orr['text'],True)))))
def prefix(a,b):
    n=0
    for x,y in zip(a.split('/')[:-1],b.split('/')[:-1]):
        if x!=y: break
        n+=1
    return n
for iid,h in late:
    R=refs[iid]
    # no same-file reference hunk: use the reference hunk of the same issue with the closest file path (longest shared directory), ties -> nearest line
    r=max(R,key=lambda r:(prefix(h['filepath'],r['file']),-abs(r['start']-h['old_start_line'])));cnt['B fallback (closest path in same issue)']+=1
    oi,orr=random.choice(byrepo[repo[iid]])
    while oi==iid: oi,orr=random.choice(byrepo[repo[iid]])
    pairs.append(dict(group='B',iid=iid,tier=3,sample=min(h['sample'],1),
        v1=(tid(changed(h['hunk_diff'])),tid(changed(r['text']))),v2=(tid(changed(h['hunk_diff'],True)),tid(changed(r['text'],True))),
        c1=(tid(changed(h['hunk_diff'])),tid(changed(orr['text']))),c2=(tid(changed(h['hunk_diff'],True)),tid(changed(orr['text'],True)))))
print(dict(cnt),'pairs',len(pairs),'unique texts',len(texts))
json.dump(dict(texts=texts,pairs=pairs,counts=dict(cnt)),open(args.output,'w'))
