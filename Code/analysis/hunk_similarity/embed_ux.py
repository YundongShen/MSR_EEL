import argparse,json,torch,numpy as np,time
from pathlib import Path
from transformers import AutoTokenizer,AutoModel
CODE = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--pairs', type=Path, default=CODE.parent / 'Data/Evaluation/hunk_similarity/pairs.json')
parser.add_argument('--output', type=Path, default=CODE / 'logs/hunk_similarity/emb_ux.npy')
parser.add_argument('--model', default='microsoft/unixcoder-base')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
d=json.load(open(args.pairs));texts=d['texts']
tok=AutoTokenizer.from_pretrained(args.model);model=AutoModel.from_pretrained(args.model).to(args.device).eval()
MAX=512
def encode(t):
    ids=tok.convert_tokens_to_ids(tok.tokenize(t if t.strip() else ' ')[:MAX-4])
    ids=[tok.cls_token_id,tok.convert_tokens_to_ids('<encoder-only>'),tok.sep_token_id]+ids+[tok.sep_token_id]
    return ids
order=sorted(range(len(texts)),key=lambda i:len(texts[i]))
out=np.zeros((len(texts),768),dtype=np.float32);t0=time.time()
with torch.no_grad():
    for b in range(0,len(order),64):
        idx=order[b:b+64];ids=[encode(texts[i]) for i in idx];L=max(len(x) for x in ids)
        inp=torch.tensor([x+[tok.pad_token_id]*(L-len(x)) for x in ids]).to(args.device)
        mask=(inp!=tok.pad_token_id).float().unsqueeze(-1)
        h=model(inp,attention_mask=(inp!=tok.pad_token_id)).last_hidden_state
        out[idx]=((h*mask).sum(1)/mask.sum(1)).cpu().numpy()
        if (b//64)%50==0: print(b,len(order),round(time.time()-t0),flush=True)
np.save(args.output,out);print('done',flush=True)
