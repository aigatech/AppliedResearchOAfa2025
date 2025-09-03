import argparse, json, os, re
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np, pandas as pd, tensorflow as tf
from datasets import load_dataset

WIN_COLS=[f"winner.card{i}.id" for i in range(1,9)]
LOS_COLS=[f"loser.card{i}.id"  for i in range(1,9)]

#maps card IDs to names from csv file
def load_card_map(path:str)->Tuple[Dict[int,str],Dict[str,int]]:
    if not path: return {}, {}
    df = pd.read_csv(path)
    id_col, name_col = None, None
    for c in df.columns:
        cl = c.lower()
        if cl.endswith(".id") and id_col is None: id_col = c
        if cl.endswith(".name") and name_col is None: name_col = c
    if id_col is None or name_col is None: return {}, {}
    id2name, name2id = {}, {}
    for _, r in df.iterrows():
        try:
            cid = int(r[id_col]); nm = str(r[name_col]).strip()
            id2name[cid] = nm; name2id[nm.lower()] = cid
        except: pass
    return id2name, name2id

#normalizes card IDs to integers
def norm_id(x):
    if x is None: return None
    s = str(x).strip()
    try: return int(float(s))
    except: return None

#creates structured signature for decks
def deck_sig(ids:List[int])->str:
    ids = [i for i in ids if i is not None]
    if len(ids) < 8: ids += [0]*(8-len(ids))
    return "-".join(str(i) for i in sorted(ids)[:8])

def ids_to_indices(ids:List[int], id2idx:Dict[int,int])->List[int]:
    out = [id2idx.get(cid,0) if cid is not None else 0 for cid in ids[:8]]
    if len(out) < 8: out += [0]*(8-len(out))
    return out

def parse_deck_text(text:str, name2id:Dict[str,int])->List[int]:
    toks = re.split(r"[\s,]+", text.strip()); ids=[]
    for t in toks:
        if not t: continue
        if t.isdigit(): ids.append(int(t))
        else:
            cid = name2id.get(t.lower())
            if cid is None: raise ValueError(f"Unknown card: {t}")
            ids.append(cid)
    if len(ids) != 8: raise ValueError("Provide exactly 8 cards (IDs or names).")
    return ids

#computes similarity between two decks via interesction/union
def similarity(a:str, b:str)->float:
    A,B = set(a.split('-')), set(b.split('-'))
    return len(A & B) / max(1, len(A | B))

#maps card ID to index and builds candidate decks for use
def build_vocab_and_candidates(ds, limit_rows, max_cards, top_candidates):
    cc, dc = Counter(), Counter(); n=0
    for r in ds:
        if any(c not in r for c in WIN_COLS+LOS_COLS): continue
        w = [norm_id(r.get(c)) for c in WIN_COLS]
        l = [norm_id(r.get(c)) for c in LOS_COLS]
        for cid in w+l:
            if cid is not None: cc[cid] += 1
        dc[deck_sig(w)] += 1; dc[deck_sig(l)] += 1
        n += 1
        if limit_rows and n >= limit_rows: break
    top_ids = [cid for cid,_ in cc.most_common(max_cards)]
    id2idx = {cid:i+1 for i, cid in enumerate(top_ids)} 
    cand = pd.DataFrame(dc.most_common(top_candidates), columns=["deck_sig","freq"])
    return id2idx, cand

#separates data into mini-batches for training
def batch_stream(ds, id2idx, batch_size, limit_rows, max_steps):
    Xw,Xl,Y,steps,n = [],[],[],0,0
    for r in ds:
        if any(c not in r for c in WIN_COLS+LOS_COLS): continue
        w = ids_to_indices(sorted([norm_id(r.get(c)) for c in WIN_COLS]), id2idx)
        l = ids_to_indices(sorted([norm_id(r.get(c)) for c in LOS_COLS]), id2idx)
        Xw += [w, l]; Xl += [l, w]; Y += [1.0, 0.0]
        if len(Y) >= batch_size:
            yield np.array(Xw), np.array(Xl), np.array(Y, dtype=np.float32)
            Xw,Xl,Y = [],[],[]; steps += 1
            if max_steps and steps >= max_steps: break
        n += 1
        if limit_rows and n >= limit_rows: break
    if Y: yield np.array(Xw), np.array(Xl), np.array(Y, dtype=np.float32)

#predicts win probability between two decks
def build_model(vocab, d=16):
    W = tf.keras.Input(shape=(8,), dtype="int32")
    L = tf.keras.Input(shape=(8,), dtype="int32")
    Emb = tf.keras.layers.Embedding(vocab, d, mask_zero=True)
    def pool(x):
        em = Emb(x)
        return tf.keras.layers.GlobalAveragePooling1D()(em)
    w = pool(W); l = pool(L)
    x = tf.keras.layers.Concatenate()([w - l, w * l])
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    logit = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model([W, L], logit)
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    return model

def pretty(sig, id2name):
    return ", ".join(id2name.get(int(x), x) for x in sig.split("-"))

#cleans up data and saves weights from training
def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    id2name, name2id = load_card_map(args.card_map)
    ds1 = load_dataset(args.hf_repo, split="train", streaming=True) if args.hf_repo \
        else load_dataset("csv", data_files={"train":args.data_files}, split="train", streaming=True)
    id2idx, cands = build_vocab_and_candidates(ds1, args.limit_rows, args.max_cards, args.top_candidates)
    with open(os.path.join(args.out_dir,"id2idx.json"),"w") as f:
        json.dump({str(k):v for k,v in id2idx.items()}, f)
    cands.to_csv(os.path.join(args.out_dir,"candidates.csv"), index=False)
    ds2 = load_dataset(args.hf_repo, split="train", streaming=True) if args.hf_repo \
        else load_dataset("csv", data_files={"train":args.data_files}, split="train", streaming=True)
    vocab = (max(id2idx.values()) if id2idx else 0) + 1
    model = build_model(vocab, args.d_model)
    for Xw,Xl,Y in batch_stream(ds2, id2idx, args.batch_size, args.limit_rows, args.max_steps):
        model.train_on_batch([Xw, Xl], Y)
    weights_path = os.path.join(args.out_dir, "model.weights.h5")
    model.save_weights(weights_path)
    print("Saved:", weights_path)

#predicts counter decks against a given deck
def query(args):
    id2name, name2id = load_card_map(args.card_map)
    with open(args.vocab,"r") as f:
        id2idx = {int(k):int(v) for k,v in json.load(f).items()}
    vocab = (max(id2idx.values()) if id2idx else 0) + 1
    model = build_model(vocab, args.d_model); model.load_weights(args.model)
    ids = parse_deck_text(args.deck, name2id)
    target_sig = deck_sig(ids)
    T = np.array([ids_to_indices(sorted(ids), id2idx)], dtype=np.int32)
    cand = pd.read_csv(args.candidates); sigs = cand["deck_sig"].tolist(); freqs = cand["freq"].tolist()
    scores, B = [], 1024
    for i in range(0, len(sigs), B):
        chunk = sigs[i:i+B]
        W = np.array([ids_to_indices([int(x) for x in s.split("-")], id2idx) for s in chunk], dtype=np.int32)
        L = np.repeat(T, len(chunk), axis=0)
        p = tf.sigmoid(model.predict([W, L], verbose=0)).numpy().reshape(-1)
        scores += p.tolist()
    ranked = sorted(zip(sigs, scores, freqs), key=lambda x: (x[1], x[2]), reverse=True)
    picks = []
    for sig, p, _ in ranked:
        if all(similarity(sig, s) < 0.5 for s,_ in picks):
            picks.append((sig, p))
        if len(picks) >= args.top: break
    print("Target:", pretty(target_sig, id2name))
    for i, (sig, p) in enumerate(picks, 1):
        print(f"#{i}: P(counter win)={p:.3f} — {pretty(sig, id2name)}")

#Interpreter for commands
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","query"], required=True)
    ap.add_argument("--hf_repo", type=str, default=None)
    ap.add_argument("--data_files", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="out")
    ap.add_argument("--limit_rows", type=int, default=200000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--max_cards", type=int, default=256)
    ap.add_argument("--top_candidates", type=int, default=5000)
    ap.add_argument("--card_map", type=str, default="")
    # query args
    ap.add_argument("--model", type=str)
    ap.add_argument("--vocab", type=str)
    ap.add_argument("--candidates", type=str)
    ap.add_argument("--deck", type=str, default="")
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    if a.mode == "train":
        if not (a.hf_repo or a.data_files): raise SystemExit("Provide --hf_repo or --data_files")
        train(a)
    else:
        if not (a.model and a.vocab and a.candidates and a.deck): raise SystemExit("Provide --model --vocab --candidates --deck")
        query(a)

if __name__ == "__main__":
    main()
