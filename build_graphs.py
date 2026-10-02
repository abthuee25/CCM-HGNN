#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_graphs.py
--------------
Read joern-exported workspace/dump/*.jsonl (one function per line: nodes +
AST/CFG/DFG edges) and build the Code Property Graph tensor file consumed by ccm.py.

Output workspace/graphs.pt:
{
  "graphs": [ {"x": FloatTensor(N,in_dim),
               "edge_index": LongTensor(2,E),
               "edge_type": LongTensor(E),   # 0=AST 1=CFG 2=DFG(REACHING_DEF)
               "y": 0/1, "id": str}, ... ],
  "splits": {"train":[idx...], "val":[...], "test":[...]},
  "meta":   {"in_dim", "num_relations", "type_vocab", "token_vocab", ...}
}

Node features = node-type one-hot + multi-hot bag of code tokens.
Vocabulary is computed from training graphs only to avoid label/test leakage.
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import torch

EDGE_MAP = {"AST": 0, "CFG": 1, "REACHING_DEF": 2}
TOKEN_RE = re.compile(
    r"0x[0-9a-fA-F]+|\d+\.?\d*|[A-Za-z_][A-Za-z0-9_]*"
    r"|==|!=|<=|>=|->|&&|\|\||\+\+|--|<<|>>"
    r"|[+\-*/%<>=!&|^~?:.,;(){}\[\]]"
)


def tokenize(code: str):
    return TOKEN_RE.findall(code)


def load_index(workspace):
    idx = {}
    with open(os.path.join(workspace, "index.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx[row["id"]] = int(row["label"])
    return idx


def load_raw_graphs(workspace, label_index):
    """Read all jsonl files, remap node ids to contiguous indices. Return list[dict]."""
    records = {}
    dup = 0
    files = sorted(glob.glob(os.path.join(workspace, "dump", "*.jsonl")))
    if not files:
        raise FileNotFoundError(
            f"{workspace}/dump/*.jsonl not found, run joern parsing and export first")
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                fid = d["f"]
                if fid not in label_index:
                    continue
                if fid in records:
                    dup += 1
                    continue
                nodes = d["n"]  # [[id,label,code],...]
                id_map = {n[0]: i for i, n in enumerate(nodes)}
                types = [n[1] for n in nodes]
                codes = [n[2] for n in nodes]
                edges = []
                for s, t, lab in d["e"]:
                    if s in id_map and t in id_map and lab in EDGE_MAP:
                        edges.append((id_map[s], id_map[t], EDGE_MAP[lab]))
                records[fid] = {
                    "id": fid,
                    "y": label_index[fid],
                    "types": types,
                    "codes": codes,
                    "edges": edges,
                }
    print(f"loaded graph records: {len(records)} (dropped duplicate keys {dup})")
    return list(records.values())


def graph_ok(g, min_nodes, max_nodes):
    n = len(g["types"])
    if n < min_nodes or n > max_nodes:
        return False
    labs = {e[2] for e in g["edges"]}
    # must contain AST edges and at least one of CFG/DFG
    return 0 in labs and (1 in labs or 2 in labs)


def balanced_sample(graphs, max_samples, seed):
    rng = np.random.RandomState(seed)
    pos = [g for g in graphs if g["y"] == 1]
    neg = [g for g in graphs if g["y"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    if max_samples is None or max_samples <= 0:
        return pos + neg
    per_class = max_samples // 2
    take = min(per_class, len(pos), len(neg))
    sel = pos[:take] + neg[:take]
    rng.shuffle(sel)
    print(f"balanced sample: {take} positive + {take} negative = {2 * take}")
    return sel


def stratified_split(graphs, seed, ratios=(0.8, 0.1, 0.1)):
    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    for y in (0, 1):
        idxs = [i for i, g in enumerate(graphs) if g["y"] == y]
        rng.shuffle(idxs)
        n = len(idxs)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        train += idxs[:n_tr]
        val += idxs[n_tr:n_tr + n_va]
        test += idxs[n_tr + n_va:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def build_vocab(graphs, split_idx, type_k=31, token_k=1500, min_tok_freq=3):
    """Count only on training graphs: node-type vocabulary, token vocabulary,
    and node-level document frequency (DF).

    Returns type_vocab, token_vocab, idf (np.ndarray, column-aligned with
    token_vocab), and the total number of training nodes used for IDF.
    """
    type_cnt, tok_df = Counter(), Counter()
    n_nodes = 0
    for i in split_idx:
        g = graphs[i]
        type_cnt.update(g["types"])
        n_nodes += len(g["types"])
        for code in g["codes"]:
            tok_df.update(set(tokenize(code)))   # node-level document frequency
    type_vocab = {"<UNK>": 0}
    for t, _ in type_cnt.most_common(type_k - 1):
        type_vocab[t] = len(type_vocab)
    tok_items = [(t, df) for t, df in tok_df.most_common(token_k)
                 if df >= min_tok_freq]
    token_vocab = {t: i for i, (t, _) in enumerate(tok_items)}
    # smoothed IDF: tokens with very high DF (punctuation/keywords) are auto down-weighted
    idf = np.array(
        [np.log((n_nodes + 1.0) / (dict(tok_items)[t] + 1.0)) + 1.0
         for t in token_vocab],
        dtype=np.float32)
    print(f"type vocab {len(type_vocab)} kinds, token vocab {len(token_vocab)} tokens, "
          f"training nodes {n_nodes}")
    return type_vocab, token_vocab, idf


def build_codebert_embeddings(graphs, model_name, max_length=32,
                              batch_size=256, device="cpu", num_threads=8):
    """Offline-encode all unique node codes across sampled graphs with CodeBERT.

    Node codes are highly repetitive (identifiers/literals/operators); we dedupe
    by string, run batched forward passes, mean-pool over the attention mask to
    get 768-d vectors. Batches are length-sorted into buckets to minimize padding.
    Returns {code: np.float16[768]}.
    """
    import time
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")      # avoid loading TF
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # MKL/libomp coexistence
    from transformers import AutoTokenizer, AutoModel

    torch.set_num_threads(num_threads)
    uniq = list({c for g in graphs for c in g["codes"]})
    # length bucketing: sequences in the same batch have similar lengths, minimal padding
    order = sorted(range(len(uniq)), key=lambda i: (len(uniq[i]), i))
    print(f"CodeBERT: {model_name} | unique node codes {len(uniq)} | "
          f"max_len={max_length} batch={batch_size} threads={num_threads} "
          f"device={device}")
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.to(device).eval()
    hid = model.config.hidden_size

    table = np.empty((len(uniq), hid), dtype=np.float16)
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for bs in range(0, len(order), batch_size):
            pos = order[bs:bs + batch_size]
            batch = [uniq[p] for p in pos]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc).last_hidden_state           # (B,L,H)
            mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
            emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            table[pos] = emb.float().cpu().numpy().astype(np.float16)
            done += len(pos)
            if (bs // batch_size) % 50 == 0:
                speed = done / max(time.time() - t0, 1e-9)
                eta = (len(order) - done) / max(speed, 1e-9)
                print(f"  {done}/{len(uniq)} "
                      f"({speed:.0f} codes/s, ETA {eta/60:.1f} min)", flush=True)
    print(f"CodeBERT encoding done in {(time.time()-t0)/60:.1f} min")
    return dict(zip(uniq, table)), hid


def encode_graph(g, type_vocab, token_vocab, idf, reverse_edges="ast",
                 feature_mode="tfidf", code_emb=None):
    n = len(g["types"])
    nt, nv = len(type_vocab), len(token_vocab)

    # ---- feature block 1: type one-hot; feature block 2 (optional): token sqrt(TF)*IDF ----
    if feature_mode == "codebert":
        type_x = torch.zeros(n, nt, dtype=torch.float32)
        type_x[torch.arange(n),
               [type_vocab.get(l, 0) for l in g["types"]]] = 1.0
        x = None
    else:
        rows, cols, vals = [], [], []
        for i, (lab, code) in enumerate(zip(g["types"], g["codes"])):
            rows.append(i)
            cols.append(type_vocab.get(lab, 0))
            vals.append(1.0)
            if feature_mode in ("tfidf", "both"):
                for tok_, cnt in Counter(tokenize(code)).items():
                    j = token_vocab.get(tok_)
                    if j is not None:
                        rows.append(i)
                        cols.append(nt + j)
                        vals.append(float(np.sqrt(cnt)) * float(idf[j]))
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        x = torch.sparse_coo_tensor(
            indices, values, (n, nt + nv)).coalesce()
        if feature_mode == "tfidf":
            pass  # sparse output, keep memory-friendly
        else:  # both: densify then concatenate with BERT
            x = x.to_dense()

    if feature_mode != "tfidf":
        # CodeBERT dense block (stored as fp16, .float() during training)
        blocks = [type_x if feature_mode == "codebert" else x]
        hid = next(iter(code_emb.values())).shape[0]
        zero = np.zeros(hid, dtype=np.float16)
        bert = np.stack([code_emb.get(c, zero) for c in g["codes"]])
        blocks.append(torch.from_numpy(bert))
        x = torch.cat(blocks, dim=1).half()

    # ---- edges: optional reverse edges (default: AST only, helps subtree info flow up) ----
    edge_list = list(g["edges"])
    extra = []
    for s, t, lab in edge_list:
        if reverse_edges == "all" or (reverse_edges == "ast" and lab == 0):
            extra.append((t, s, lab))
    edge_list += extra
    if edge_list:
        arr = np.asarray(edge_list, dtype=np.int64)
        edge_index = torch.from_numpy(arr[:, :2].T.copy())     # (2,E)
        edge_type = torch.from_numpy(arr[:, 2].copy())        # (E,)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "y": int(g["y"]),
        "id": g["id"],
    }


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--workspace", default=os.path.join(base, "workspace"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-samples", type=int, default=4000,
                    help="total balanced sample size; <=0 means use all")
    ap.add_argument("--min-nodes", type=int, default=8)
    ap.add_argument("--max-nodes", type=int, default=400)
    ap.add_argument("--type-k", type=int, default=31)
    ap.add_argument("--token-k", type=int, default=1500)
    ap.add_argument("--min-tok-freq", type=int, default=3)
    ap.add_argument("--reverse-edges", default="ast",
                    choices=["none", "ast", "all"],
                    help="reverse edges: none=off; ast=AST only (default); all=all relations")
    ap.add_argument("--feature-mode", default="tfidf",
                    choices=["tfidf", "codebert", "both"],
                    help="tfidf=type one-hot+TF-IDF (default); "
                         "codebert=type one-hot+CodeBERT; both=concatenate all three")
    ap.add_argument("--codebert-model", default="microsoft/codebert-base")
    ap.add_argument("--codebert-batch-size", type=int, default=256)
    ap.add_argument("--codebert-max-len", type=int, default=32)
    ap.add_argument("--codebert-threads", type=int, default=8)
    ap.add_argument("--codebert-device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = args.out or os.path.join(args.workspace, "graphs.pt")
    label_index = load_index(args.workspace)
    graphs = load_raw_graphs(args.workspace, label_index)

    before = len(graphs)
    graphs = [g for g in graphs
              if graph_ok(g, args.min_nodes, args.max_nodes)]
    print(f"after node-count/edge-type filter: {len(graphs)} "
          f"(removed {before - len(graphs)})")
    npos = sum(g["y"] for g in graphs)
    print(f"available graph label distribution: positive {npos} / "
          f"negative {len(graphs) - npos}")

    # print edge-type statistics
    ec = Counter()
    nc = []
    for g in graphs:
        ec.update(e[2] for e in g["edges"])
        nc.append(len(g["types"]))
    nc = np.asarray(nc)
    print(f"edge count: AST={ec[0]} CFG={ec[1]} DFG={ec[2]}")
    print(f"node count: mean {nc.mean():.1f} median {np.median(nc):.0f} "
          f"p90={np.percentile(nc, 90):.0f}")

    graphs = balanced_sample(graphs, args.max_samples, args.seed)
    splits = stratified_split(graphs, args.seed)

    code_emb, bert_dim = None, 0
    token_k = args.token_k if args.feature_mode != "codebert" else 0
    type_vocab, token_vocab, idf = build_vocab(
        graphs, splits["train"], args.type_k, token_k, args.min_tok_freq)

    if args.feature_mode in ("codebert", "both"):
        code_emb, bert_dim = build_codebert_embeddings(
            graphs, args.codebert_model,
            max_length=args.codebert_max_len,
            batch_size=args.codebert_batch_size,
            device=args.codebert_device,
            num_threads=args.codebert_threads)

    encoded = [encode_graph(g, type_vocab, token_vocab, idf,
                            args.reverse_edges, args.feature_mode,
                            code_emb) for g in graphs]
    in_dim = len(type_vocab) + len(token_vocab) + bert_dim
    meta = {
        "in_dim": in_dim,
        "num_relations": 3,
        "relation_names": ["AST", "CFG", "DFG"],
        "type_vocab": type_vocab,
        "token_vocab": token_vocab,
        "idf": torch.from_numpy(idf),
        "reverse_edges": args.reverse_edges,
        "feature_mode": args.feature_mode,
        "codebert_model": args.codebert_model if bert_dim else None,
        "bert_dim": bert_dim,
        "feature": {"tfidf": "type-onehot + sqrt(TF)*IDF sparse",
                    "codebert": "type-onehot + CodeBERT mean-pool dense fp16",
                    "both": "type-onehot + TF-IDF + CodeBERT dense fp16"}[
            args.feature_mode],
        "num_graphs": len(encoded),
        "seed": args.seed,
    }
    torch.save({"graphs": encoded, "splits": splits, "meta": meta}, out)
    print(f"saved {len(encoded)} graphs -> {out} (feature: {args.feature_mode}, "
          f"in_dim={in_dim}, reverse_edges: {args.reverse_edges})")
    print(f"splits: train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])}")


if __name__ == "__main__":
    main()
