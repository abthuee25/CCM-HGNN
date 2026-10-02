import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, accuracy_score,
                             roc_auc_score)


# ============================================================
# 1. Poincaré Ball hyperbolic geometry operations
# ============================================================
class PoincareBall:
    def __init__(self, c=1.0, eps=1e-7):
        self.c = c
        self.eps = eps

    def _lambda(self, x):
        """Conformal factor lambda_x^c = 2 / (1 - c ||x||^2)."""
        c = self.c
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        return 2.0 / (1.0 - c * x2).clamp(min=self.eps)

    def mobius_add(self, x, y):
        """Möbius addition x ⊕_c y."""
        c = self.c
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
        den = 1.0 + 2.0 * c * xy + c * c * x2 * y2
        return num / den.clamp(min=self.eps)

    def expmap0(self, x):
        """Exponential map at the origin exp_0^c(x)."""
        c = self.c
        if c == 0.0:
            return x
        sqrt_c = math.sqrt(c)
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return torch.tanh(sqrt_c * norm) * x / (sqrt_c * norm)

    def logmap0(self, x):
        """Logarithmic map at the origin log_0^c(x)."""
        c = self.c
        if c == 0.0:
            return x
        sqrt_c = math.sqrt(c)
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(
            min=self.eps, max=1.0 / sqrt_c - self.eps
        )
        return torch.atanh(sqrt_c * norm) * x / (sqrt_c * norm)

    def expmap(self, x, v):
        """Exponential map at point x, exp_x^c(v)."""
        c = self.c
        if c == 0.0:
            return x + v
        sqrt_c = math.sqrt(c)
        lam = self._lambda(x)
        norm_v = torch.norm(v, dim=-1, keepdim=True).clamp(min=self.eps)
        second = torch.tanh(sqrt_c * lam * norm_v / 2.0) * v / (sqrt_c * norm_v)
        return self.mobius_add(x, second)

    def logmap(self, x, y):
        """Logarithmic map at point x, log_x^c(y)."""
        c = self.c
        if c == 0.0:
            return y - x
        sqrt_c = math.sqrt(c)
        lam = self._lambda(x)
        diff = self.mobius_add(-x, y)
        norm_diff = torch.norm(diff, dim=-1, keepdim=True).clamp(
            min=self.eps, max=1.0 / sqrt_c - self.eps
        )
        return (2.0 / (sqrt_c * lam)) * torch.atanh(sqrt_c * norm_diff) * diff / norm_diff

    def dist(self, x, y):
        """Geodesic distance d_c(x, y)."""
        c = self.c
        if c == 0.0:
            return torch.norm(x - y, dim=-1)
        sqrt_c = math.sqrt(c)
        diff = self.mobius_add(-x, y)
        norm_diff = torch.norm(diff, dim=-1, keepdim=True).clamp(
            min=self.eps, max=1.0 / sqrt_c - self.eps
        )
        return 2.0 / sqrt_c * torch.atanh(sqrt_c * norm_diff)


# ============================================================
# 2. Heterogeneous hyperbolic GNN layer with topology-aware modulation
# ============================================================
class CCMHGNNLayer(nn.Module):
    def __init__(self, dim, num_relations, c=1.0, dropout=0.1,
                 use_modulation=True, mod_scale=None):
        super().__init__()
        self.dim = dim
        self.num_relations = num_relations
        self.c = c
        self.ball = PoincareBall(c)
        # ablation switch: S≡1 when use_modulation=False; mod_scale controls c_i = S·mod_scale
        self.use_modulation = use_modulation
        self.mod_scale = c if mod_scale is None else mod_scale

        # relation-specific linear transforms W_r (small gain init to suppress tangent norm growth)
        self.rel_linear = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_relations)
        ])
        for lin in self.rel_linear:
            nn.init.xavier_uniform_(lin.weight, gain=0.5)
            nn.init.zeros_(lin.bias)
        # relation-specific attention params a_r (small scale init for stable early softmax)
        self.rel_attn = nn.ParameterList([
            nn.Parameter(torch.randn(2 * dim) * 0.1)
            for _ in range(num_relations)
        ])

        # node curvature learning module: predicts modulation factor S_i
        self.mod_net = nn.Sequential(
            nn.Linear(dim + 1, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # gated residual fusion network
        self.gate_net = nn.Sequential(
            nn.Linear(2 * dim + 1, dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        """
        x: (N, dim) current hyperbolic node representations (on the Poincaré ball)
        edge_index: (2, E) edge indices, [0] = source, [1] = target
        edge_type: (E,) relation type in 0..num_relations-1
        """
        N = x.size(0)
        device = x.device

        # tangent-space representation (at the origin)
        x_tan = self.ball.logmap0(x)  # (N, dim)

        # ---------- neighborhood feature heterogeneity rho_i ----------
        src, dst = edge_index[0], edge_index[1]
        diff = x_tan[src] - x_tan[dst]          # (E, dim)
        dist = torch.norm(diff, dim=-1)         # (E,)
        sum_dist = torch.zeros(N, device=device)
        count = torch.zeros(N, device=device)
        sum_dist.index_add_(0, dst, dist)
        count.index_add_(0, dst, torch.ones_like(dist))
        rho = (sum_dist / count.clamp(min=1.0)).unsqueeze(-1)  # (N, 1)

        # ---------- node topology-aware modulation factor ----------
        if self.use_modulation:
            mod_input = torch.cat([x_tan, rho], dim=-1)   # (N, dim+1)
            S = self.mod_net(mod_input)                   # (N, 1) ∈ (0,1)
        else:                                             # ablation: S ≡ 1
            S = torch.ones(N, 1, device=device)
        c_i = S * self.mod_scale                          # (N, 1)

        # ---------- relation-aware message computation ----------
        messages = []          # transformed neighbor tangent representation per edge
        attn_logits = []       # unnormalized attention coefficient per edge
        all_dst = []           # target node index per edge

        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.sum() == 0:
                continue
            src_r = src[mask]
            dst_r = dst[mask]

            # origin tangent representation of neighbor j
            xj_tan = x_tan[src_r]                     # (E_r, dim)
            # relation-specific linear transform
            trans = self.rel_linear[r](xj_tan)        # (E_r, dim)
            # map back to manifold then take log, yielding transformed tangent representation
            xj_hyp = self.ball.expmap0(trans)
            xj_tan_trans = self.ball.logmap0(xj_hyp)  # (E_r, dim)

            # origin tangent representation of target node i
            xi_tan = x_tan[dst_r]                     # (E_r, dim)

            # attention: concat [log_0(x_i) || log_0(\tilde{x}_j^r)], scaled by modulation factor c_i
            attn_input = torch.cat([xi_tan, xj_tan_trans], dim=-1)  # (E_r, 2*dim)
            e = F.leaky_relu(attn_input @ self.rel_attn[r])         # (E_r,)
            e = e * c_i[dst_r].squeeze(-1)                           # multiply by c_i

            messages.append(xj_tan_trans)
            attn_logits.append(e)
            all_dst.append(dst_r)

        if len(messages) == 0:
            return x  # return unchanged when there are no edges

        all_messages = torch.cat(messages, dim=0)      # (E, dim)
        all_logits = torch.cat(attn_logits, dim=0)     # (E,)
        all_dst = torch.cat(all_dst, dim=0)            # (E,)

        # ---------- Softmax normalization ----------
        exp_logits = torch.exp(all_logits - all_logits.max())
        sum_exp = torch.zeros(N, device=device)
        sum_exp.index_add_(0, all_dst, exp_logits)
        alpha = exp_logits / sum_exp[all_dst].clamp(min=1e-10)   # (E,)

        # ---------- tangent-space message aggregation ----------
        weighted_messages = all_messages * alpha.unsqueeze(-1)   # (E, dim)
        m = torch.zeros(N, self.dim, device=device)
        m.index_add_(0, all_dst, weighted_messages)              # (N, dim)

        # ---------- gated residual fusion ----------
        gate_input = torch.cat([x_tan, m, c_i], dim=-1)          # (N, 2*dim+1)
        g = self.gate_net(gate_input)                            # (N, dim)
        z = (1.0 - g) * x_tan + g * m                            # (N, dim)

        # ---------- map back to manifold ----------
        x_new = self.ball.expmap0(z)
        x_new = self.dropout(x_new)
        return x_new


# ============================================================
# 3. Karcher flow graph readout (Fréchet mean)
# ============================================================
def karcher_flow(ball, x, T=10):
    """
    Compute the Fréchet mean of node representations on the Poincaré ball via Karcher iteration.
    x: (N, dim) node representations on the manifold
    returns: (dim,) graph-level representation
    """
    N, dim = x.size()
    device = x.device
    mu = torch.zeros(1, dim, device=device)  # initialize at the origin
    for _ in range(T):
        # logarithmic map of all nodes to the current mean mu
        v = ball.logmap(mu.expand(N, -1), x)   # (N, dim)
        v_avg = v.mean(dim=0, keepdim=True)    # (1, dim)
        mu = ball.expmap(mu, v_avg)            # (1, dim)
    return mu.squeeze(0)


# ============================================================
# 4. CCM-HGNN full model
# ============================================================
class CCMHGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations,
                 num_layers=2, c=1.0, dropout=0.1,
                 use_modulation=True, mod_scale=None):
        super().__init__()
        self.c = c
        self.ball = PoincareBall(c)
        self.input_linear = nn.Linear(in_dim, hidden_dim)
        nn.init.xavier_uniform_(self.input_linear.weight, gain=1.0)
        nn.init.zeros_(self.input_linear.bias)
        # shrink initial tangent vectors into the Poincaré ball to avoid exp_0 saturation at the boundary (gradient vanishing)
        self.input_scale = 0.3
        self.layers = nn.ModuleList([
            CCMHGNNLayer(hidden_dim, num_relations, c, dropout,
                         use_modulation, mod_scale)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_type):
        # x: (N, in_dim) Euclidean node features (can be sparse COO tensor)
        if x.is_sparse:
            x = x.to_dense()
        h = self.input_linear(x) * self.input_scale
        h = self.ball.expmap0(h)                     # initial hyperbolic node representation
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
        h_graph = karcher_flow(self.ball, h, T=10)   # graph-level representation on the manifold
        h_graph = self.ball.logmap0(h_graph)         # back to origin tangent space before classifier
        logits = self.classifier(h_graph)            # (1,)
        return logits.squeeze(-1)


# ============================================================
# 5. Training example (on randomly generated small graph data)
# ============================================================
def train_example():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # mock hyperparameters
    N = 20                # number of nodes
    in_dim = 16           # input feature dimension
    hidden_dim = 32       # hidden dimension
    num_relations = 3     # AST, CFG, DFG
    num_layers = 2
    c = 1.0
    lr = 1e-3
    epochs = 50

    # randomly generate graph data
    x = torch.randn(N, in_dim).to(device)
    # randomly generate edges
    E = 60
    src = torch.randint(0, N, (E,)).to(device)
    dst = torch.randint(0, N, (E,)).to(device)
    edge_index = torch.stack([src, dst], dim=0)
    edge_type = torch.randint(0, num_relations, (E,)).to(device)
    label = torch.tensor([1.0], device=device)  # binary classification label

    # initialize model
    model = CCMHGNN(in_dim, hidden_dim, num_relations,
                    num_layers, c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()

    # training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_type)
        loss = criterion(logits.unsqueeze(0), label)
        # L2 regularization is handled via weight_decay
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

    # inference
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_type)
        prob = torch.sigmoid(logits)
        print(f"predicted probability: {prob.item():.4f}")


# ============================================================
# 6. Devign code-graph dataset training pipeline (reads build_graphs.py output)
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name):
    """Select device; some hyperbolic ops may not be supported on MPS, fall back to CPU on probe failure."""
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # auto: CUDA first; Apple MPS has huge kernel-launch overhead for "many small graphs, one forward pass each"
    # (measured tens of times slower than CPU), so it is not enabled by default; force with --device mps if needed
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_graph(g, device):
    # CodeBERT features are stored as fp16 dense, uniformly cast to fp32; sparse TF-IDF stays sparse
    return (g["x"].float().to(device), g["edge_index"].to(device),
            g["edge_type"].to(device))


@torch.no_grad()
def collect_probs(model, graphs, indices, device):
    """Forward to collect sigmoid probabilities, labels, and mean BCE loss (unweighted)."""
    model.eval()
    probs, labels = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    for i in indices:
        g = graphs[i]
        x, ei, et = move_graph(g, device)
        logit = model(x, ei, et)
        y = torch.tensor([float(g["y"])], device=device)
        total_loss += criterion(logit.view(1), y).item()
        probs.append(torch.sigmoid(logit).cpu().item())
        labels.append(g["y"])
    return np.asarray(probs), np.asarray(labels), total_loss / max(len(indices), 1)


def metrics_at_threshold(labels, probs, threshold):
    """Full metric set at a deterministic threshold."""
    preds = (probs >= threshold).astype(int)
    return {
        "acc": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "pr_auc": average_precision_score(labels, probs)
        if labels.sum() > 0 else float("nan"),
        "roc_auc": roc_auc_score(labels, probs)
        if labels.sum() > 0 and labels.sum() < len(labels) else float("nan"),
        "threshold": threshold,
    }


def find_best_threshold(labels, probs):
    """Scan candidate quantiles of validation-set probabilities for the F1-optimal threshold (deterministic, never touches test set)."""
    cands = np.unique(np.clip(probs, 1e-6, 1 - 1e-6))
    # if too many candidates, evenly sample 200 to keep scan cost negligible
    if len(cands) > 200:
        cands = np.quantile(probs, np.linspace(0.01, 0.99, 200))
        cands = np.unique(cands)
    best_t, best_f1 = 0.5, -1.0
    for t in cands:
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def train_one_seed(data, args, device, seed):
    set_seed(seed)
    graphs = data["graphs"]
    splits = data["splits"]
    meta = data["meta"]
    tr, va, te = splits["train"], splits["val"], splits["test"]

    num_relations = meta["num_relations"]
    if getattr(args, "ablation", "full") == "single_rel":
        # heterogeneous-relation disentanglement ablation: merge all three edge types into one (same W/a params)
        graphs = [dict(g, edge_type=torch.zeros_like(g["edge_type"]))
                  for g in graphs]
        num_relations = 1

    n_tr_pos = sum(graphs[i]["y"] for i in tr)
    n_tr_neg = len(tr) - n_tr_pos
    if getattr(args, "balance_batches", False):
        # for naturally imbalanced full data: each epoch oversample the minority class (vulnerable)
        # with replacement to match the majority class, so batches are naturally balanced;
        # pos_weight set to 1 (same dynamics as v2 balanced data)
        pos_weight = torch.tensor([1.0], device=device)
        tr_pos = [i for i in tr if graphs[i]["y"] == 1]
        tr_neg = [i for i in tr if graphs[i]["y"] == 0]
        print(f"balanced-batch mode: each epoch {len(tr_neg)} negatives + "
              f"{len(tr_neg)} positives (positive oversampling {len(tr_neg)/max(len(tr_pos),1):.2f}x)")
    else:
        pos_weight = torch.tensor([n_tr_neg / max(n_tr_pos, 1)], device=device)

    ablation = getattr(args, "ablation", "full")
    if ablation == "euclidean":
        # Euclidean degeneration ablation: c=0 (maps degenerate to identity), keep adaptive modulation
        model = CCMHGNN(in_dim=meta["in_dim"],
                        hidden_dim=args.hidden_dim,
                        num_relations=num_relations,
                        num_layers=args.num_layers,
                        c=0.0, dropout=args.dropout,
                        use_modulation=True, mod_scale=1.0).to(device)
    else:
        model = CCMHGNN(in_dim=meta["in_dim"],
                        hidden_dim=args.hidden_dim,
                        num_relations=num_relations,
                        num_layers=args.num_layers,
                        c=args.curvature, dropout=args.dropout,
                        use_modulation=(ablation != "no_mod")).to(device)
    print(f"[seed {seed}] ablation: {ablation} | num_relations: {num_relations} | "
          f"use_modulation: {ablation != 'no_mod'}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # model-selection metric aligned with reported metric: val F1@best-val-threshold (F1 first)
    best_val_score, best_state, best_epoch, best_thr = -1.0, None, -1, 0.5
    rng = random.Random(seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if getattr(args, "balance_batches", False):
            order = tr_neg + [rng.choice(tr_pos) for _ in tr_neg]
            rng.shuffle(order)
        else:
            order = list(tr)
            rng.shuffle(order)
        t0 = time.time()
        running = 0.0
        for bstart in range(0, len(order), args.batch_size):
            batch = order[bstart:bstart + args.batch_size]
            logits, ys = [], []
            for i in batch:
                g = graphs[i]
                x, ei, et = move_graph(g, device)
                logits.append(model(x, ei, et))
                ys.append(float(g["y"]))
            logits = torch.stack(logits)
            ys = torch.tensor(ys, device=device)
            loss = criterion(logits, ys)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += loss.item() * len(batch)

        train_loss = running / len(order)
        vp, vy, val_loss = collect_probs(model, graphs, va, device)
        val_thr = find_best_threshold(vy, vp)
        val_tuned = metrics_at_threshold(vy, vp, val_thr)
        val_05 = metrics_at_threshold(vy, vp, 0.5)
        dt = time.time() - t0
        print(f"[seed {seed}] epoch {epoch:03d} | {dt:5.1f}s | "
              f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| f1@0.5 {val_05['f1']:.4f} | f1@tuned {val_tuned['f1']:.4f} "
              f"(t={val_thr:.3f}) | pr {val_tuned['pr_auc']:.4f}")

        if val_tuned["f1"] > best_val_score:
            best_val_score = val_tuned["f1"]
            best_epoch = epoch
            best_thr = val_thr
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            print(f"early stop at epoch {epoch} (best epoch {best_epoch}, "
                  f"best val F1@tuned {best_val_score:.4f})")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
        best_epoch = epoch
    model.load_state_dict(best_state)
    model.to(device)
    tp, ty, _ = collect_probs(model, graphs, te, device)
    test_05 = metrics_at_threshold(ty, tp, 0.5)
    test_tuned = metrics_at_threshold(ty, tp, best_thr)
    print(f"[seed {seed}] TEST @best-epoch {best_epoch}")
    print(f"  threshold 0.5  : acc {test_05['acc']:.4f} prec {test_05['precision']:.4f} "
          f"rec {test_05['recall']:.4f} f1 {test_05['f1']:.4f} pr {test_05['pr_auc']:.4f}")
    print(f"  threshold {best_thr:.3f}: acc {test_tuned['acc']:.4f} prec {test_tuned['precision']:.4f} "
          f"rec {test_tuned['recall']:.4f} f1 {test_tuned['f1']:.4f} pr {test_tuned['pr_auc']:.4f}")
    test_metrics = dict(test_tuned)
    test_metrics["f1_05"] = test_05["f1"]
    test_metrics["acc_05"] = test_05["acc"]

    # ---- balanced 50/50 protocol: comparable to v1/v2/v3 balanced test sets ----
    # subsets use a fixed random source (independent of training seed, consistent across experiments);
    # threshold is searched only on the balanced validation set
    brng = random.Random(999)

    def _balanced(idxs):
        pos = [i for i in idxs if graphs[i]["y"] == 1]
        neg = [i for i in idxs if graphs[i]["y"] == 0]
        k = min(len(pos), len(neg))
        return brng.sample(pos, k) + brng.sample(neg, k)

    bv, bte = _balanced(va), _balanced(te)
    bvp, bvy, _ = collect_probs(model, graphs, bv, device)
    bthr = find_best_threshold(bvy, bvp)
    btp, bty, _ = collect_probs(model, graphs, bte, device)
    bm = metrics_at_threshold(bty, btp, bthr)
    print(f"  balanced protocol (val {len(bv)}/test {len(bte)}): threshold {bthr:.3f} "
          f"f1 {bm['f1']:.4f} pr {bm['pr_auc']:.4f} roc {bm['roc_auc']:.4f}")
    test_metrics["f1_bal"] = bm["f1"]
    test_metrics["pr_bal"] = bm["pr_auc"]
    test_metrics["roc_bal"] = bm["roc_auc"]
    test_metrics["thr_bal"] = bthr

    if getattr(args, "results_csv", ""):
        new = not os.path.exists(args.results_csv)
        with open(args.results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["name", "ablation", "seed", "acc", "precision",
                            "recall", "f1", "pr_auc", "roc_auc", "threshold",
                            "f1_at_0.5", "acc_at_0.5",
                            "f1_balanced", "pr_balanced", "roc_balanced",
                            "threshold_balanced", "best_epoch"])
            w.writerow([args.name or ablation, ablation, seed,
                        f"{test_tuned['acc']:.4f}",
                        f"{test_tuned['precision']:.4f}",
                        f"{test_tuned['recall']:.4f}",
                        f"{test_tuned['f1']:.4f}",
                        f"{test_tuned['pr_auc']:.4f}",
                        f"{test_tuned['roc_auc']:.4f}",
                        f"{best_thr:.4f}",
                        f"{test_05['f1']:.4f}",
                        f"{test_05['acc']:.4f}",
                        f"{bm['f1']:.4f}",
                        f"{bm['pr_auc']:.4f}",
                        f"{bm['roc_auc']:.4f}",
                        f"{bthr:.4f}",
                        best_epoch])
    return test_metrics


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Train CCM-HGNN on Devign code graphs (AST/CFG/DFG generated by joern)")
    ap.add_argument("--data", default=os.path.join(base, "workspace", "graphs.pt"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--curvature", type=float, default=1.0, help="global curvature c")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=1, help="number of independent random seeds (paper uses 5)")
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--ablation", default="full",
                    choices=["full", "euclidean", "no_mod", "single_rel"],
                    help="ablation: full=complete model; euclidean=Euclidean degeneration (c=0); "
                         "no_mod=remove node topology modulation (S≡1); single_rel=merge all relations into one")
    ap.add_argument("--threads", type=int, default=0, help="limit CPU threads when > 0")
    ap.add_argument("--balance-batches", action="store_true",
                    help="for naturally imbalanced data, oversample the minority class each epoch "
                         "to balance batch class ratios (pos_weight auto-set to 1)")
    ap.add_argument("--results-csv", default="",
                    help="if set, append each seed's test results to this CSV")
    ap.add_argument("--name", default="", help="experiment name in results CSV")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    print(f"loading code graph data: {args.data}")
    data = torch.load(args.data, map_location="cpu")
    meta = data["meta"]
    print(f"num graphs: {meta['num_graphs']} | in_dim={meta['in_dim']} | "
          f"relations: {meta['relation_names']}")
    for k, v in data["splits"].items():
        n_pos = sum(data["graphs"][i]["y"] for i in v)
        print(f"  {k:5s}: {len(v):5d} (vulnerable {n_pos} / non-vulnerable {len(v) - n_pos})")

    device = pick_device(args.device)
    print(f"device: {device}")

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    results = [train_one_seed(data, args, device, s) for s in seeds]

    print("\n================ Final results (threshold optimized on validation set) ================")
    for s, r in zip(seeds, results):
        print(f"seed {s}: f1 {r['f1']:.4f} (f1@0.5 {r['f1_05']:.4f}) | "
              f"acc {r['acc']:.4f} prec {r['precision']:.4f} rec {r['recall']:.4f} "
              f"pr_auc {r['pr_auc']:.4f} roc {r.get('roc_auc', float('nan')):.4f} "
              f"t={r['threshold']:.3f}")
        if "f1_bal" in r:
            print(f"        balanced 50/50: f1 {r['f1_bal']:.4f} pr {r['pr_bal']:.4f} "
                  f"roc {r['roc_bal']:.4f} t={r['thr_bal']:.3f}")
    if len(results) > 1:
        print("\nmean ± std:")
        keys = ["acc", "precision", "recall", "f1", "pr_auc"]
        if "roc_auc" in results[0]:
            keys.append("roc_auc")
        for key in keys:
            vals = np.array([r[key] for r in results])
            print(f"  {key:9s}: {vals.mean()*100:.2f} ± {vals.std()*100:.2f}")
        if "f1_bal" in results[0]:
            print("  balanced 50/50:")
            for key, name in [("f1_bal", "f1"), ("pr_bal", "pr_auc"),
                              ("roc_bal", "roc_auc")]:
                vals = np.array([r[key] for r in results])
                print(f"  {name:9s}: {vals.mean()*100:.2f} ± {vals.std()*100:.2f}")


if __name__ == "__main__":
    main()