# CCM-HGNN: Vulnerability Detection with Node Topology-Aware Modulation

On the Devign dataset, C functions are parsed by joern into Code Property Graphs (AST / CFG / DFG, three heterogeneous relation edge types), and the **CCM-HGNN** (Curvature-aware Hyperbolic Graph Neural Network) model performs binary classification (vulnerable / non-vulnerable) on each function.

- **Hyperbolic geometry**: Node embeddings live on the Poincaré Ball (c=1.0). Aggregation during message passing is performed via Möbius addition and exponential/logarithmic maps, giving stronger expressiveness for hierarchical structures in hyperbolic space.
- **Node topology-aware modulation**: Messages are adaptively scaled according to each node's local topology (degree, relation composition), suppressing noise propagation.
- **Heterogeneous relation disentanglement**: AST / CFG / DFG edges each use an independent relation weight matrix, carrying syntactic structure, control flow, and data flow information respectively.

## Environment

- Python 3.8+, `torch`, `scikit-learn`, `numpy`
- Steps 1 require [joern](https://joern.io)
- The P3 feature mode additionally requires `transformers` (CodeBERT encoding, can run on CPU)

```bash
pip install torch scikit-learn numpy   # P3: pip install transformers
```

## Usage

### Step 1: joern Parsing and Graph Export

```bash
bash run_joern.sh
# For each chunk: joern-parse generates CPG -> dump_graphs.sc exports AST/CFG/DFG
# Outputs workspace/cpg/*.bin and workspace/dump/*.jsonl (one function's nodes+edges per line)
```

### Step 2: Build Tensor Dataset

```bash
# v2 protocol: 4,000 balanced samples (default)
python3 build_graphs.py --out workspace/graphs.pt

# P2 protocol: full natural distribution (imbalanced, 28.6% positive)
python3 build_graphs.py --max-samples 0 --out workspace/graphs_full.pt

# Optional: CodeBERT features (requires transformers)
python3 build_graphs.py --feature-mode codebert --out workspace/graphs_codebert.pt
```

### Step 3: Train and Evaluate

```bash
# v2 reproduction (4,000 balanced graphs, 3 seeds)
python3 ccm.py --data workspace/graphs.pt --seeds 3 --epochs 100 --patience 10 \
  --batch-size 32 --lr 3e-4 --weight-decay 5e-4 --hidden-dim 64 --num-layers 2 \
  --device cpu --results-csv workspace/results_v2.csv --name v2

# P2 full-data reproduction (natural distribution + class-balanced batches)
python3 ccm.py --data workspace/graphs_full.pt --seeds 3 --epochs 10 --patience 3 \
  --balance-batches --threads 4 --device cpu \
  --results-csv workspace/results_v4.csv --name p2_full
```
