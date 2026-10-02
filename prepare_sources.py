#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_sources.py
------------------
Read the three Devign data files in the directory:
  - function.json      : [{project, commit_id, target, func}]
  - vulnerables.json   : [{code, hash, project, size}]  (label=1)
  - non-vulnerables.json: [{code, hash, project, size}] (label=0)

Merge and deduplicate (whitespace-normalize then dedupe by full code text;
labels are unioned: a sample is positive if it ever appears as vulnerable),
then write each function as a standalone .c source file for fuzzy parsing by joern.

Output (defaults to workspace/, does not modify any original data file):
  workspace/src/chunk_xxx/000123.c
  workspace/index.csv  : id,path,label,source,project,length
"""
import argparse
import csv
import json
import os
import re

WS = re.compile(r"\s+")


def normalize(code: str) -> str:
    return WS.sub(" ", code).strip()


def load_all(root: str):
    """Return [(raw_code, label, source, project), ...] (not deduplicated)."""
    items = []

    with open(os.path.join(root, "function.json"), "r", encoding="utf-8") as f:
        for d in json.load(f):
            items.append((d["func"], int(d["target"]),
                          "function", d.get("project", "")))

    with open(os.path.join(root, "vulnerables.json"), "r", encoding="utf-8") as f:
        for d in json.load(f):
            items.append((d["code"], 1, "vulnerables", d.get("project", "")))

    with open(os.path.join(root, "non-vulnerables.json"), "r", encoding="utf-8") as f:
        for d in json.load(f):
            items.append((d["code"], 0, "non-vulnerables", d.get("project", "")))

    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--workspace", default=None,
                    help="output directory, defaults to <root>/workspace")
    ap.add_argument("--chunk-size", type=int, default=2000,
                    help="number of source files per joern parsing chunk directory")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=12000)
    args = ap.parse_args()

    workspace = args.workspace or os.path.join(args.root, "workspace")
    src_root = os.path.join(workspace, "src")
    os.makedirs(src_root, exist_ok=True)

    raw = load_all(args.root)
    print(f"total raw records: {len(raw)}")

    # dedupe: key=normalized code; labels are unioned; source/project keep first seen
    merged = {}
    skipped_short = skipped_long = 0
    for code, label, source, project in raw:
        key = normalize(code)
        if len(key) < args.min_chars:
            skipped_short += 1
            continue
        if len(key) > args.max_chars:
            skipped_long += 1
            continue
        if key not in merged:
            merged[key] = {
                "code": code,
                "label": label,
                "source": source,
                "project": project,
            }
        else:
            merged[key]["label"] = max(merged[key]["label"], label)

    print(f"filtered out too-short(<{args.min_chars}): {skipped_short}, "
          f"too-long(>{args.max_chars}): {skipped_long}")
    n_pos = sum(v["label"] for v in merged.values())
    print(f"unique functions after dedup: {len(merged)} "
          f"(vulnerable {n_pos} / non-vulnerable {len(merged) - n_pos})")

    # write in fixed order (by source then key) for reproducibility
    ordered = sorted(merged.items(), key=lambda kv: (kv[1]["source"], kv[0]))

    index_path = os.path.join(workspace, "index.csv")
    with open(index_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["id", "path", "label", "source", "project", "length"])
        for i, (key, info) in enumerate(ordered):
            chunk = i // args.chunk_size
            cdir = os.path.join(src_root, f"chunk_{chunk:03d}")
            os.makedirs(cdir, exist_ok=True)
            fid = f"{i:06d}"
            rel = os.path.join("src", f"chunk_{chunk:03d}", fid + ".c")
            fpath = os.path.join(workspace, rel)
            with open(fpath, "w", encoding="utf-8") as cf:
                cf.write(info["code"])
                if not info["code"].endswith("\n"):
                    cf.write("\n")
            writer.writerow([fid, rel, info["label"], info["source"],
                             info["project"], len(key)])

    n_chunks = (len(ordered) + args.chunk_size - 1) // args.chunk_size
    print(f"wrote {len(ordered)} source files in {n_chunks} chunks -> {src_root}")
    print(f"index file: {index_path}")


if __name__ == "__main__":
    main()
