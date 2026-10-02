#!/usr/bin/env bash
# 分块运行 joern：parse -> 导出 AST/CFG/DFG 图（断点续跑，已完成的块跳过）
set -u
cd "$(dirname "$0")"

JOERN_PARSE=./tools/joern/joern-cli/joern-parse
JOERN=./tools/joern/joern-cli
SCRIPT=/Users/choggle/Desktop/ccm/dump_graphs.sc
# 将 ammonite/joern 的用户目录重定向到工作区内（避免沙箱/权限问题）
export JOERN_HOME=/Users/choggle/Desktop/ccm/workspace/.ammhome
mkdir -p "$JOERN_HOME"

mkdir -p workspace/cpg workspace/dump workspace/logs

for src in workspace/src/chunk_*; do
  name=$(basename "$src")
  bin=workspace/cpg/${name}.bin
  out=workspace/dump/${name}.jsonl
  plog=workspace/logs/parse_${name}.log
  dlog=workspace/logs/dump_${name}.log

  if [ -s "$out" ]; then
    echo "[skip] $name already dumped ($(wc -l < "$out") lines)"
    continue
  fi

  echo "[$(date +%H:%M:%S)] parse $name ..."
  if [ ! -s "$bin" ]; then
    $JOERN_PARSE -J-Duser.home="$JOERN_HOME" "$src" -o "$bin" > "$plog" 2>&1
    if [ $? -ne 0 ]; then
      echo "  [ERROR] parse failed, see $plog"
      continue
    fi
  else
    echo "  reuse existing $bin"
  fi

  echo "[$(date +%H:%M:%S)] dump  $name ..."
  "$JOERN/joern" -J-Duser.home="$JOERN_HOME" --script "$SCRIPT" \
    --params "cpgFile=/Users/choggle/Desktop/ccm/$bin,outFile=/Users/choggle/Desktop/ccm/$out" \
    > "$dlog" 2>&1
  if [ $? -ne 0 ]; then
    echo "  [ERROR] dump failed, see $dlog"
    continue
  fi
  echo "  -> $(wc -l < "$out") graphs"
done
echo "[$(date +%H:%M:%S)] ALL DONE"
