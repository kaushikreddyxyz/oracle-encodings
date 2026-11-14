#!/bin/bash
# Mac-side: one line per pod.
for k in 1 2 3 4 5 6; do
  echo "p$k: $(ssh -o ConnectTimeout=15 runpod-onion-p$k 'tail -n 1 /workspace/runs/STATUS | cut -c1-70; f=$(ls -t /workspace/runs/train-*.log 2>/dev/null | head -1); [ -n "$f" ] && { grep -oE "iter [0-9]+/[0-9]+ loss=[0-9.]+" $f | tail -1; grep -m1 -E "Traceback|Error" $f; }; grep -h -m1 -E "Traceback|Error" /workspace/runs/pretok*.log /workspace/runs/eval-*.log /workspace/runs/dist-*.log /workspace/runs/local-*.log 2>/dev/null' 2>&1 | tr '\n' ' ')"
done
