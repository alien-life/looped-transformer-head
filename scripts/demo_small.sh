#!/usr/bin/env bash
# Small-scale end-to-end run of the whole pipeline on an 8 GB GPU.
# Four arms with an identical budget (same data, steps, batch size, seed), each from scratch:
# the 1x single-pass control and the 5x / 10x / 50x tiers. Then one depth sweep over all of them.
#
#   bash scripts/demo_small.sh            # ~1 hour on an RTX 2070
#   STEPS=300 LIMIT=100 bash scripts/demo_small.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-HuggingFaceTB/SmolLM2-360M-Instruct}
DATA=${DATA:-data/abc}
RUNS=${RUNS:-runs/demo}
STEPS=${STEPS:-1200}
BATCH=${BATCH:-8}
LIMIT=${LIMIT:-300}
AMP=${AMP:-none}
GATE=${GATE:-matrix}
CKPT=${CKPT:-head_best.pt}

if [ ! -f "$DATA/train.jsonl" ]; then
  python3 -m lth.data --out "$DATA" --train 8000 --val 200 --test 600 --hops 2-4 --ood-hops 5-6 --ood 300
fi

for tier in 1x 5x 10x 50x; do
  if [ ! -f "$RUNS/$tier/head.pt" ]; then
    python3 -m lth.train --model "$MODEL" --dtype fp32 --amp "$AMP" --gate "$GATE" --data "$DATA" --out "$RUNS/$tier" \
      --tier "$tier" --steps "$STEPS" --batch-size "$BATCH" --eval-every 400 --eval-examples 64 --log-every 50
  fi
done

python3 -m lth.evaluate --model "$MODEL" --dtype fp32 --amp "$AMP" \
  --data "test=$DATA/test.jsonl" --data "ood=$DATA/test_ood.jsonl" --limit "$LIMIT" \
  --head "1x=$RUNS/1x/$CKPT" --head "5x=$RUNS/5x/$CKPT" \
  --head "10x=$RUNS/10x/$CKPT" --head "50x=$RUNS/50x/$CKPT" \
  --control 1x --depths 0,1,5,10,25,50 --batch-size 16 --out "$RUNS/report"
