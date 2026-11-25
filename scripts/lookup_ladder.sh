#!/usr/bin/env bash
# Hop ladder: does one more loop buy one more lookup hop?
# Lookup-only data (1-3 hops); arms differ only in loop depth. Backbone states are cached,
# so each run trains the head alone.
#
#   LAYERS=2 SEEDS="0 1 2" bash scripts/lookup_ladder.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-HuggingFaceTB/SmolLM2-360M-Instruct}
DATA=${DATA:-data/lookup}
RUNS=${RUNS:-runs/lookup}
LAYERS=${LAYERS:-2}
STEPS=${STEPS:-4000}
SEEDS=${SEEDS:-0}
TIERS=${TIERS:-"1x 2x 3x"}
AMP=${AMP:-none}
GATE=${GATE:-matrix}
CKPT=${CKPT:-head_best.pt}

if [ ! -f "$DATA/train.jsonl" ]; then
  python3 -m lth.data --out "$DATA" --train 8000 --val 300 --test 600 --hops 1-3 --tasks lookup_chain --seed 1
fi

heads=()
for seed in $SEEDS; do
  for tier in $TIERS; do
    name="$tier-L$LAYERS-s$seed"
    if [ ! -f "$RUNS/$name/head.pt" ]; then
      python3 -m lth.train --model "$MODEL" --dtype fp32 --amp "$AMP" --gate "$GATE" --data "$DATA" --out "$RUNS/$name" \
        --tier "$tier" --num-layers "$LAYERS" --steps "$STEPS" --batch-size 16 --seed "$seed" \
        --eval-every 1000 --eval-examples 96 --log-every 500 --cache-hidden "$RUNS/hidden_cache"
    fi
    heads+=(--head "$name=$RUNS/$name/$CKPT")
  done
done

first_seed=${SEEDS%% *}
python3 -m lth.evaluate --model "$MODEL" --dtype fp32 --amp "$AMP" --data "test=$DATA/test.jsonl" \
  "${heads[@]}" --control "1x-L$LAYERS-s$first_seed" --depths 0,1,2,3 --batch-size 32 --max-new-tokens 24 \
  --latency-prompts 0 --no-regression --out "$RUNS/report-L$LAYERS"
