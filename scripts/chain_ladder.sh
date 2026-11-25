#!/usr/bin/env bash
# Chain-supervised loops: loop r is trained to hold the answer after hop r.
# Single-token name chains, 1-3 hops. Arms (same data, steps, batch size, seed):
#   1x:plain  single pass, final answer only        (control)
#   3x:plain  three loops, final answer only        (does looping help on its own?)
#   2x:chain  two loops, per-loop supervision       (should solve up to 2 hops)
#   3x:chain  three loops, per-loop supervision     (should solve up to 3 hops)
# Then: generated accuracy per hop (lth.evaluate) and per-loop decoding (lth.probe).
#
#   LAYERS=2 bash scripts/chain_ladder.sh
#   TAP=8 bash scripts/chain_ladder.sh        # the head reads backbone layer 8 instead of the final layer
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-HuggingFaceTB/SmolLM2-360M-Instruct}
DATA=${DATA:-data/names}
RUNS=${RUNS:-runs/names}
LAYERS=${LAYERS:-2}
HEAD_DIM=${HEAD_DIM:-256}
HEADS=${HEADS:-8}
STEPS=${STEPS:-5000}
BATCH=${BATCH:-32}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-5e-4}
SCHEDULE=${SCHEDULE:-wsd}
SEED=${SEED:-0}
ARMS=${ARMS:-"1x:plain 3x:plain 2x:chain 3x:chain"}
TAP=${TAP:-8}
AMP=${AMP:-none}
GATE=${GATE:-matrix}
CKPT=${CKPT:-head_best.pt}

suffix=""
tapflag=()
if [ -n "$TAP" ]; then
  suffix="-tap$TAP"
  tapflag=(--tap-layer "$TAP")
fi
dimflag=()
if [ -n "$HEAD_DIM" ]; then
  dimflag=(--head-dim "$HEAD_DIM")
fi

if [ ! -f "$DATA/train.jsonl" ]; then
  python3 -m lth.data --out "$DATA" --train 16000 --val 300 --test 600 --hops 1-3 --tasks name_chain --seed 2
fi

heads=()
for arm in $ARMS; do
  tier=${arm%%:*}
  mode=${arm##*:}
  name="$tier-$mode-L$LAYERS$suffix-s$SEED"
  extra=()
  if [ "$mode" = chain ]; then extra=(--chain-supervision); fi
  if [ ! -f "$RUNS/$name/head.pt" ]; then
    python3 -m lth.train --model "$MODEL" --dtype fp32 --amp "$AMP" --gate "$GATE" --data "$DATA" --out "$RUNS/$name" \
      --tier "$tier" --num-layers "$LAYERS" --num-heads "$HEADS" "${dimflag[@]}" \
      --steps "$STEPS" --batch-size "$BATCH" --grad-accum "$GRAD_ACCUM" --lr "$LR" --schedule "$SCHEDULE" --seed "$SEED" \
      --eval-every 1000 --eval-examples 96 --log-every 500 --cache-hidden "$RUNS/hidden_cache$suffix" "${tapflag[@]}" "${extra[@]}"
  fi
  heads+=(--head "$name=$RUNS/$name/$CKPT")
done

tag="L$LAYERS$suffix-s$SEED"
python3 -m lth.evaluate --model "$MODEL" --dtype fp32 --amp "$AMP" --data "test=$DATA/test.jsonl" \
  "${heads[@]}" --control "1x-plain-$tag" --depths 0,1,2,3 --batch-size 32 --max-new-tokens 16 \
  --latency-prompts 0 --no-regression --out "$RUNS/report-$tag"
python3 -m lth.probe --model "$MODEL" --dtype fp32 --amp "$AMP" --data "$DATA/test.jsonl" \
  "${heads[@]}" --batch-size 32 --out "$RUNS/probe-$tag"
